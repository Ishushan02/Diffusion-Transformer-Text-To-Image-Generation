import torch
import torch.nn as nn
import math
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from torchvision import transforms
from diffusers import AutoencoderDC
from diffusers import DDPMScheduler, DDIMScheduler


if(torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



class TextEmbedding(nn.Module):
    def __init__(self, maxSequenceLength = 1024, embeddingDimension = 768, gptNeoModel = "EleutherAI/gpt-neo-125M", 
                 t5Model = "t5-base", clipModel = "openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.gptNeoTokenizer = AutoTokenizer.from_pretrained(gptNeoModel)
        self.gptNeoTokenizer.pad_token = self.gptNeoTokenizer.eos_token
        self.gptNeoModel = AutoModel.from_pretrained(gptNeoModel)
        self.t5Tokenizer = T5Tokenizer.from_pretrained(t5Model)
        self.t5Model = T5EncoderModel.from_pretrained(t5Model)
        self.clipTokenizer = CLIPTokenizer.from_pretrained(clipModel)
        self.clipModel = CLIPModel.from_pretrained(clipModel)
        self.maxSeqLen = maxSequenceLength
        self.embedDimension = embeddingDimension
        self.text_projection = nn.Linear(self.embedDimension, self.embedDimension)

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        elif isinstance(text, (list, tuple)):
            pass
        else:
            raise ValueError(f"Give string or list of strings, recieved this {type(text)}")
            
        batchSize = len(text)

        gptneoTokens = self.gptNeoTokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.maxSeqLen)
        t5Tokens = self.t5Tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.maxSeqLen)
        clipTokens = self.clipTokenizer(text, return_tensors="pt", truncation=True, padding='max_length')

        device = next(self.parameters()).device

        gptneoTokens = {k: v.to(device) for k, v in gptneoTokens.items()}
        t5Tokens = {k: v.to(device) for k, v in t5Tokens.items()}
        clipTokens = {k: v.to(device) for k, v in clipTokens.items()}

        
        with torch.no_grad():
            output = self.gptNeoModel(**gptneoTokens)
            gptNeoembeddings = output.last_hidden_state

            output = self.t5Model(**t5Tokens)
            t5embeddings = output.last_hidden_state

            output = self.clipModel.text_model(**clipTokens)
            clipembeddings = output.last_hidden_state
            clipembeddings = Fn.pad(clipembeddings, (0, self.embedDimension-512))

        
        concatenatedtextEmbeddings = torch.concat([gptNeoembeddings, t5embeddings, clipembeddings], dim=1)
        textEmbeddings = self.text_projection(concatenatedtextEmbeddings)

        return textEmbeddings


# text = ["Generate an Image of a Dog Eating", "Generate an Image of a Cat Eating"]
# textModel = TextEmbedding()
# out = textModel(text)
# out.shape



class TimeEmbedding(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.linear1 = nn.Linear(embedDimension, 4 * embedDimension)
        self.silu = nn.SiLU()
        self.outlayer = nn.Linear(4 * embedDimension, embedDimension)



    def forward(self, t):

        half = self.embedDimension// 2
        exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        freq = torch.exp(exponent.to(device))

        timedimMap = t.float().unsqueeze(0) * freq[None]
        sinusoidal = torch.cat([torch.cos(timedimMap), torch.sin(timedimMap)], dim = -1)

        sinusoidal = self.linear1(sinusoidal)
        sinusoidal = self.silu(sinusoidal)
        out = self.outlayer(sinusoidal)

        return out

# tEmbed = TimeEmbedding(768)
# tEmbed.to(device)
# time = torch.tensor([5]).to(device)
# out = tEmbed(time)
# out.shape


class PatchEmbedding(nn.Module):
    def __init__(self, imageSize, patchSize, inChannels, embedDimension):
        super().__init__()
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.embedDimension = embedDimension
        self.imageSize = imageSize

        self.patches = imageSize//patchSize * imageSize//patchSize

        self.encode = nn.Conv2d(in_channels = inChannels, out_channels = embedDimension, kernel_size = patchSize, stride = patchSize, bias = True)
        self.decode = nn.ConvTranspose2d(in_channels=embedDimension, out_channels=inChannels, kernel_size=patchSize, stride=patchSize, bias=True)
        self.positionalEmbedding = nn.Parameter(torch.zeros(1, self.patches, embedDimension))
        nn.init.trunc_normal_(self.positionalEmbedding, std=0.02)

    def unPatchify(self, x):
        batchSize, NPatches, EmbedDim = x.shape
        patchPerDim = self.imageSize // self.patchSize
        x = x.transpose(1, 2).reshape(batchSize, EmbedDim, patchPerDim, patchPerDim)
        out = self.decode(x)
        return out


    def forward(self, latentImage):

        allPatch = self.encode(latentImage)
        # print(allPatch.shape)
        flattened = allPatch.flatten(2).transpose(1, 2)
        # print(flattened.shape, self.positionalEmbedding.shape)
        out = flattened + self.positionalEmbedding
        return out

# latent = torch.randn(128, 8, 8).unsqueeze(0)
# pEmbed = PatchEmbedding(imageSize = 8, patchSize = 2, inChannels = 128, embedDimension = 768)
# out = pEmbed(latent)
# unpatched = pEmbed.unPatchify(out)
# out.shape, unpatched.shape
# (batchSize, totalPatches, embeddinDimension)




class ImageInputToDiT(nn.Module):
    def __init__(self, latentSize, latentChannels, embedDimension, patchSize, totalTimestamps = 1000, beta_schedule = "squaredcos_cap_v2", modelName="mit-han-lab/dc-ae-f64c128-in-1.0-diffusers"):
        super().__init__()

        self.dc_ae = AutoencoderDC.from_pretrained(modelName, torch_dtype=torch.float32)
        self.noiseScheduler = DDPMScheduler(num_train_timesteps=totalTimestamps, beta_schedule=beta_schedule)
        self.patchEmbedding = PatchEmbedding(imageSize = latentSize, patchSize = patchSize, inChannels = latentChannels, embedDimension = embedDimension)

    def forward(self, x, timestamp):

        with torch.no_grad():
            latents = self.dc_ae.encode(x).latent

        noise = torch.randn_like(latents)
        alphaT = self.noiseScheduler.alphas_cumprod[timestamp].view(1, 1, 1, 1)
        noisyLatents = torch.sqrt(alphaT) * latents + torch.sqrt(1 - alphaT) * noise
        
        patches = self.patchEmbedding(noisyLatents)

        return patches, noise


# preprocess = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),                 
#     transforms.Normalize([0.5]*3, [0.5]*3)])

# image = Image.open("Images/testImage.jpg").convert("RGB")
# plt.imshow(image)
# image_tensor = preprocess(image).unsqueeze(0)
# # image_tensor = torch.randn(1, 3, 512, 512)

# imgenc = ImageInputToDiT(latentSize = 8, latentChannels = 128, embedDimension = 768, patchSize = 2)
# patches, noise = imgenc(image_tensor, 10)
# patches.shape, noise.shape




class AdaptiveLayerNorm(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedDimension, 6 * embedDimension)
        )
        self.scaleShiftParameters = nn.Parameter(torch.zeros(6, embedDimension))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)      
    
    def forward(self, t):
        batchSize, _ = t.shape
        t = self.adaLN(t)
        t = t.reshape(batchSize, 6, -1)
        gamma_msa, beta_msa, alpha_msa, gamma_mlp, beta_mlp, alpha_mlp = (
            (self.scaleShiftParameters[None] + t).chunk(6, dim = 1)
        )
        gamma_msa = gamma_msa.squeeze(1)
        beta_msa = beta_msa.squeeze(1)
        alpha_msa = alpha_msa.squeeze(1)
        gamma_mlp = gamma_mlp.squeeze(1)
        beta_mlp = beta_mlp.squeeze(1)
        alpha_mlp = alpha_mlp.squeeze(1)
        return gamma_msa, beta_msa, alpha_msa, gamma_mlp, beta_mlp, alpha_mlp
    
# adaNorm = AdaptiveLayerNorm(embedDimension)
# g1, b1, a1, g2, b2, a2 = adaNorm(tout)
# g1.shape, b1.shape, a1.shape, g2.shape, b2.shape, a2.shape




def shiftModulate(x, scale, shift):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class ScaleShiftBlock(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.norm = nn.LayerNorm(embedDimension, elementwise_affine=False, eps=1e-6)

    def forward(self, x, beta, gamma):
        B, N, W = x.shape
        x_norm = self.norm(x)
        out = shiftModulate(x_norm, gamma, beta)
        return out
    
# patchify_latents = torch.randn(1, 16, 768)
# scShft = ScaleShiftBlock(embedDimension)
# out = scShft(patchify_latents, g1, b1)
# out.shape





def scaleModulate(x, scale):
    return x * (1 + scale.unsqueeze(1))

class ScaleBlock(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()
        self.embedDimension = embedDimension
        self.norm = nn.LayerNorm(embedDimension, elementwise_affine=False, eps=1e-6)

    def forward(self, x, alpha):
        B, N, W = x.shape
        x_norm = self.norm(x)
        out = scaleModulate(x_norm, alpha)
        return out
    
# patchify_latents = torch.randn(1, 16, 768)
# scShft = ScaleBlock(embedDimension)
# out = scShft(patchify_latents, a1)
# out.shape

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedDimension, numHeads, dropout = 0.2):
        super().__init__()

        assert embedDimension%numHeads == 0, "Embedding Dimension is Not Divisible By NumHeads"
        self.embedDimension = embedDimension
        self.numHeads = numHeads
        self.headDim = embedDimension//numHeads

        self.queryKeyValue = nn.Linear(embedDimension, embedDimension * 3, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = self.headDim ** -0.5 
        self.outProjection = nn.Linear(embedDimension, embedDimension)

    def forward(self, x):
        BatchSize, N, EmbedDim = x.shape

        qkv = self.queryKeyValue(x)
        qkv = qkv.reshape(BatchSize, N, 3, self.numHeads, EmbedDim // self.numHeads)
        q, k, v = qkv.unbind(2)
        attentionScore = (q @ k.transpose(-2, -1)) * self.scale
        attn = attentionScore.softmax(dim=-1)
        out = attn @ v 
        out = out.transpose(1, 2).reshape(BatchSize, N, EmbedDim)
        out = self.outProjection(out)
        out = self.drop(out)
        return out
    
# input = torch.randn(1, 16, 768)
# msa = MultiHeadSelfAttention(embedDimension=768, numHeads=8)
# out = msa(input)
# out.shape




class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embedDimension, numHeads, dropout = 0.2):
        super().__init__()

        assert embedDimension%numHeads == 0, "Embedding Dimension is Not Divisible By NumHeads"
        self.embedDimension = embedDimension
        self.numHeads = numHeads
        self.headDim = embedDimension//numHeads

        self.q = nn.Linear(embedDimension, embedDimension)
        self.k = nn.Linear(embedDimension, embedDimension)
        self.v = nn.Linear(embedDimension, embedDimension)
        self.kv = nn.Linear(embedDimension, 2 * embedDimension)

        self.outProjection = nn.Linear(embedDimension, embedDimension)
        self.drop = nn.Dropout(dropout)
        self.scale = self.headDim ** -0.5 

    def forward(self, x, textCondition):
        batch, Nimg, embedDim = x.shape
        batch, Ntext, embedDim = textCondition.shape

        q = self.q(x)
        k = self.k(textCondition)
        v = self.v(textCondition)

        q = q.view(batch, Nimg, self.numHeads, self.headDim).transpose(1, 2)
        k = k.view(batch, Ntext, self.numHeads, self.headDim).transpose(1, 2)
        v = v.view(batch, Ntext, self.numHeads, self.headDim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = Fn.softmax(attn_scores, dim=-1)
       
        out = attn_probs @ v 
        out = out.transpose(1, 2).contiguous().view(batch, Nimg, embedDim)
        out = self.outProjection(out)
        out = self.drop(out)
        
        return out


# text = ["Generate an Image of a Dog Eating"]
# textModel = TextEmbedding()
# textembed = textModel(text)
# xinp = torch.randn(1, 16, 768)

# mca = MultiHeadCrossAttention(embedDimension, numHeads=8)
# out = mca(xinp, textembed)
# out.shape




class FeedForwardBlock(nn.Module):
    def __init__(self, embedDimension):
        super().__init__()

        self.linear1 = nn.Linear(embedDimension, embedDimension * 4)
        self.linear2 = nn.Linear(embedDimension * 4, embedDimension)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x
    
# latents = torch.randn(1, 16, 768)
# ff = FeedForwardBlock(embedDimension)
# out = ff(latents)
# out.shape




class DiTModule(nn.Module):
    def __init__(self, embedDimension, numHeads, dropout = 0.2):
        super().__init__()
        self.embedDimension = embedDimension
        self.numHeads = numHeads

        self.scaleShift1 = ScaleShiftBlock(embedDimension)
        self.multiHeadselfAtten = MultiHeadSelfAttention(embedDimension, numHeads, dropout)
        self.scale1 = ScaleBlock(embedDimension)

        self.multiHeadcrossAtten = MultiHeadCrossAttention(embedDimension, numHeads, dropout)

        self.scaleShift2 = ScaleShiftBlock(embedDimension)
        self.pointwiseFeedForward = FeedForwardBlock(embedDimension)
        self.scale2 = ScaleBlock(embedDimension)

    def forward(self, imageLatents, textEmbeddings, sharedParameters):
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = sharedParameters

        x = imageLatents
        scaleShiftOut1 = self.scaleShift1(x, gamma1, beta1)
        selfAttnOut = self.multiHeadselfAtten(scaleShiftOut1)
        scaleOut1 = self.scale1(selfAttnOut, alpha1)

        x =  x + scaleOut1

        y = self.multiHeadcrossAtten(x, textEmbeddings)

        y = y + x

        scaleShiftOut2 = self.scaleShift2(y, gamma2, beta2)
        mlpOut = self.pointwiseFeedForward(scaleShiftOut2)
        scaleOut2 = self.scale2(mlpOut, alpha2)

        z = y + scaleOut2

        return z
    

# text = ["Generate an Image of a Dog Eating"]
# textModel = TextEmbedding()
# textembed = textModel(text)
# noisedlatents = torch.randn(1, 16, 768)
# adaNorm = AdaptiveLayerNorm(embedDimension)
# sharedParameters = adaNorm(tout)

# dit = DiTModule(embedDimension, numHeads=12)

# out = dit(noisedlatents, textembed, sharedParameters)
# out.shape






class NDiTModule(nn.Module):
    def __init__(self, blocks, embedDimension, numHeads, dropout = 0.2):
        super().__init__()

        self.adaNorm = AdaptiveLayerNorm(embedDimension)
        
        self.eachBlocks = nn.ModuleList([
            DiTModule(embedDimension, numHeads, dropout)
            for _ in range(blocks)
        ])
        self.finalNorm = nn.LayerNorm(embedDimension, elementwise_affine=False, eps=1e-6)

    def forward(self, imageLatents, textEmbeddings, timeEmbeddings):
        
        x = imageLatents
        sharedParams = self.adaNorm(timeEmbeddings)
        for block in self.eachBlocks:
            x = block(x, textEmbeddings, sharedParams)

        x = self.finalNorm(x)
        return x
        
# nDit = NDiTModule(blocks=12, embedDimension=embedDimension, numHeads=12, dropout=0.2)
# tEmbed = TimeEmbedding(embedDimension=embedDimension)
# tEmbed.to(device)
# time = torch.tensor([1000]).to(device)
# tout = tEmbed(time)


# text = ["Generate an Image of a Dog Eating"]
# textModel = TextEmbedding()
# textembed = textModel(text)
# noisedlatents = torch.randn(1, 16, 768)

# out = nDit(noisedlatents, textembed, tout)
# out.shape