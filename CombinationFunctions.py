import torch
import torch.nn as nn
import math
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPModel, CLIPProcessor


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

        self.encode = nn.Conv2d(in_channels = inChannels, out_channels = embedDimension, kernel_size = patchSize, stride = patchSize)
        self.positionalEmbedding = nn.Parameter(torch.zeros(1, self.patches, embedDimension))
        nn.init.trunc_normal_(self.positionalEmbedding, std=0.02)

    def forward(self, latentImage):

        allPatch = self.encode(latentImage)
        # print(allPatch.shape)
        flattened = allPatch.flatten(2).transpose(1, 2)
        # print(flattened.shape, self.positionalEmbedding.shape)
        out = flattened + self.positionalEmbedding
        return out
    


