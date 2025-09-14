import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import random
from CombinationFunctions import ImageInputToDiT, NDiTModule, Decoder, TimeEmbedding, TextEmbedding



if(torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device: ", device)



class FinalModel(nn.Module):
    def __init__(self, latentSize, latentChannel, embedDimension, patchSize, T, numHeads, blocks, dropout, beta_schedule = "squaredcos_cap_v2", modelName="mit-han-lab/dc-ae-f64c128-in-1.0-diffusers"):
        super().__init__()

        self.input = ImageInputToDiT(latentSize, latentChannel, embedDimension, patchSize, T, beta_schedule, modelName)

        self.timeEmbedding = TimeEmbedding(embedDimension)
        self.textEmbeding = TextEmbedding()

        self.ditBlocks = NDiTModule(blocks, embedDimension, numHeads, dropout)
        
        self.output = Decoder(embedDimension, latentSize, latentChannel, patchSize, T, beta_schedule, modelName)

    def forward(self, x, captions, t):

        batchSize, channels, height, width = x.shape
        noisedLatents, noise = self.input(x, t)

        timeEmbed = self.timeEmbedding(t)
        textembed = self.textEmbeding(captions)

        ditOutput = self.ditBlocks(noisedLatents, textembed, timeEmbed)
        
        predictedNoise = self.output(ditOutput)
        
        return predictedNoise, noise


IMAGEHEIGHT = 512
IMAGEWIDTH = 512
EMBEDDINGDIM = 768
BATCHSIZE = 16
INCHANNELS = 3
LATENTSIZE = 8
LATENTCHANNEL = 128
PATCHSIZE = 2
T = 1000
DITBLOCK = 12
HEADS = 12
dropout = 0.2

epochs = 1000


model = FinalModel(latentSize=LATENTSIZE, latentChannel=LATENTCHANNEL, embedDimension=EMBEDDINGDIM, patchSize=PATCHSIZE,
                    T = T, numHeads=HEADS, blocks=DITBLOCK, dropout=dropout,
                    beta_schedule = "squaredcos_cap_v2", modelName = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers")

transform = transforms.Compose([
    transforms.Resize((IMAGEHEIGHT, IMAGEHEIGHT)),
    transforms.ToTensor(),                 
    transforms.Normalize([0.5]*3, [0.5]*3)])


lossFn =  nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5, weight_decay=3e-2, eps=1e-10)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


checkpoint_path = os.path.join("model", "dit.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print("Loading pretrained model...")

model.to(device)

image = torch.randn(3, 512, 512)
to_pil = transforms.ToPILImage()
imagePIL = to_pil(image)
imagePIL = transform(imagePIL).unsqueeze(0)
caption = ["A dog is running towards me"]
batchProduce = len(caption)
modelName = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers"
beta_schedule = "squaredcos_cap_v2"
noisyLatent = torch.randn(1, 128, 8, 8)

for t in reversed(range(1000)):

    tBatch = torch.Tensor([t]).long()
    tBatch = tBatch.to(device)
    imagePIL = imagePIL.to(device)
    
    predictedNoise, actualNoise = model(imagePIL, caption, tBatch)
    
    with torch.no_grad():
        latents = model.input.dc_ae.encode(imagePIL).latent

    model.input.noiseScheduler.betas = model.input.noiseScheduler.betas.to(device)
    model.input.noiseScheduler.alphas_cumprod = model.input.noiseScheduler.alphas_cumprod.to(device)

    alphaT = model.input.noiseScheduler.alphas_cumprod.to(device)[tBatch].view(batchProduce, 1, 1, 1)
    noisyLatents = torch.sqrt(alphaT) * latents + torch.sqrt(1 - alphaT) * actualNoise
    originallatents = (noisyLatents - torch.sqrt(1 - alphaT) * predictedNoise) / torch.sqrt(alphaT)
    
    if(t > 0):
        betaT = model.input.noiseScheduler.betas[tBatch]
        mean = torch.sqrt(model.input.noiseScheduler.alphas_cumprod[t-1]) * originallatents + \
            torch.sqrt(1 - model.input.noiseScheduler.alphas_cumprod[t-1]) * predictedNoise
        noise = torch.randn_like(originallatents)
        nextStepLatents = mean + torch.sqrt(betaT) * noise
        with torch.no_grad():
            originalImage = model.input.dc_ae.decode(nextStepLatents).sample
            originalImage = originalImage.to(device)
            imagePIL = originalImage
        # print(originalImage.shape)
        # break
    else:
        with torch.no_grad():
            originalImage = model.input.dc_ae.decode(originallatents).sample
            imagePIL = originalImage
            originalImage = originalImage.to(device)
        imagePIL = imagePIL * 0.5 + 0.5
        break
    if t % 50 == 0:
        print(f" {t} order Completed ")

originalImage = imagePIL.squeeze(0)
image = to_pil(originalImage)
plt.imshow(image)
plt.axis('off')

image.save("output.png")
