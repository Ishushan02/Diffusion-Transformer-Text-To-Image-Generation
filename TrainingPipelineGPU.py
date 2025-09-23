import wandb
import torch
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
from torch.optim.lr_scheduler import StepLR
import random
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPModel, CLIPProcessor
import torch.nn.functional as Fn
from CombinationFunctions import ImageInputToDiT, NDiTModule, Decoder, TimeEmbedding, TextEmbedding
from tqdm import tqdm

if(torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Device: ", device)

os.environ['K8S_TIMEOUT_SECONDS'] = '43200'

wandb.login()

wandb.init(
    project="diffusion-transformer",  
    name="experiment-1",    
    id="siv0nm05",  
    resume="allow",
)




class ImageTextData(Dataset):
    def __init__(self, data, transform = None, rootDir = ""):
        super().__init__()
        self.data = data
        self.transform = transform
        self.rootDir = rootDir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = os.path.join(self.rootDir, row['imagePath'])
        captions = [
            row['caption1'],
            row['caption2'],
            row['caption3'],
            row['caption4'],
            row['caption5']
        ]

        caption = random.choice(captions)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, caption



# data = pd.read_csv("dataset/COCO2017.csv")
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),
#     transforms.ToTensor(),                 
#     transforms.Normalize([0.5]*3, [0.5]*3)])

# idata = ImageTextData(data, transform)

# image, caption = idata.__getitem__(2000)
# image.shape, caption


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
    

# fModel = FinalModel(8, 128, 768, 2, 1000, 12, 12, 0.2)

    

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
data = pd.read_csv("dataset/COCO2017.csv")
transform = transforms.Compose([
    transforms.Resize((IMAGEHEIGHT, IMAGEWIDTH)),
    transforms.ToTensor(),                 
    transforms.Normalize([0.5]*3, [0.5]*3)])


model = FinalModel(latentSize=LATENTSIZE, latentChannel=LATENTCHANNEL, embedDimension=EMBEDDINGDIM, patchSize=PATCHSIZE,
                    T = T, numHeads=HEADS, blocks=DITBLOCK, dropout=dropout,
                    beta_schedule = "squaredcos_cap_v2", modelName = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers")


data = pd.read_csv("dataset/COCO2017.csv")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),                 
    transforms.Normalize([0.5]*3, [0.5]*3)])

torchDataset = ImageTextData(data, transform)
# dataloader = DataLoader(torchDataset, batch_size=BATCHSIZE, shuffle = True, num_workers=8, persistent_workers=True)
dataloader = DataLoader(torchDataset, batch_size=BATCHSIZE, shuffle = True, num_workers=0)




lossFn =  nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=2e-5, weight_decay=3e-2, eps=1e-10)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)



start_epoch = 0

# checkpoint_path = os.path.join("", "model")
# checkpoint_path = os.path.join(checkpoint_dir, "dit.pt")
baseDir = os.path.dirname(__file__)

checkpoint_path = os.path.join(baseDir, "model", "dit.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print(f"Resuming from epoch {start_epoch}")
else:
    print("Loading pretrained model...")


model = torch.nn.DataParallel(model)
model.to(device)

for each_epoch in range(start_epoch, epochs):
    model.train()
    
    loop = tqdm(dataloader, f"{each_epoch}/{epochs}")
    ditloss = 0.0
    for X, captions in loop:
        cBatch = X.shape[0]
        t = torch.randint(0, T, (cBatch,), device=device).long()


        predictedNoise, noise = model(X, captions, t)
       
        # print(predictedNoise.shape, noise.shape)
    #     break
    # break

        loss = lossFn(predictedNoise, noise)
        
        ditloss += loss.item()
   
        
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loop.set_postfix({
            "DIT Loss": f"{ditloss}"
        })

    ditloss /= len(dataloader)   

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save({
        'epoch': each_epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, checkpoint_path)
    
    
    wandb.log({
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Decoder Loss": ditloss
    })
    scheduler.step()
    
