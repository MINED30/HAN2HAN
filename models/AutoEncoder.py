import math
import torch
import torch.nn as nn
import numpy as np
from models.Base import Conv, ConvBlock
from utils.DataLoader import character_dataloader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class DeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch,out_ch,2,2)
        self.conv = Conv(out_ch,out_ch,apply_batchnorm=apply_batchnorm)
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        return x

## Character Embedder
class CharacterEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_conv = Conv(1,32,apply_batchnorm=False)
        self.down1 = ConvBlock(32,64, apply_batchnorm=False)
        self.down2 = ConvBlock(64,128, apply_batchnorm=True)
        self.down3 = ConvBlock(128,256, apply_batchnorm=True)
        self.down4 = ConvBlock(256,512, apply_batchnorm=True)
        self.down5 = ConvBlock(512,2402, apply_batchnorm=True)

    def forward(self, x):
        x = self.inp_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return x

class CharacterDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = DeConvBlock(2402,512, apply_batchnorm=False)
        self.up2 = DeConvBlock(512,256, apply_batchnorm=False)
        self.up3 = DeConvBlock(256,128, apply_batchnorm=False)
        self.up4 = DeConvBlock(128,64, apply_batchnorm=False)
        self.up5 = DeConvBlock(64,32, apply_batchnorm=False)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.out_conv(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CharacterEncoder()
        self.decoder = CharacterDecoder()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.flatten(x)
        x = self.decoder(x)
        return x,x1

    def from_pretrained(self, PATH):
        '''
        model = AutoEncoder()
        model.from_pretained(PATH)
        '''
        self.load_state_dict(torch.load(PATH))

    def get_features(self,
                     source_fonts, 
                     target_fonts,
                     batch_size=256,
                     save_path=None,
                     device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        trans_dataloader = character_dataloader(source_fonts, target_fonts, shuffle=False, batch_size=batch_size)
        progress_bar = tqdm(range(trans_dataloader.__len__()))
        for b,batch in enumerate(trans_dataloader):
            inputs = batch[0].reshape(-1,1,32,32)/255
            inputs = inputs.to(device)
            
            _,emd = self(inputs)

            progress_bar.update(1)

            if b==0:
                temp = emd
                labels = batch[1]
            else :
                with torch.no_grad():
                    temp = torch.cat((temp,emd),dim=0)
                    labels = torch.cat((labels,batch[1]))

        embed = temp.to('cpu').detach().numpy()
        label = labels.to('cpu').detach().numpy()
        if save_path :
            np.savez(f"{save_path}/Emb.npz",
                      embed = embed,
                      label = label
                      )
        return embed, label
        
    def fit(self,
            source_fonts,
            target_fonts,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            epochs=30,
            batch_size=8):

        '''
        model = AutoEncoder()
        model.train(source_fonts,target_fonts)
        '''
        dataloader = character_dataloader(source_fonts, target_fonts, shuffle=True, batch_size=batch_size)
        self.to(device),
        optimizer=torch.optim.AdamW(self.parameters())
        loss_function=nn.MSELoss()
        progress_bar = tqdm(range(dataloader.__len__()*epochs))
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for b,batch in enumerate(dataloader):

                inputs = batch[0].reshape(-1,1,32,32)/255
                inputs = inputs.to(device)
                
                optimizer.zero_grad()
                output, _ = self(inputs)
                loss = loss_function(output,inputs)
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

                with torch.no_grad():
                    if b%100==0:
                        total_loss += loss.sum()
                        print(loss.sum().item())

                    plt.figure(figsize=(6,5))
                    for i in range(8):
                        plt.subplot(4,4,(i*2)+1)
                        plt.imshow(inputs.reshape(-1,32,32)[i].to('cpu').numpy(),cmap='gray')
                        plt.axis('off')
                        plt.subplot(4,4,(i*2)+2)
                        plt.imshow(output.reshape(-1,32,32)[i].to('cpu').numpy(),cmap='gray')
                        plt.axis('off')
                    plt.show()
            print(total_loss)
