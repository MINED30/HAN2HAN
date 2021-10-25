import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.DataLoader import category_dataloaer
from tqdm.auto import tqdm

from Base import Conv, ConvBlock, DeConvBlock, Encoder


## Category Embedder
class CategoryDecoder(nn.Module):
    def __init__(self,cat=False):
        super().__init__()
        self.cat = cat
        if self.cat:
            self.up0 = Conv(2048,1024, apply_batchnorm=False)
        self.up1 = DeConvBlock(1024,512, apply_batchnorm=False)
        self.up2 = DeConvBlock(512,256, apply_batchnorm=False)
        self.up3 = DeConvBlock(256,128, apply_batchnorm=False)
        self.up4 = DeConvBlock(128,64, apply_batchnorm=False)
        self.up5 = DeConvBlock(64,32, apply_batchnorm=False)
        self.conv = Conv(32,32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5, x6, x0=None):
        if self.cat:
            x = torch.cat((x0,x6),dim=1)
            x = self.up0(x)
            x = self.up1(x,x5)
        else :
            x = self.up1(x6,x5)
        x = self.up2(x,x4)
        x = self.up3(x,x3)
        x = self.up4(x,x2)
        x = self.up5(x,x1)
        x = self.conv(x)
        x = self.out_conv(x)
        return x

class WNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_encoder = Encoder()
        self.x_decoder = CategoryDecoder()
        self.y_encoder = Encoder()
        self.y_decoder = CategoryDecoder(cat=True)

    def forward(self, inputs):
        x_ = self.x_encoder(inputs)
        y = self.x_decoder(*x_)
        y_ = self.y_encoder(y)
        x = self.y_decoder(*y_,x_[-1])
        return y, x
    
    def from_pretrained(PATH):
        self.load_state_dict(torch.load(PATH))

    def get_features(self,inputs):
        feature = self.x_encoder(inputs)
        return feature

    def get_sourcefont_features(self, source_fonts, save_as_np:path=None):
        feature = torch.Tensor([source_fonts[:32,i*32:(i+1)*32]/255 for i in range(2402)]).reshape(2402,1,32,32)
        features = self.get_features(feature)
        if save_as_np:
            np.savez(f"{save_as_np}/Embedded_Fonts.npz", 
                     CategoryLayer1 = features[0].detach().numpy(),
                     CategoryLayer2 = features[1].detach().numpy(),
                     CategoryLayer3 = features[2].detach().numpy(),
                     CategoryLayer4 = features[3].detach().numpy(),
                     CategoryLayer5 = features[4].detach().numpy(),
                     CategoryLayer6 = features[5].detach().numpy()
                        )
        return features

    def plotting_img(self,source_fonts):
        word = [random.randint(0,2350) for _ in range(16)]  # Hangul
        word.extend([random.randint(2350,2402) for _ in range(8)])  # Alphabet

        plotting = [self(torch.Tensor([source_fonts[:32,word[i]*32:(word[i]+1)*32]/255 for i in range(8)]).reshape(8,1,32,32)),
                    self(torch.Tensor([source_fonts[:32,word[i]*32:(word[i]+1)*32]/255 for i in range(8,16)]).reshape(8,1,32,32)),
                    self(torch.Tensor([source_fonts[:32,word[i]*32:(word[i]+1)*32]/255 for i in range(16,24)]).reshape(8,1,32,32))]

        plt.figure(figsize=(12,10))
        for i in range(3):
            for j in range(8):
                plt.subplot(6,8,(i*8)+j+1)
                plt.imshow(plotting[i][0][j].to('cpu').reshape(32,32).detach().numpy()*255,cmap='gray')
                plt.axis('off')
        for i in range(3):
            for j in range(8):
                plt.subplot(6,8,(i*8)+j+25)
                plt.imshow(plotting[i][1][j].to('cpu').reshape(32,32).detach().numpy()*255,cmap='gray')
                plt.axis('off')
        plt.show()

    def train(model,
              source_fonts,
              target_fonts,
              device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
              epochs=30,
              save_checkpoint:path=None,
              save_plt:path=None,
              loss_function=nn.L1Loss(),
              LAMBDA = 0.2):

        dataloader = category_dataloaer(source_fonts, target_fonts, shuffle=True, batch_size=8)

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters())

        progress_bar = tqdm(range(dataloader.__len__()*epochs))
        total_num = dataloader.__len__()*8
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_loss_x = 0
            total_loss_y = 0
            for batch in dataloader:
                inputs = batch[0].reshape(-1,1,32,32)/255
                target = batch[1].reshape(-1,1,32,32)/255
                inputs = inputs.to(device)
                target = target.to(device)

                output_y, output_x = model(inputs)

                loss_y = loss_function(output_y,target)
                loss_x = loss_function(output_x,inputs)
                loss = LAMBDA*loss_x + loss_y

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)

                loss_avg = loss.sum() / total_num
                total_loss += loss_avg
                total_loss_x += loss_x.sum() / total_num
                total_loss_y += loss_y.sum() / total_num

        print(total_loss,total_loss_x,total_loss_y)

        with torch.no_grad():
            plotting = [model(torch.Tensor([source_fonts[:32,word[i]*32:(word[i]+1)*32]/255 for i in range(8)]).reshape(8,1,32,32).to(device)),
                        model(torch.Tensor([source_fonts[:32,word[i]*32:(word[i]+1)*32]/255 for i in range(8,16)]).reshape(8,1,32,32).to(device)),
                        model(torch.Tensor([source_fonts[:32,word[i]*32:(word[i]+1)*32]/255 for i in range(16,24)]).reshape(8,1,32,32).to(device))]
            plt.figure(figsize=(12,10))
            for i in range(3):
                for j in range(8):
                    plt.subplot(6,8,(i*8)+j+1)
                    plt.imshow(plotting[i][0][j].to('cpu').reshape(32,32).detach().numpy()*255,cmap='gray')
                    plt.axis('off')
            for i in range(3):
                for j in range(8):
                    plt.subplot(6,8,(i*8)+j+25)
                    plt.imshow(plotting[i][1][j].to('cpu').reshape(32,32).detach().numpy()*255,cmap='gray')
                    plt.axis('off')
            if save_plt:
                plt.savefig(f"{save_plt}/plotting_img_{epoch:05}.png",dpi=300)
            plt.show()
            
            if save_checkpoint:
                torch.save(model.state_dict(), f"{save_checkpoint}/WNET_state_dict_{epoch:05}_{total_loss.item():.4f}.pt")

