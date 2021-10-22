import math
import torch
import torch.nn as nn
from Base import Conv, ConvBlock, DeConvBlock, Encoder


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
