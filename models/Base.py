import math
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=True):
        super().__init__()
        self.apply_batchnorm = apply_batchnorm
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
    
    def forward(self, x):

        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.apply_batchnorm:
            x = self.batch_norm(x)
        x = self.relu(x)

        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_block = Conv(in_ch, out_ch, apply_batchnorm=apply_batchnorm)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x

class DeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch,out_ch,2,2)
        self.conv = Conv(in_ch,out_ch,apply_batchnorm=apply_batchnorm)
    
    def forward(self, x1, x2):
        x = self.deconv(x1)
        x = torch.cat((x,x2),dim=1)
        x = self.conv(x)
        return x 

class Encoder(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.inp_conv = Conv(1,32,apply_batchnorm=False)
        self.down1 = ConvBlock(32,64, apply_batchnorm=False)
        self.down2 = ConvBlock(64,128, apply_batchnorm=True)
        self.down3 = ConvBlock(128,256, apply_batchnorm=True)
        self.down4 = ConvBlock(256,512, apply_batchnorm=True)
        self.down5 = ConvBlock(512,1024, apply_batchnorm=True)
    
    def forward(self, inputs):
        x1 = self.inp_conv(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        return x1, x2, x3, x4, x5, x6
