import torch
import torch.nn as nn
from Base import Conv, ConvBlock, DeConvBlock, Encoder

## font_generator
class Decoder(nn.Module):
    def __init__(self,cat=False):
        super().__init__()
        self.up0 = Conv(1024*2,1024, apply_batchnorm=False)
        self.up1 = DeConvBlock(512*3,512, apply_batchnorm=False)
        self.up2 = DeConvBlock(256*3,256, apply_batchnorm=False)
        self.up3 = DeConvBlock(128*3,128, apply_batchnorm=False)
        self.up4 = DeConvBlock(64*3,64, apply_batchnorm=False)
        self.up5 = DeConvBlock(32*3,32, apply_batchnorm=False)
        self.conv = Conv(32,32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    
    def forward(self, x1, x2, x3, x4, x5, x6,
                      e1, e2, e3, e4, e5, e6):
        x = torch.cat((x6,e6),dim=1)
        x = self.up0(x)
        x = self.up1(x,x5,e5)
        x = self.up2(x,x4,e4)
        x = self.up3(x,x3,e3)
        x = self.up4(x,x2,e2)
        x = self.up5(x,x1,e1)
        x = self.conv(x)
        x = self.out_conv(x)
        return x

class GeneativeModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.x_encoder = Encoder()
    self.x_decoder = Decoder()

  def forward(self, inputs,e1,e2,e3,e4,e5,e6):
    x = self.x_encoder(inputs)
    x = self.x_decoder(*x,e1,e2,e3,e4,e5,e6)
    return x
    
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.down1 = Conv(2,64, apply_batchnorm=False)
    self.down2 = Conv(64,64*2, apply_batchnorm=False)
    self.down3 = Conv(64*2,64*4, apply_batchnorm=False)
    self.down4 = Conv(64*4,64*8, apply_batchnorm=False)
    self.out_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False)
    self.maxpool = nn.MaxPool2d(2)

  def forward(self, inp, tar):
    x = torch.cat((inp,tar),axis=1)
    x = self.down1(x)
    x = self.maxpool(x)
    x = self.down2(x)
    x = self.maxpool(x)
    x = self.down3(x)
    x = self.maxpool(x)
    x = self.down4(x)
    x = self.out_conv(x)
    return x
