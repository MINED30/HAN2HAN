import torch
import torch.nn as nn
from utils.DataLoader import gan_dataloader
from models.Base import Conv, ConvBlock, DeConvBlock, Encoder
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils.font_test import common_han

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



def gan_train(generator,
              discriminator,
              source_fonts,
              target_fonts,
              device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
              epochs = 60,
              gen_loss_function=nn.L1Loss(),
              dis_loss_function=nn.BCEWithLogitsLoss(),
              LAMBDA=1000,
              train_batch_size=32,
              sample_batch_size=8,
              generate_img=True,
              save_checkpoint=None,
              save_img=None):
  
  train_dataloader = gan_dataloader(source_fonts, target_fonts, shuffle=True, batch_size=train_batch_size)
  sample_dataloader = gan_dataloader(source_fonts, target_fonts, shuffle=True, batch_size=sample_batch_size)
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  gen_loss =gen_loss_function
  dis_loss = dis_loss_function
  LAMBDA = LAMBDA
  optimizer_G = torch.optim.AdamW(generator.parameters(),lr=5e-4)
  optimizer_D = torch.optim.AdamW(discriminator.parameters(),lr=5e-5)

  progress_bar = tqdm(range(train_dataloader.__len__()*(epochs)))
  t = []
  for epoch in range(epochs):
    generator.train()
    total_loss = 0
    total_generative_loss = 0
    total_discriminative_loss = 0
    
    for batch in enumerate(train_dataloader):

      optimizer_G.zero_grad()

      inputs = batch['source'].reshape(-1,1,32,32)/255
      target = batch['target'].reshape(-1,1,32,32)/255
      inputs = inputs.to(device)
      target = target.to(device)
      catemb = [emb.to(device) for emb in batch['emb']]

      output = generator(inputs,*catemb)
      real_output = discriminator(inputs,target)
      disc_output = discriminator(inputs,output)
      gene_output = discriminator(inputs,output.detach())
      
      error_G = dis_loss(disc_output,torch.ones_like(disc_output))
      l1_loss = gen_loss(output,target)*LAMBDA
      gen_loss = error_G + l1_loss
      gen_loss.backward()
      optimizer_G.step()

      optimizer_D.zero_grad()
      discriminator_lossT = dis_loss(real_output,torch.ones_like(real_output))
      discriminator_lossF = dis_loss(gene_output,torch.zeros_like(gene_output))
      discriminator_loss = discriminator_lossF + discriminator_lossT
      discriminator_loss.backward()
      optimizer_D.step()


      with torch.no_grad():
        progress_bar.update(1)

        generative_loss_sum = l1_loss.sum() + error_G.sum()
        discriminative_loss_sum = discriminator_lossF.sum() + discriminator_lossT.sum()
        loss_sum = l1_loss.sum() + error_G.sum() + discriminator_lossF.sum() + discriminator_lossT.sum()

        total_generative_loss += generative_loss_sum
        total_discriminative_loss += discriminative_loss_sum
        total_loss += loss_sum

    print(epoch, "total_loss", round(total_loss.item(),4), "total_generative_loss", round(total_generative_loss.item(),4), "total_discriminative_loss",round(total_discriminative_loss.item(),4))
    t.append([epoch,total_loss.item(),total_generative_loss.item(),total_discriminative_loss.item()])
    with torch.no_grad():

      if generate_img:

        plotting = []
        for i in range(3):
          for sample in sample_dataloader:
            source = sample['source']/255
            target = sample['target']/255
            source = source.to(device)
            catemb = [emb.to(device) for emb in sample['emb']]
            genera = generator(source.reshape(-1,1,32,32),*catemb)
            plotting.append((source,genera,sample['word'],target))
            break

        plt.figure(figsize=(18,10))
        for i in range(3):
          for j in range(8):
            plt.subplot(6,12,(24*i)+3*j+1)
            plt.title(f"source : {common_han[plotting[i][2][j]]}")
            plt.imshow(plotting[i][0][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
            plt.axis('off')
            plt.subplot(6,12,(24*i)+3*j+2)
            plt.title(f"target : {common_han[plotting[i][2][j]]}")
            plt.imshow(plotting[i][1][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
            plt.axis('off')
            plt.subplot(6,12,(24*i)+3*j+3)
            plt.title(f"target : {common_han[plotting[i][2][j]]}")
            plt.imshow(plotting[i][3][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
            plt.axis('off')
        if save_img:
          plt.savefig(f"{save_img}/GAN_epoch_{epoch:04}.png",dpi=300)
        plt.show()

      if save_checkpoint:
        torch.save(generator.state_dict(), f"{save_checkpoint}/GAN_Generator_state_dict_{epoch:04}_{total_loss.item():.8f}.pt")
        torch.save(discriminator.state_dict(), f"{save_checkpoint}/GAN_Discriminator_state_dict_{epoch:04}_{total_loss.item():.8f}.pt")
