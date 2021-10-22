# Load Pretrained Weight
import os
PATH = os.listdir("/content/drive/MyDrive/HAN2HAN/cycle_unet_trainimg")[-1]
last_epoch = int(PATH.split('_')[3])
PATH = "/content/drive/MyDrive/HAN2HAN/cycle_unet_trainimg/" + PATH

model = WNet()
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))

train_dataloader = DataLoader(datasets, shuffle=True, batch_size=8)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loss_function = nn.L1Loss()

model.to(device)

optimizer = torch.optim.AdamW(model.parameters())
print("device :",device)

from tqdm.auto import tqdm

LAMBDA = 0.2

def train(model = model,
          device = device,
          dataloader = train_dataloader,
          epochs = 30,
          save_checkpoint = True,
          last_epoch = 0):
  
  progress_bar = tqdm(range(train_dataloader.__len__()*epochs))
  total_num = train_dataloader.__len__()*8
  for epoch in range(last_epoch,epochs):
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
      plt.savefig(f"/content/drive/MyDrive/HAN2HAN/cycle_unet_trainimg/epoch_x_{epoch:05}.png",dpi=300)
      plt.show()
      

      torch.save(model.state_dict(), f"/content/drive/MyDrive/HAN2HAN/cycle_unet_trainimg/UNET_state_dict_{epoch:05}_{total_loss.item():.4f}.pt")

train(last_epoch=last_epoch)

PATH = "/content/drive/MyDrive/HAN2HAN/cycle_unet_trainimg/" + 'UNET_state_dict_00023_0.0111.pt'
model = CycleUNet()
model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))