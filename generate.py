# Load Generator Weight and Finetune

from utils.util import custom_img, knock_the_door
from utils.DataLoader import char_dataloader
import glob
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np

from models.AutoEncoder import AutoEncoder
from models.GAN import GeneativeModel

def finetuning(img_dir="/content/yourimg/", 
               ae_weight:"pt"="/content/drive/MyDrive/HAN2HAN/CharacterClustering/Temp_state_dict2.pt",
               character_emb_path:"npz"="/content/drive/MyDrive/HAN2HAN/CharacterClustering/Emb.npz",
               font_np_path:"npz"="/content/drive/MyDrive/HAN2HAN/Saved_Fonts.npz",
               category_layer:"npz"="/content/drive/MyDrive/HAN2HAN/Embedded_Fonts.npz",
               gen_weight:"pt"="/content/drive/MyDrive/HAN2HAN/GAN2/GAN_Generator_state_dict_0021_711820.62500000.pt",
               epochs=201,
               learning_rate=5e-4):
    # Load your img
    custom_char = custom_img(img_dir)

    # Load character embedder
    model = AutoEncoder()
    model.load_state_dict(torch.load(ae_weight))
    char_embedding = []
    char_labels = []
    with torch.no_grad():
        for i in range(int(len(custom_char)/2)):
            inputs = torch.cat((torch.Tensor(custom_char[(2*i)][0]).reshape(1,1,32,32),torch.Tensor(custom_char[(2*i)+1][0]).reshape(1,1,32,32)),dim=0)
            output,emd = model(inputs)
            char_embedding.append(emd[0].to('cpu').numpy())
            char_labels.append(custom_char[(2*i)][1])
            char_embedding.append(emd[1].to('cpu').numpy())
            char_labels.append(custom_char[(2*i)+1][1])

            # Matching characters to common_hangul
            char_dictionary = knock_the_door(character_emb_path,char_embedding)

    # Load layer embedding, source fonts
    datasets = np.load(font_np_path)
    embeded = np.load(category_layer)
    source_fonts = datasets['source_fonts']
    embed = {}
    embed['cl1'] = torch.Tensor(embeded['cl1'])
    embed['cl2'] = torch.Tensor(embeded['cl2'])
    embed['cl3'] = torch.Tensor(embeded['cl3'])
    embed['cl4'] = torch.Tensor(embeded['cl4'])
    embed['cl5'] = torch.Tensor(embeded['cl5'])
    embed['cl6'] = torch.Tensor(embeded['cl6'])

    # DataLoader
    dataloader, sample_dataloader, train_dataloader = char_dataloader(source_fonts, char_dictionary, custom_char, char_labels, embed)

    # Load Generator
    model = GeneativeModel()
    model.load_state_dict(torch.load(gen_weight))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device :",device)
    model.to(device)

    gen_loss = nn.L1Loss()
    optimizer_G = torch.optim.AdamW(model.parameters(),lr=learning_rate) # You need to calibrate the learning rate (5e-4 ~ 4e-4 recomendded)

    # Trainstep
    progress_bar = tqdm(range(train_dataloader.__len__()*epochs))
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for b,batch in enumerate(train_dataloader):

            optimizer_G.zero_grad()

            inputs = batch['source'].reshape(-1,1,32,32)/255
            target = batch['target'].reshape(-1,1,32,32)/255
            inputs = inputs.to(device)
            target = target.to(device)
            catemb = [emb.to(device) for emb in batch['emb']]

            output = model(inputs,*catemb)
            loss = gen_loss(output,target)
            loss.backward()
            optimizer_G.step()

            with torch.no_grad():
                progress_bar.update(1)
                total_loss += loss.sum()
            print(epoch,total_loss.item())

            if epoch%100==0:
                with torch.no_grad():
                    plotting = []
                    for i in range(3):
                        for sample in sample_dataloader:
                            source = sample['source']/255
                            target = sample['target']/255
                            source = source.to(device)
                            catemb = [emb.to(device) for emb in sample['emb']]
                            genera = model(source.reshape(-1,1,32,32),*catemb)
                            plotting.append((source,genera,[0],target))
                            break

                    plt.figure(figsize=(18,10))
                    for i in range(3):
                        for j in range(8):
                            plt.subplot(6,12,(24*i)+3*j+1)
                            plt.imshow(plotting[i][0][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
                            plt.axis('off')
                            plt.subplot(6,12,(24*i)+3*j+2)
                            plt.imshow(plotting[i][1][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
                            plt.axis('off')
                            plt.subplot(6,12,(24*i)+3*j+3)
                            # plt.imshow(plotting[i][3][j].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
                            plt.imshow(np.full((32,32,3),1,dtype=float),cmap='gray')
                            plt.axis('off')
                    plt.show()  
    return model, dataloader

def generate(model,
            dataloader,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),):
    generated_font = []
    progress_bar = tqdm(range(dataloader.__len__()))
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['source'].reshape(-1,1,32,32)/255
            target = batch['target'].reshape(-1,1,32,32)/255
            inputs = inputs.to(device)
            target = target.to(device)
            catemb = [emb.to(device) for emb in batch['emb']]
            
            output = model(target,*catemb)
            progress_bar.update(1)
            for gf in output.reshape(-1,32,32).to('cpu').detach().numpy():
                generated_font.append(np.vectorize(lambda x : x if x<250 else 255)(gf*255))
            # plt.subplot(1,3,1)
            # plt.imshow(inputs[1].reshape(32,32).to('cpu')*255,cmap='gray')
            plt.subplot(1,2,1)
            plt.imshow(target[1].reshape(32,32).to('cpu').detach().numpy()*255,cmap='gray')
            plt.axis('off')
            plt.subplot(1,2,2)
            x = output[1].reshape(32,32).to('cpu').detach().numpy()*255
            x = np.vectorize(lambda x : x if x<250 else 255)(x)
            plt.imshow(x,cmap='gray')
            plt.axis('off')
            plt.show()

    return generated_font

def write(to_gen, font_gen):
    gen = to_gen.split('\n')
    len_max = max(list(map(len,to_gen.split('\n'))))
    row_num = len(to_gen.split('\n'))
    w,h = len_max,len(to_gen.split('\n'))

    plt.figure(figsize=(w,h))
    for row in range(row_num):
        for c in range(len(to_gen.split('\n')[row])):
            plt.subplot(row_num,len_max,(row*len_max)+c+1)
            # plt.title(f"{gen[row][c]}")
            if gen[row][c] not in common_han:
                plt.imshow(np.full((32,32,3),1,dtype=float),cmap='gray')
            else:
                plt.imshow(font_gen[common_han.index(gen[row][c])], cmap='gray')
            plt.axis('off')
    plt.show()

model, dataloader = finetuning()
font_gen = generate(model, dataloader)
write("안녕하세요",font_gen)