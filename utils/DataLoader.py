import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.font_test import common_han
import random

# Category Dataloader
class CategoryDataset(torch.utils.data.Dataset):
    def __init__(self, source_fonts, target_fonts):
        self.source_fonts = source_fonts
        self.target_fonts = target_fonts

    def __len__(self):
        return int(self.target_fonts.shape[0]/32*self.target_fonts.shape[1]/32)

    def __getitem__(self, idx):
      font = idx//2402
      word = idx%2402
      source = self.source_fonts[:32,word*32:(word+1)*32]
      target = self.target_fonts[font*32:(font+1)*32,word*32:(word+1)*32]
      return source, target

    def __repr__(self):
      return f"data size : {self.__len__()} "

# GAN DataLoader
class GANDataset(torch.utils.data.Dataset):
    def __init__(self, source_fonts, target_fonts):
        self.source_fonts = source_fonts
        self.target_fonts = target_fonts

    def __len__(self):
        return int(self.target_fonts.shape[0]/32*self.target_fonts.shape[1]/32)
        # return 2402

    def __getitem__(self, idx):
      font = idx//2402
      word = idx%2402
      while True :
        rand = random.randint(0,2401)
        if rand != word:
          break
      source = self.target_fonts[font*32:(font+1)*32,rand*32:(rand+1)*32]
      target = self.target_fonts[font*32:(font+1)*32,word*32:(word+1)*32]
      # print(source.shape,idx, rand, word)
      return {"source":source, 
              "target":target, 
              "word":word, 
              "emb":(embed['cl1'][word], embed['cl2'][word], embed['cl3'][word], embed['cl4'][word], embed['cl5'][word], embed['cl6'][word])}

    def __repr__(self):
      return f"data size : {self.__len__()} "


# Character Dataloader
class CharacterDataset(torch.utils.data.Dataset):
    def __init__(self, source_fonts, target_fonts, common_han=common_han):
        self.common_han = common_han
        self.fonts = np.concatenate((source_fonts,target_fonts),axis=0)

    def __len__(self):
        return int(self.fonts.shape[0]/32*self.fonts.shape[1]/32)

    def __getitem__(self, idx):
      font = idx//2402
      word = idx%2402
      source = self.fonts[font*32:(font+1)*32,word*32:(word+1)*32]
      return source,word 

    def __repr__(self):
      return f"data size : {self.__len__()} "

# Finetuning Dataloader
class CharTrain(torch.utils.data.Dataset):
    def __init__(self, source_fonts, char_dictionary, custom_char, char_labels, embed):
        self.source_fonts = source_fonts
        self.common_han = common_han
        self.char_dictionary = char_dictionary
        self.custom_char = custom_char
        self.char_labels = char_labels
        self.embed_layer = embed

    def __len__(self):
        return len(self.custom_char)

    def __getitem__(self, idx):
      tar = self.custom_char[idx]
      target_img = tar[0] # 타겟이미지
      label = tar[1] # 타겟이미지의 글자
      label_idx = self.char_labels.index(label)
      source_idx = self.common_han.index(label) # 타겟이미지의 인덱스 2402
      while True :
        random_idx = random.randint(0,self.__len__()-1)
        if random_idx != label_idx:
          break
      source_img = self.custom_char[random_idx][0] # 스타일링을위한 소스이미지

      return {"source":source_img, 
              "target":target_img, 
              "emb":(self.embed_layer['cl1'][source_idx], 
                     self.embed_layer['cl2'][source_idx], 
                     self.embed_layer['cl3'][source_idx], 
                     self.embed_layer['cl4'][source_idx], 
                     self.embed_layer['cl5'][source_idx], 
                     self.embed_layer['cl6'][source_idx])}

    def __repr__(self):
      return f"font size : {self.__len__()} "

class CharMatch(torch.utils.data.Dataset):
    def __init__(self, source_fonts, char_dictionary, custom_char, char_labels, embed):
        self.source_fonts = source_fonts
        self.common_han = common_han
        self.char_dictionary = char_dictionary
        self.custom_char = custom_char
        self.char_labels = char_labels
        self.embed_layer = embed

    def __len__(self):
        return len(self.char_dictionary)

    def __getitem__(self, idx):
      character_togenerate = self.common_han[idx] # 생성하려는 글자
      idx_tomatch = self.char_dictionary[character_togenerate] # 매핑된글자로 인덱스확인
      target = self.custom_char[idx_tomatch] # 참조할 커스텀글자(img) 선택 (img, 글자)
      assert target[1]==self.char_labels[idx_tomatch],"mapping error"
      return {"source":target[0], 
              "target":target[0], 
              "emb":(self.embed_layer['cl1'][idx], 
                     self.embed_layer['cl2'][idx], 
                     self.embed_layer['cl3'][idx], 
                     self.embed_layer['cl4'][idx], 
                     self.embed_layer['cl5'][idx], 
                     self.embed_layer['cl6'][idx])}

    def __repr__(self):
      return f"font size : {self.__len__()} "


def category_dataloader(source_fonts, target_fonts, shuffle=True, batch_size=8):
    datasets = CategoryDataset(source_fonts, target_fonts)
    dataloaer = DataLoader(datasets, shuffle=shuffle, batch_size=batch_size)
    return dataloaer

def gan_dataloader(source_fonts, target_fonts, shuffle=True, batch_size=8):
    datasets = GANDataset(source_fonts, target_fonts)
    dataloaer = DataLoader(datasets, shuffle=shuffle, batch_size=batch_size)
    return dataloaer

def character_dataloader(source_fonts, target_fonts, shuffle=True, batch_size=8):
    datasets = CharacterDataset(source_fonts, target_fonts)
    dataloaer = DataLoader(datasets, shuffle=shuffle, batch_size=batch_size)
    return dataloaer

def char_dataloader(source_fonts, char_dictionary, custom_char, char_labels, embed):
    datasets = CharMatch(source_fonts, char_dictionary, custom_char, char_labels, embed)
    train_datasets = CharTrain(source_fonts, char_dictionary, custom_char, char_labels, embed)
    dataloader = DataLoader(datasets, shuffle=False, batch_size=256)
    sample_dataloader = DataLoader(datasets, shuffle=True, batch_size=8)
    train_dataloader = DataLoader(train_datasets, shuffle=True, batch_size=2)
    return dataloader, sample_dataloader, train_dataloader
