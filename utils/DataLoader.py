import torch
import torch.nn as nn
from Font2Numpy.FontTest import common_han

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

def category_dataloaer(source_fonts, target_fonts, shuffle=True, batch_size=8):
    datasets = CategoryDataset(source_fonts, target_fonts)
    dataloaer = DataLoader(datasets, shuffle=shuffle, batch_size=batch_size)
    return dataloaer

def gan_dataloaer(source_fonts, target_fonts, shuffle=True, batch_size=8):
    datasets = GANDataset(source_fonts, target_fonts)
    dataloaer = DataLoader(datasets, shuffle=shuffle, batch_size=batch_size)
    return dataloaer

def character_dataloaer(source_fonts, target_fonts, shuffle=True, batch_size=8):
    datasets = CharacterDataset(source_fonts, target_fonts)
    dataloaer = DataLoader(datasets, shuffle=shuffle, batch_size=batch_size)
    return dataloaer