import torch
import torch.nn as nn
# emb 넣기

## Category Embedding
class CharDataset(torch.utils.data.Dataset):
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


## Font Generation
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, source_fonts, target_fonts):
        self.source_fonts = source_fonts
        self.target_fonts = target_fonts

    def __len__(self):
        return int(self.target_fonts.shape[0]/32*self.target_fonts.shape[1]/32)

    def __getitem__(self, idx):
        font = idx//2402
        word = idx%2402
        while True :
            rand = random.randint(0,2401)
            if rand != word:
                break
        source = self.target_fonts[font*32:(font+1)*32,rand*32:(rand+1)*32]
        target = self.target_fonts[font*32:(font+1)*32,word*32:(word+1)*32]
        return {"source":source, 
                "target":target, 
                "word":word, 
                "emb": (embed['cl1'][word], 
                        embed['cl2'][word], 
                        embed['cl3'][word], 
                        embed['cl4'][word], 
                        embed['cl5'][word], 
                        embed['cl6'][word])}

    def __repr__(self):
      return f"data size : {self.__len__()} "
