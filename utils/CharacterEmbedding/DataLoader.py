
## Character Embedding
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, fonts):
        self.common_han = common_han
        self.fonts = fonts

    def __len__(self):
        return int(self.fonts.shape[0]/32*self.fonts.shape[1]/32)

    def __getitem__(self, idx):
      font = idx//2402
      word = idx%2402
      source = self.fonts[font*32:(font+1)*32,word*32:(word+1)*32]
      return source,word 

    def __repr__(self):
      return f"data size : {self.__len__()} "