
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
