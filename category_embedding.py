from models.WNET import WNet
import torch
import numpy as np

'''
You don't have to run this script for category embedding.
Embedded layers are provided via google drive,
simply run `!wget ` to download at HAN2HAN directory
'''

if __name__=="__main__":
    datasets = np.load("./fonts/font.npz")
    source_fonts = datasets['source_fonts']
    target_fonts = datasets['target_fonts']

    model = WNet()
    model.fit(source_fonts, target_fonts)

    features = model.get_features(torch.Tensor([source_fonts[:32,i*32:(i+1)*32]/255 for i in range(2402)]).reshape(2402,1,32,32))
    np.savez("./yourpath.npz",
            cl1 = features[0].detach().numpy(),
            cl2 = features[1].detach().numpy(),
            cl3 = features[2].detach().numpy(),
            cl4 = features[3].detach().numpy(),
            cl5 = features[4].detach().numpy(),
            cl6 = features[5].detach().numpy()
            )