from models.GAN import GeneativeModel, Discriminator, gan_train
from utils.font_test import common_han
import numpy as np
import torch
# Load source fonts & target fonts
datasets = np.load("./fonts/font.npz")
source_fonts = datasets['source_fonts']
target_fonts = datasets['target_fonts']

# Load category embedded layers
category_ = np.load("./category_emb.npz")
category_emb = {}
category_emb['cl1'] = torch.Tensor(category_['cl1'])
category_emb['cl2'] = torch.Tensor(category_['cl2'])
category_emb['cl3'] = torch.Tensor(category_['cl3'])
category_emb['cl4'] = torch.Tensor(category_['cl4'])
category_emb['cl5'] = torch.Tensor(category_['cl5'])
category_emb['cl6'] = torch.Tensor(category_['cl6'])

generator = GeneativeModel()
discriminator = Discriminator()

gan_train(generator,
        discriminator,
        source_fonts,
        target_fonts,
        category_emb)