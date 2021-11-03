from models.AutoEncoder import AutoEncoder

import numpy as np
import torch

# Load source fonts & target fonts
datasets = np.load("./fonts/font.npz")
source_fonts = datasets['source_fonts']
target_fonts = datasets['target_fonts']

# fit font 
model = AutoEncoder()
# model.fit(source_fonts,target_fonts)

# get features

embed, label = model.get_features(source_fonts,target_fonts)
np.savez("./character_emb.npz",
         embed = embed,
         label = label
         )