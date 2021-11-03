
from models.WNET import WNet
import torch
import numpy as np

datasets = np.load("./fonts/font.npz")
source_fonts = datasets['source_fonts']
target_fonts = datasets['target_fonts']

model = WNet()
model.fit(source_fonts, target_fonts)
