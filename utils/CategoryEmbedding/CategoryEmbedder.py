import numpy as np
import matplotlib.pyplot as plt 
saved_fonts = np.load("FONT/test.npz")
source_fonts = saved_fonts['source_fonts']
target_fonts = saved_fonts['target_fonts']