import sys, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import time
from FontTest import common_han, test_fonts_in_path

def draw_single_char(ch, font_n, size, font_path, display_font=False):
  adding=4
  while True:
    font = ImageFont.truetype(font = os.path.join(font_path + font_n), size = size+adding)
    x, y = font.getsize(ch)
    font_size = max(x,y)
    if font_size+2 > size:
      adding-=1
      continue

    img = Image.new('RGB', (size, size), (255, 255, 255)).convert('L')
    draw = ImageDraw.Draw(img)
    draw.text(((size-x)/2,(size-y)/2), ch, font=font)

    img = np.array(img)
    tmp = np.where(img!=255)
    ws,we,hs,he = max(min(tmp[0]-5),0), min(max(tmp[0]+5),size), max(min(tmp[1])-5,0), min(max(tmp[1]+5),size)
    img = img[ws:we,hs:he]

    w_left = int((size-img.shape[0])/2)
    w_right = size-img.shape[0]-w_left
    h_left = int((size-img.shape[1])/2)
    h_right = size-img.shape[1]-h_left

    img = np.pad(img,((w_left,w_right),(h_left,h_right)), 'constant', constant_values=255)
    
    if img.shape!=(size,size):
      print(ch,font,img.shape,"ERROR")

    if display_font:
      plt.figure(figsize=(1,1))
      plt.imshow(img,cmap='gray')
      plt.show()

    return img


def font2numpy(fonts, font_path):
    progress_bar = tqdm(range(len(fonts*len(common_han))))

    for i in range(len(fonts)):
        for letter in common_han:
            char_img = draw_single_char(letter,fonts[i],32,font_path = font_path,display_font=False)
            if letter == '가':
              temp = char_img
            else :
              temp = np.concatenate((temp,char_img),axis=1)
              progress_bar.update(1)
            
        if i == 0 :
          total = temp
        else :
          total = np.concatenate((total,temp),axis=0)
    
    return total

if __name__=='__main__':
  font_path = "your path"
  fonts = test_fonts_in_path(font_path)

  source_font = fonts[0]  # you can choose source font
  target_font = fonts[1:] # the rest of the source font

  source_font_np = font2numpy(source_font, font_path)
  target_font_np = font2numpy(target_font, font_path)

  print("source font shape :",source_font_np.shape)
  print("target font shape :",target_font_np.shape)

  np.savez("your path", 
         source_fonts=source_font_np, target_fonts=target_font_np)