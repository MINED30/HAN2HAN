from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def custom_img(PATH):

  def white(x):
    if x > 230 :
      return 255
    else :
      return x

  vfunc = np.vectorize(white)
  custom_char = []

  ## Load Image
  character_list = os.listdir(PATH)
  i=1
  plt.figure(figsize=(16,10))
  for c in character_list:
    plt.subplot(6,10,i)
    img = Image.open(os.path.join(PATH,c)).convert("L")
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.4)
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(1.1)
    img = np.asarray(img)
    img= vfunc(img)

    img = np.array(img)
    size = max(img.shape)
    tmp = np.where(img!=255)
    ws,we,hs,he = max(min(tmp[0]-5),0), min(max(tmp[0]+5),size), max(min(tmp[1])-5,0), min(max(tmp[1]+5),size)
    img = img[ws:we,hs:he]

    w, h = img.shape
    m = max(int(w*1.2),int(h*1.2))
    w_left = int((m-w)/2)
    w_right = m-w_left-w
    h_left = int((m-h)/2)
    h_right = m-h_left-h

    img = np.pad(img, ((w_left,w_right), (h_left,h_right)), 'constant', constant_values=255)
    img = Image.fromarray(np.uint8(img))
    img = img.resize((32,32))
    img = np.asarray(img)
    plt.imshow(img, cmap='gray')
    i+=1
    custom_char.append((img,c.split('.')[0]))
  plt.show()

  return custom_char

if __name__ == "__main__":
  print(custom_img("C:/Users/cllhp/Desktop/k"))

