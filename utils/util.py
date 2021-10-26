from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from utils.Font2Numpy.FontTest import common_han
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os

def custom_img(PATH:"image_dir"):
  '''
  Function to crop & resize & enhance <your image>
  '''

  def whiteing(x):
    if x > 230 :
      return 255
    else :
      return x

  vfunc = np.vectorize(whiteing)
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


def knock_the_door(PATH:"character_emb",embed_word):
    cos_embedded = np.load(PATH)
    cos_embed = cos_embedded['embed']
    cos_label = cos_embedded['label']
    dic = {}
    for i in range(2402):
      find_cos = cosine_similarity(cos_embed[i:i+1],embed_word)
      best_cos = np.argmax(find_cos)
      print(common_han[i],char_labels[best_cos], best_cos, find_cos[0][best_cos])
      dic[common_han[i]] = best_cos
    return dic