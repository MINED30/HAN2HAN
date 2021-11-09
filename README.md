작성중
 - [x] pipeline code
 - [x] finetuning & inference code
 - [x] embedding module
 - [x] model schematic diagram
 - [ ] explanation
 - [x] demo
 - [x] colab tutorial
 - [x] packaging

<a href="https://colab.research.google.com/github/MINED30/HAN2HAN/blob/main/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/seo-dae-moon.gif"/>

<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/result.gif"/>

<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/ENGFONT.gif"/>

# HAN2HAN
Korean Handwriting Style Font Generation

# Architecture

## Category embedding
<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/CategoryEmbedding.gif"/>

## Backbone of Generator
![image](https://user-images.githubusercontent.com/73981982/140708404-b7fb5e32-27b9-4d06-a473-88fa85075816.png)

## Conditional Pix2Pix
![image](https://user-images.githubusercontent.com/73981982/140708533-b014f9af-4836-47c1-a906-2b7e50ed04be.png)


```
git clone https://github.com/MINED30/HAN2HAN
cd HAN2HAN
mkdir targetimg
bash download.sh
python generate.py
```

### additional
```bash
python utils/Font2Numpy/font2numpy.py # TTF file to numpy array
```
