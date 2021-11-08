작성중
 - [x] pipeline code
 - [x] finetuning & inference code
 - [x] embedding module
 - [ ] model schematic diagram
 - [ ] explanation
 - [ ] demo
 - [x] colab tutorial
 - [x] packaging

<a href="https://colab.research.google.com/github/MINED30/HAN2HAN/blob/main/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# HAN2HAN
Korean Handwriting Style Font Generation

![image](https://user-images.githubusercontent.com/73981982/140700211-0b492a65-10c2-4dac-960e-926bc159dd7b.png)

![image](https://user-images.githubusercontent.com/73981982/140700225-8dbb170a-bcd3-4d1d-be84-0a90c8a9a2c0.png)

![image](https://user-images.githubusercontent.com/73981982/140700237-5711483e-a6ea-4359-9691-dde5b3e8bc32.png)

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

## 결과물 (임시)

### Input
![image](https://user-images.githubusercontent.com/73981982/138725507-fa104664-bbed-47a5-b125-614a5348f70c.png)


### 10글자('나랏말싸미 듕귁에달아')로 만들어낸 윤동주 시인의 서시


![image](https://user-images.githubusercontent.com/73981982/138566749-9933493e-b29a-45a6-999e-314b33f3f3b8.png)
