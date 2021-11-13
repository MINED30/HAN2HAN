

<a href="https://colab.research.google.com/github/MINED30/HAN2HAN/blob/main/colab_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# HAN2HAN : Hangul Font Generation

```
git clone https://github.com/MINED30/HAN2HAN
cd HAN2HAN
mkdir targetimg
bash download.sh
python generate.py
```

## Results

### Letter from Seodaemun Prison
<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/seo-dae-moon.gif"/>
Generation hangul font from alphabet (gif)
(Above : Hello Snow Font / Below : Spooky Christmas Font)
Reference : https://www.1001fonts.com/

The above picture is an example of creating a font by extracting two sentences, "Are you doing well at school(그새 학교 잘다니냐)" and "I'm fine(나는 잘있다)," from the letter sent by Lee Yeon-ho at Seodaemun Prison. There are 13 letters in the extracted sentence, but 11 letters are actually usable by the model due to overlapping of 'Jal(잘)' and 'Da(다)'. Lee Yeon-ho's handwriting is characterized by a gentle flow and a clean feel. In the created 'Song of Cell No. 8 <Daehan Lived(대한이 살았다)>', it can be seen that the characteristics are well lived. In particular, it can be seen that 'ㅎ' and 'ㅇ' are well utilized in the initial consonant, and the 'ㄹ' and 'ㄴ' consonants in the middle and final consonants are also well characterized. On the other hand, it is confirmed that the initial consonant 'ㄷ' is blurred and the font is cut off, and that 'ㅗ' is not well implemented because the font of '그' is unique.
### Generation Hangul Font from Alphabet
<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/ENGFONT.gif"/>
Generation font from 10 chracters of other fonts
(Solmoe Kim Dae-geonche (top left), Jeongseon Arirang font (top right), KOTRA handwritten font (bottom left), Happy Goheung font (bottom right))

What is surprising is that it captures the characteristics of the English alphabet well and puts Hangul fonts on it. I got the font from the website. The font above is snow piled on top of the font, and the font below is a horror style font. It is not perfectly created, but it follows the thickness of the font and the softness of the font well. In particular, it was very surprising to see the eyes gradually taking shape as the number of epochs increased in the font above.
### Other fonts

<img src="https://github.com/MINED30/HAN2HAN/blob/main/demo/result.gif"/>
Generation font from 10 chracters of other fonts
(Solmoe Kim Dae-geonche (top left), Jeongseon Arirang font (top right), KOTRA handwritten font (bottom left), Happy Goheung font (bottom right))

The above 4 fonts are not used for training. It takes 10 characters of '나랏말싸미 듕귁에달아' and generates the rest of the Hunminjeongeum. You can see that it is generate so well that it is difficult for the human to distinguish it.
## Architecture

### Category Embedding
![1 category embedder](https://user-images.githubusercontent.com/73981982/140964445-af7ac346-437d-45e4-910b-c292b3c15586.gif)
In this project, the source font plays the role of 'Condition' that tells what type of character it is. Therefore, the source font should be able to change to any style and not lose the character of the characteristic. In the example picture above, it should be able to change to a different style of '밝', and at the same time, the characteristic of '밝' should not be lost. If a characteristic is lost, it cannot function as a Condition. By reconstructing the text whose style has changed in the source font, it is possible to better maintain the characteristic of the text. This structure was inspired by CycleGAN.

In general, the last embedding value of the encoder is used as the category embedding. However, this model uses all layers, not just the last embedding layer. That is, all values (red in the image) encoded in the source font are reflected when creating the font. For this purpose, the generator and encoder structures are designed identically.

### Pix2Pix - Generator part
<img src="https://user-images.githubusercontent.com/73981982/140964533-13fdabe0-2196-4a0a-a8e2-91009cebde23.png" height="350">
It is created in the same form as U-NET, but the difference is that category embedding is added. The encoded values are concatenated in the decoder part at the back stage, and the embedded values are also concatenated at this time. The concat values are listed below.

1. Feature map from encoder
2. The upconvolutional feature map in the previous step
3. Category embedded feature map


### Pix2Pix - GAN Part

<img src="https://user-images.githubusercontent.com/73981982/140965094-839ba148-71ff-41b3-a368-ac44ac52f259.png" height="350">

The important thing is to be able to train with the font according to the embedding value of the source font, even if you put a different font. In the picture above, when the source font is '밝' regardless of '창' or any character, the generator should create '밝'. The discriminator simply classifies it as 0(True Image) or 1(Fake Image). The generator is trained to deceive the discriminator, and the discriminator is trained to pick out the real image. PatchGAN seems meaningless because the size of the image was small as 32x32. The pre-trained model learned 138 Naver nanum fonts, trained the generator 30 epochs, then trained the discriminator 30 epochs independently and run 30 epochs together.

In Finetuning, you have to create a lot of characters beyond just 10 characters. For example, when finetuning using 10 characters, each character is trained to make 1 character by putting 9 characters. That is, you can train 90 times (10*9 times) with 10 characters. L1 Loss was used as the loss function, and the results were good when the learning rate was assigned to 4e-5~5e-5 and trained for 100~200 epochs.

### Character Embedding

For example, when 10 characters exist as input values, there are 10 cases to generate one character as a source font. It can be generated by selecting it randomly, but I hypothesize that it would be better if it was entered as the source font by inserting related characters.
AutoEncoder was used to embed characters

# Strength

- A font is created with only a small number of characters, less than 10 characters.
- Even if you put the alphabet, it generates Hangul font well, and as you can see in the example, it is possible to express the snowy font.
- Using Google Colab's GPU
- The cost of designing fonts can be lowered.
# Weakness

1. Part of some characters disappears.
    - A small number of characters
    - If the format deviates a lot from the previously trained data, having it has a negative effect. (In the letter sent from Seodaemun Prison, it has better result without '그')
        - The first reason and the second reason conflict with each other. In conclusion, it is thought that data of 'consistent font' is necessary.
2. The number of characters that can be created is limited to 2420 characters.
    - With only 2420 characters of 138 fonts, the number of characters that can be generated by training is limited to 2420 characters.
        - Better performance is expected if pretraining is performed with 11,172 characters and more fonts.
    - 2420 characters of 138 fonts are too small to cluster by character. If you train more characters, I think character embeddings will be more effective.
    - The more characters, the harder to category embed
        - it is more important to increase the number of fonts.
3. Low quality by learning and generating images with a small size of 32x32
    - It is possible to generate high-quality fonts if trained to a larger size using more resources in the future.
4. the effectiveness of character embedding is not proved
    - The results, although matching seem better, have not been proven numerically.
    - For now, model matches with cosine similarity, but it may be more effective if it is further advanced (Nural Net, etc.) in the future.
5. discrepancy
    - The training is a digital font, but if it is created with an image such as a camera, discrepancy occurs.
    - Most of the designed fonts have a certain typeface, but a real person sometimes uses a typeface that is different from their own, causing discrepancy.
        - However, as in the example of 'Correspondence from Seodaemun Prison', if you 'font' the characters through preprocessing before generating them, there is no significant hindrance to the creation performance.
