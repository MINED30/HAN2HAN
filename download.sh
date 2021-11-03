#!/bin/bash
mkdir download
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1DzfVSMJQiowT2In-E-IBrEP1jAYOCu-P" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1DzfVSMJQiowT2In-E-IBrEP1jAYOCu-P" -o ./fonts/source_font.npz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1yGbWUTcIqX9mhMyukK22WeZJ6sgthyTv" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1yGbWUTcIqX9mhMyukK22WeZJ6sgthyTv" -o ./download/gen_weight.pt

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1ih6D0_QwZpR-7g0ME7Uw52yuWN4rV2Rk" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1ih6D0_QwZpR-7g0ME7Uw52yuWN4rV2Rk" -o ./download/character_emb.npz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=12Sdj1Zl5kxQnLZf5EgYF5H1KuIsc7QR9" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=12Sdj1Zl5kxQnLZf5EgYF5H1KuIsc7QR9" -o ./download/category_emb.npz

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1HnHystgvjCZg1kVdbJ8o7YM2pjTPGNmG" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1HnHystgvjCZg1kVdbJ8o7YM2pjTPGNmG" -o ./download/ae_weight.pt
