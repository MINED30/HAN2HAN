#!/bin/bash
mkdir download
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1jNWqamF_umHtpZcZcHeCAoY23eSemxhf" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1jNWqamF_umHtpZcZcHeCAoY23eSemxhf" -o ./download/download.zip
unzip ./download/download.zip -d ./download/