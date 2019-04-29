#!/bin/bash

GDRIVE_ID=12qsCkQD2QQ9mCKRRqSzWt0COTjsCrbjf

mkdir -p results/pretrained_model

cd results/pretrained_model
echo 'Downloading from id ' ${GDRIVE_ID} '...'
gdown https://drive.google.com/uc?id=${GDRIVE_ID}
