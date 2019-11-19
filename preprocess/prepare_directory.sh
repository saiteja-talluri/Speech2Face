#!/bin/bash

mkdir -p data
mkdir -p data/speaker_video_embeddings/

mkdir -p data/audios
mkdir -p data/videos
mkdir -p data/audio_spectrograms
mkdir -p data/frames
mkdir -p data/pretrained_model
mkdir -p data/cropped_models

echo "Done setting up directories set"

pip install -r ../requirements.txt

git clone https://github.com/davidsandberg/facenet.git facenet/
pip install face_recognition
sudo apt-get --assume-yes install ffmpeg
sudo apt-get --assume-yes --fix-missing install youtube-dl

echo "Done setting up directories and environment, now proceed to download dataset...."
