#!/bin/bash


rm -r -f data/speaker_video_embeddings/
rm -r -f data/audios
rm -r -f data/videos
rm -r -f data/audio_spectrograms
rm -r -f data/frames
rm -r -f data/cropped_frames

mkdir -p data/audios
mkdir -p data/videos
mkdir -p data/audio_spectrograms
mkdir -p data/frames
mkdir -p data/speaker_video_embeddings/
mkdir -p data/cropped_frames

echo "Done cleaning up and setting up directories again"