import os
import pickle
import shutil
import imageio
import pandas as pd
import subprocess
from PIL import Image
import face_recognition
import numpy as np
import skimage
import scipy
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
FNULL = open(os.devnull, 'w')

class VideoExtract():

    def __init__(self, fps, duration, face_extraction_model, verbose):

        self.destination_dir = "data/speaker_video_embeddings/"
        self.videos = "data/videos/"
        self.frames_dir = "data/frames/"
        self.frame_cropped = "data/cropped_frames/"
        self.model_dir = "data/pretrained_model/"
        self.fps = fps
        self.duration = duration
        self.face_extraction_model = face_extraction_model
        self.vgg = VGGFace(model='vgg16')
        self.out = self.vgg.get_layer('fc7').output
        self.vgg_model = Model(self.vgg.input, self.out)
        self.verbose = verbose

        if not os.path.isdir(self.destination_dir):
            os.mkdir(self.destination_dir)

        if not os.path.isdir(self.frames_dir):
            os.mkdir(self.frames_dir)

    def extract_video(self, id, x, y):
        embeddings = np.zeros((4096))
        if not os.path.isfile(self.videos + id + ".mp4"):
            if self.verbose:
                print("--------Video {} not found-----------".format(self.videos + id + ".mp4"))
            return 1

        if (not os.path.isfile(self.destination_dir + id + ".pkl")):
            
            if self.verbose:
                print("Resampling video", id)
            resample = "ffmpeg -nostats -loglevel 0 -y -i {1}{2}.mp4 -r {0} -t {3} '{4}{2}.mp4'".format(self.fps, self.videos, id, self.duration, self.destination_dir)
            res2 = subprocess.Popen(resample, stdout = FNULL, shell=True).communicate()

            if not os.path.isfile(self.destination_dir + id  + ".mp4"):
                if self.verbose:
                    print("--------Fault in video {}--------".format(id))
                return 1

            extract_frames = "ffmpeg -nostats -loglevel 0 -i '{0}{1}.mp4' {2}/%02d.jpg".format(self.destination_dir, id, self.frames_dir)
            rs = subprocess.Popen(extract_frames, stdout = FNULL, shell = True).communicate()

            for j in range(1, 7):

                if not os.path.isfile(self.frames_dir + "%02d" % j + ".jpg"):
                    if self.verbose:
                        print("------MISSING FRAME DETECTED FOR {} FRAME NO {}----".format(id, j))
                    continue

                if self.verbose:
                    print("reading frame - {0}".format(j))
                frame = Image.open(self.frames_dir + "%02d" % j + ".jpg")
                face_boxes = face_recognition.face_locations(np.array(frame), model= self.face_extraction_model)

                if(len(face_boxes) > 1):
                    if self.verbose:
                        print("-----2 faces detected in {0} frame {1}-----".format(id, j))
                        return 1

                elif len(face_boxes) == 0:
                    if self.verbose:
                        print("-----No face detected in {} frame {}-----".format(id, j))
                    return 1
                    
                top, right, bottom, left = np.squeeze(face_boxes)
                frame_cropped = frame.crop(box = (left, top, right, bottom))

                frame_resized = scipy.misc.imresize(np.array(frame_cropped), size = (224,224))
                Image.fromarray(frame_resized).save(self.frame_cropped + id + '.jpg')
                frame_resized = np.expand_dims(np.array(frame_resized, dtype=np.float64), 0)
                frame_resized = utils.preprocess_input(frame_resized, version=1)
                embeddings = self.vgg_model.predict(frame_resized)
                break
               
            pickle.dump(embeddings, open(self.destination_dir + id + ".pkl", "wb"))
            
            delete_frames = "rm {0}*".format(self.frames_dir)
            delete_video = "rm '{0}'".format(self.destination_dir + id + ".mp4")
            rs = subprocess.Popen(delete_frames, stdout = subprocess.PIPE, shell = True).communicate()
            rs = subprocess.Popen(delete_video, stdout = subprocess.PIPE, shell = True).communicate()

        return 0
