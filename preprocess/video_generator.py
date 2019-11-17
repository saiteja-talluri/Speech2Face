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
from facenet.src.download_and_extract import *
from facenet.src.facenet import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class VideoExtract():

    def __init__(self, fps, duration, video_part, face_extraction_model):

        self.destination_dir = "data/speaker_video_embeddings/part_" + str(video_part) + "/"
        self.videos = "data/videos/"
        self.frames_dir = "data/frames/"
        self.model_dir = "data/pretrained_model/"

        self.fps = fps
        self.duration = duration
        self.face_extraction_model = face_extraction_model

        if not os.path.isdir(self.destination_dir):
            os.mkdir(self.destination_dir)

        if not os.path.isdir(self.frames_dir):
            os.mkdir(self.frames_dir)

        if not os.path.isfile(self.model_dir + "20180402-114759.zip"):
            download_and_extract_file(model_name = "20180402-114759", data_dir = self.model_dir)

    def extract_video(self, id, x, y):

        with tf.Graph().as_default():
            with tf.Session() as sess:
                load_model(self.model_dir + "20180402-114759/")
                ip = tf.get_default_graph().get_tensor_by_name("input:0")
                embedding = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                embeddings = np.zeros((6, 512))

                if not os.path.isfile(self.videos + id + ".mp4"):   # if video in csv but not in dataset -> wasn't available on youtube
                  print("--------Video {} not found-----------".format(self.videos + id + ".mp4"))
                  return 1

                if (not os.path.isfile(self.destination_dir + id + ".pkl")):
                    # Resample 1 fps, 6 seconds
                    print("Resampling video", id)
                    resample = "ffmpeg -y -i {1}{2}.mp4 -r {0} -t {3} '{4}{2}.mp4'".format(self.fps, self.videos, id, self.duration, self.destination_dir)
                    print(resample)
                    res2 = subprocess.Popen(resample, stdout = subprocess.PIPE, shell=True).communicate()

                    if not os.path.isfile(self.destination_dir + id  + ".mp4"):  # Fault in encoding of video so skip it
                      print("--------Fault in video {}--------".format(id))
                      return 1

                    # save 6 frames in frames
                    extract_frames = "ffmpeg -i '{0}{1}.mp4' {2}/%02d.jpg".format(self.destination_dir, id, self.frames_dir)
                    rs = subprocess.Popen(extract_frames, stdout = subprocess.PIPE, shell = True).communicate()

                    # detect faces and crop rectangle
                    for j in range(1,7):

                        if not os.path.isfile(self.frames_dir + "%02d" % j + ".jpg"):    # If all 6 frames are not recovered by mpeg
                          print("------MISSING FRAME DETECTED FOR {} FRAME NO {}----".format(id, j))
                          return 1

                        print("reading frame, {0}".format(j))
                        frame = Image.open(self.frames_dir + "%02d" % j + ".jpg")
                        face_boxes = face_recognition.face_locations(np.array(frame), model= self.face_extraction_model)

                        if(len(face_boxes) > 1):  # if 2 faces are detected,then take center of box from csv and find nearest center from 2 boxes
                            print("-----2 faces detected in {0} frame {1}-----".format(data.loc[i, "id"], j))

                            min_box_distance = 99999999
                            min_box_index = 0

                            orig_center_x = x * np.array(frame).shape[1]  # un-normalize original center by multiplying 1280 and 720
                            orig_center_y = y * np.array(frame).shape[0]

                            for box_index, box in enumerate(face_boxes):
                                top, right, bottom, left = np.squeeze(box)

                                box_center_x = left+ (right - left)/ 2
                                box_center_y = top+ (bottom - top)/ 2
                                distance = np.sqrt(np.square(box_center_x - orig_center_x) + np.square(box_center_y - orig_center_y))    #d(P, Q) = √ (x2 − x1)2 + (y2 − y1)2
                                if distance < min_box_distance:
                                    min_box_distance = distance
                                    min_box_index = box_index

                            print("Closest face for {0} is {1} from {2}".format((orig_center_x, orig_center_y), min_box_index, face_boxes))
                            face_boxes = face_boxes[min_box_index]  # Select closest face

                        if len(face_boxes) == 0:   # No face detected
                            print("-----No face detected in {} frame {}-----".format(id, j))
                            return 1
                        top, right, bottom, left = np.squeeze(face_boxes)

                        frame_cropped = frame.crop(box = (left, top, right, bottom))

                        # give frames to inception network and add in embedding array
                        frame_resized = scipy.misc.imresize(np.array(frame_cropped), size = (160,160)) # for embedding model
                        frame_resized = np.expand_dims(np.array(frame_resized), 0) # for batch dim

                        feed_dict = {ip : frame_resized, train_placeholder: False}
                        embeddings[j-1, :] = sess.run(embedding, feed_dict= feed_dict)  # -1 for j starts from 1 for numbers for frames


                pickle.dump(embeddings, open(self.destination_dir + id + ".pkl", "wb"))
                # Delete frames and videos
                delete_frames = "rm {0}*".format(self.frames_dir)
                delete_video = "rm '{0}'".format(self.destination_dir + id + ".mp4")
                rs = subprocess.Popen(delete_frames, stdout = subprocess.PIPE, shell = True).communicate()
                rs = subprocess.Popen(delete_video, stdout = subprocess.PIPE, shell = True).communicate()

        print("----DONE----")
        return 0
