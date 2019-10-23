import os
import librosa
import functools
import pickle
import shutil

import tensorflow as tf
from tensorflow.python.ops import io_ops

class Speaker():

	def __init__(self, sample_rate = 16000, duration = 6, mono = True, window = 400, stride = 160, fft_length = 512, amp_norm = 0.3):

		self.videos_path = "data/videos/"
		self.audios_path = "data/audios/"
		self.spect_path  = "data/audio_spectrograms/"

		self.sample_rate = sample_rate
		self.duration = duration
		self.mono = mono

		self.window = window
		self.stride = stride
		self.fft_length = fft_length
		self.amp_norm = amp_norm

	def find_spec(self, filename):
		print("-------------finding spectrogram for {filename}----------------")
		with tf.compat.v1.Session(graph=tf.Graph()) as sess:

			holder = tf.compat.v1.placeholder(tf.string, [])
			file = tf.io.read_file(holder)
			decoder = tf.contrib.ffmpeg.decode_audio(file, file_format = "wav", samples_per_second = self.sample_rate, channel_count = 1)

			stft = tf.signal.stft(tf.transpose(a=decoder), frame_length = self.window, frame_step = self.stride, fft_length = self.fft_length, window_fn = tf.signal.hann_window)

			amp = tf.squeeze(tf.abs(stft)) ** self.amp_norm
			phase = tf.squeeze(tf.math.angle(stft))

			stacked = tf.stack([amp, phase], 2)

			stft = sess.run(stacked, feed_dict = {holder : self.spect_path + filename + ".wav"})
			pickle.dump(stft, open(self.spect_path + filename  + ".pkl", "wb"))
			
			print("============STFT SHAPE IS {0}=============".format(stft.shape))

	def extract_wav(self, filename):
		print("-----------extracting audio-------------")
		wavfile = filename + ".wav"

		if (not os.path.isfile(self.spect_path + wavfile)):

			os.popen("ffmpeg -t " + str(self.duration) + " -stream_loop -1  -i " + self.videos_path + filename + ".mp4" + " -vn " + self.spect_path + "/" + wavfile).read()	# Extract audio
			
			if(not os.path.isfile(self.spect_path + wavfile)):
				print("----------------ffmpeg can't extract audio so deleting --------------")
				os.remove(self.spect_path + "/" + wavfile)
				return 1

			data, _ = librosa.load(self.spect_path + wavfile, sr = self.sample_rate, mono = self.mono, duration = self.duration)
			librosa.output.write_wav(self.spect_path + wavfile, data, self.sample_rate, norm = False)
			librosa.output.write_wav(self.audios_path +  wavfile, data, self.sample_rate, norm = False)    

			self.find_spec(filename)
			os.remove(self.spect_path + wavfile)

		else:
			print("skipping audio extraction for {0}".format(filename))