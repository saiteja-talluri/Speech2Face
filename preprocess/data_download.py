import pandas as pd
import os
import subprocess
import argparse
from speaker import *
from video_generator import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
FNULL = open(os.devnull, 'w')

def main():

	videos_path = "data/videos/"
	audios_path = "data/audios/"
	data = pd.read_csv("avspeech_train.csv", header = None, names = ["id", "start", "end", "x", "y"])

	parser = argparse.ArgumentParser()
	parser.add_argument("--from_id", type = int, default = 30000)
	parser.add_argument("--to_id", type = int, default = 80000)
	parser.add_argument("--low_memory", type = str, default = "no")
	parser.add_argument("--sample_rate", type = int, default = 16000)
	parser.add_argument("--duration", type = int, default = 6)
	parser.add_argument("--fps", type = int, default = 1)
	parser.add_argument("--mono", type = str, default = True)
	parser.add_argument("--window", type = int, default = 400)
	parser.add_argument("--stride", type = int, default = 160)
	parser.add_argument("--fft_length", type = int, default = 512)
	parser.add_argument("--amp_norm", type = int, default = 0.3)
	parser.add_argument("--face_extraction_model", type = str, default = "cnn")
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args()

	sb = Speaker(sample_rate = args.sample_rate, duration = args.duration, mono = args.mono, window = args.window,
		stride = args.stride, fft_length = args.fft_length, amp_norm = args.amp_norm, verbose=args.verbose)

	vs = VideoExtract(args.fps, args.duration, args.face_extraction_model, args.verbose)

	for i in range(args.from_id, args.to_id):
		if (not os.path.isfile(videos_path + data.loc[i, "id"] + ".mp4")):
			if args.verbose:
				print("downloading", data.loc[i, "id"], data.loc[i, "start"], data.loc[i, "end"])
			url = "youtube-dl -f best --get-url https://www.youtube.com/watch?v=" + str(data.loc[i, "id"])
			res1 = subprocess.run(url, stdout = subprocess.PIPE, shell=True).stdout.decode("utf-8").rstrip()
			if(res1 == ""):
				if args.verbose:
					print("----------------------Video not available---------------------")
				continue

			download = "ffmpeg" + " -nostats -loglevel 0 -ss " + str(data.loc[i,"start"]) + " -i \"" + res1 + "\"" + " -t " + str(min(args.duration, float(data.loc[i, "end"]) - float(data.loc[i, "start"]))) + " -c:v copy -c:a copy " + videos_path + str(data.loc[i,"id"]) + ".mp4"
			res2 = subprocess.Popen(download, stdout = FNULL, shell=True).communicate()
		else:
			if args.verbose:
				print("Skipping ", i)

		result = vs.extract_video(data.loc[i, "id"], data.loc[i, "x"], data.loc[i, "y"])
		if result == 1 :
			if args.verbose:
				print("video embedding extraction failed see logs for reason")
			continue

		error = sb.extract_wav(data.loc[i, "id"])
		if error == 1:
			if args.verbose:
				print("========extract wav failed skipping getting audio or video extraction=======")
			continue

	print("Done creating dataset")
	if (args.low_memory == "yes"):
		delete_audios = "rm {0}*".format(self.audios_path)
		delete_videos = "rm {0}*".format(self.videos_path)
		rs = subprocess.Popen(delete_audios, stdout = subprocess.PIPE, shell = True).communicate()
		rs = subprocess.Popen(delete_videos, stdout = subprocess.PIPE, shell = True).communicate()

if __name__ == "__main__":
	main()
