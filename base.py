import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import argparse
import os
from tensorflow.python.client import device_lib
from model import AudioEmbeddingModel, DataLoading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    count = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    print("Number of GPU : ", count)

def test_id_loss(_id,string):
    x_train, y_train = load_data(ids[_id:_id+1])
    y_predicted = my_model.predict(x_train)
    print(string, loss(y_predicted,y_train[-1:]))   


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--from_id", type = int, default = 0)
	parser.add_argument("--to_id", type = int, default = 100000)

	parser.add_argument("--epochs", type = int, default = 3)
	parser.add_argument("--start_epoch", type = int, default = 39)
	parser.add_argument("--batchsize", type = int, default = 1)
	parser.add_argument("--num_gpu", type = int, default = 1)
	parser.add_argument("--num_samples", type = int, default = 100)

	parser.add_argument("--load_model", type = str, default = "models/final.h5")
	parser.add_argument("--save_model", type = str, default = "models/")
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--verbose", action="store_true")
	args = parser.parse_args()

	audemb_model = AudioEmbeddingModel(from_id=args.from_id,to_id=args.to_id)

	if args.verbose:
		get_available_gpus()
		audemb_model.model_summary()
	
	if args.num_gpu > 1:
	    audemb_model.multi_gpu_model(args.num_gpu)

	if len(args.load_model) != 0:
		if os.path.isfile(args.load_model):
			audemb_model.load_weights(args.load_model)
			print("Model Loaded")
		else:
			print("Model not present in the given path")
			exit()

	if len(args.save_model) == 0:
		print("Path to save the model is not specified")
		exit()
	else :
		if not os.path.exists(args.save_model):
			os.makedirs(args.save_model)
	
	ids = DataLoading.load_ids(args.from_id, args.to_id)
	
	train_ids ,valid_ids,test_ids = DataLoading.split_data(ids)

	valid_ids,test_ids=test_ids,valid_ids

	if args.train:
		audemb_model.train(train_ids,valid_ids,args.batchsize,args.save_model,start_epoch = args.start_epoch, num_epoch = args.epochs, num_samples = args.num_samples)
	
	audemb_model.get_L1_L2_loss(test_ids,batchsize=args.batchsize,test_str=' test')
	audemb_model.get_L1_L2_loss(train_ids,batchsize=args.batchsize,test_str=' train')
	audemb_model.Test_accuracy(train_ids,ids,batchsize=args.batchsize,test_str=' train')
	audemb_model.Test_accuracy(valid_ids,ids,batchsize=args.batchsize,test_str=' valid')
	audemb_model.Test_accuracy(test_ids,ids,batchsize=args.batchsize,test_str=' test')

if __name__ == "__main__":
	main()