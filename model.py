import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
class DataLoading:
	def __init__(self):
		self.train_csv = "preprocess/avspeech_train.csv"

	@staticmethod
	def load_ids(from_id, to_id, split='train'):
		train_csv = "preprocess/avspeech_train.csv"
		print("Loading IDs ............")
		data = pd.read_csv(train_csv, header = None, names = ["id", "start", "end", "x", "y"])
		ids = set([])
		for i in range(from_id, to_id + 1):
			if (not os.path.isfile('preprocess/' + split + '/spectrograms/' + data.loc[i, "id"] + ".pkl")):
				continue
			elif (not os.path.isfile('preprocess/' + split + '/embeddings/' + data.loc[i, "id"] + ".pkl")):
				continue
			else:
				ids.add(data.loc[i, "id"])
		print("Total " ,len(ids),  split , " IDs Loaded !! ")
		return list(ids)

	@staticmethod
	def split_data(_ids, split = [0.8, 0.1, 0.1]):
		data = np.array(_ids)
		# np.random.shuffle(data)#train on same data everytime
		(valid, test) = (int(split[1]*len(_ids)), int(split[2]*len(_ids)))
		train =  len(_ids) - (valid + test)
		train_split, valid_split, test_split = data[0:train], data[train:train+valid], data[train+valid:]
		print("Total " ,"train:",train, "valid:",valid, "test:", test, " IDs Loaded !! ")
		return train_split, valid_split, test_split

	@staticmethod
	def load_data(_ids, split='train'):
		x_data = np.zeros((len(_ids), 598, 257, 2))
		y_data = np.zeros((len(_ids), 4096))
		for i in range(len(_ids)):
			with open('preprocess/' + split + '/spectrograms/' +  _ids[i] + ".pkl", 'rb') as f:
				x_data[i] = pickle.load(f)
			with open('preprocess/' + split + '/embeddings/' + _ids[i] + ".pkl", 'rb') as f:
				y_data[i] = pickle.load(f)
		return x_data,y_data

	@staticmethod
	def load_Y_data(_ids, split='train'):
		y_data = np.zeros((len(_ids), 4096))
		for i in range(len(_ids)):
			with open('preprocess/' + split + '/embeddings/' + _ids[i] + ".pkl", 'rb') as f:
				y_data[i] = pickle.load(f)
		return y_data

	@staticmethod
	def load_X_data(_ids, split='train'):
		x_data = np.zeros((len(_ids), 598, 257, 2))
		for i in range(len(_ids)):
			with open('preprocess/' + split + '/spectrograms/' +  _ids[i] + ".pkl", 'rb') as f:
				x_data[i] = pickle.load(f)
		return x_data


class AudioEmbeddingModel:
	def __init__(self, from_id, to_id,audio_shape = (598,257,2)):
		self.from_id = from_id
		self.to_id = to_id
		def build_model(audio_shape):
			ip = tf.keras.layers.Input(shape = audio_shape)

			x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(ip)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)
			
			x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

			x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

			x = tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2,1))(x)

			x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=2,padding="VALID",activation="relu")(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			x = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=2,padding="VALID")(x)

			x = tf.keras.layers.AveragePooling2D(pool_size=(6,1),strides=1,padding="VALID")(x)
			x = tf.keras.layers.ReLU()(x)
			x = tf.keras.layers.BatchNormalization(axis=-1)(x)

			flatten = tf.keras.layers.Flatten()(x)
			
			dense = tf.keras.layers.Dense(4096, activation = "relu")(flatten)
			dense = tf.keras.layers.Dense(4096)(dense)

			model = tf.keras.Model(ip, dense)
			return model
		self.model = build_model(audio_shape)
		self._lambda = 1
	
	def multi_gpu_model(self,num_gpu):
		if num_gpu > 1:
			self.model = tf.keras.utils.multi_gpu_model(self.model, gpus = num_gpu)

	def model_summary(self):
		self.model.summary()

	def l2_norm_loss_fn(self,y_true, y_pred):
		return 2 * tf.nn.l2_loss(tf.math.l2_normalize(y_true, axis=1, epsilon=1e-12) - tf.math.l2_normalize(y_pred, axis=1, epsilon=1e-12))

	def loss_fn(self,y_true, y_pred):
		return self._lambda * self.l2_norm_loss_fn(y_true,y_pred)

	def get_loss(num_samples,split='valid'):
		ids = load_ids(self.from_id, self.to_id, split)
		ids_helper = np.array(ids)
		np.random.shuffle(ids_helper)

		for i in range(int(np.ceil(len(ids_helper)/num_samples))):
			x_data, y_data = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])	  
			self.model.predict(x = x_data, y = y_train, epochs = curr_epoch + 1, batch_size = batchsize, verbose = True, initial_epoch = curr_epoch)

	def train(self,train_ids,valid_ids, batchsize, model_save_path,start_epoch = 0, num_epoch = 10, num_samples = 102 ):

		opt_fn = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.5, decay = 0.95/10000, amsgrad=False)
		# If there is less GPU memory then try below
		# opt_fn = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

		self.model.compile(optimizer = opt_fn,loss=self.loss_fn)

		for curr_epoch in range(start_epoch, start_epoch + num_epoch):
			print("Current Epoch : ", curr_epoch)
			ids_helper = np.array(train_ids)
			np.random.shuffle(ids_helper)

			# opt_fn = tf.keras.optimizers.Adam(learning_rate = 0.001*(0.98**curr_epoch), beta_1 = 0.5, decay = 0, amsgrad=False)
			# If there is less GPU memory then try below
			# opt_fn = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
			# self.model.compile(optimizer = opt_fn,loss=self.loss_fn)

			for i in range(int(np.ceil(len(ids_helper)/num_samples))):
				print("i : ",i,end=" ,")
				x_train, y_train = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])	  
				self.model.fit(x = x_train, y = y_train, epochs = curr_epoch + 1, batch_size = batchsize, verbose = True, initial_epoch = curr_epoch)

			ids_helper = np.array(valid_ids)
			Validation_loss=0
			for i in range(int(np.ceil(len(ids_helper)/num_samples))):
				X_val, Y_val = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])
				Validation_loss +=  len(X_val)/batchsize*self.model.evaluate(X_val, Y_val,batch_size=batchsize, verbose=0)
			print("Avg Validation Loss after epoch {} : {}".format(curr_epoch, Validation_loss/len(valid_ids)))

			if (curr_epoch % 2) == 0:
				if model_save_path.endswith('/'):
					self.model.save_weights(model_save_path + 'epoch_' + str(int(curr_epoch)) + '.h5')
				else:
					self.model.save_weights(model_save_path + '/' + 'epoch_' + str(int(curr_epoch)) + '.h5')
				print("Model saved after",curr_epoch,"epoch")
		

	def test(self,test_ids, batchsize, num_samples = 102 ):
		ids_helper = np.array(test_ids)
		Total_loss=0
		opt_fn = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(optimizer = opt_fn,loss=self.loss_fn)
		for i in range(int(np.ceil(len(ids_helper)/num_samples))):
			X_test, Y_test = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])
			Total_loss +=  len(X_val)/batchsize*self.model.evaluate(X_test, Y_test,batch_size=batchsize ,verbose=0)
		print("Avg Test Loss : {}".format(Total_loss/len(test_ids)))
		return Total_loss/len(test_ids)

	def get_top_prediction(self,y_pred,speaker_video_embeddings,k):
		y_pred=y_pred/np.linalg.norm(y_pred,axis=1,keepdims=True)
		return ((speaker_video_embeddings - y_pred)**2).sum(axis=1).argsort()[:k]

	def Test_accuracy(self,test_ids,speaker_ids,batchsize,test_str=' test'):
		ids_helper = np.array(test_ids)
		speaker_video_embeddings = DataLoading.load_Y_data(speaker_ids)
		speaker_video_embeddings=speaker_video_embeddings/np.linalg.norm(speaker_video_embeddings,axis=1,keepdims=True)

		top_pred_for = [1,5,10,25,50,75,100]
		top_n_pred = np.array([0,0,0,0,0,0,0])
		for i in range(len(test_ids)):
			X_test = DataLoading.load_X_data(ids_helper[i:i+1])
			y_pred = self.model.predict(X_test,batch_size=batchsize)
			get_top_k_prediction = self.get_top_prediction(y_pred,speaker_video_embeddings,k=100)
			get_top_k_prediction = [speaker_ids[index] for index in get_top_k_prediction ]
			if test_ids[i] in get_top_k_prediction[0:1]:
				print("best1",test_ids[i],get_top_k_prediction[0:5])
				top_n_pred[0:]+=1
			elif test_ids[i] in get_top_k_prediction[1:5]:
				print("best5",test_ids[i],get_top_k_prediction[0:5])
				top_n_pred[1:]+=1
			elif test_ids[i] in get_top_k_prediction[5:10]:
				top_n_pred[2:]+=1
			elif test_ids[i] in get_top_k_prediction[10:25]:
				top_n_pred[3:]+=1
			elif test_ids[i] in get_top_k_prediction[25:50]:
				top_n_pred[4:]+=1
			elif test_ids[i] in get_top_k_prediction[50:75]:
				top_n_pred[5:]+=1
			elif test_ids[i] in get_top_k_prediction[75:100]:
				top_n_pred[6:]+=1
			else :
				print("not in top100",test_ids[i],get_top_k_prediction[0:5])


		print("\ntop",top_pred_for,":",top_n_pred,"for ",len(test_ids),test_str+" IDs and ",len(speaker_video_embeddings),"Embeddings")
		return top_n_pred


	def get_L1_Loss(self,y_pred,y_true):
		a1= (abs(y_true - y_pred)**2).sum()
		y_pred=y_pred/np.linalg.norm(y_pred,axis=1,keepdims=True)
		y_true=y_true/np.linalg.norm(y_true,axis=1,keepdims=True)
		a2= (abs(y_true - y_pred)**2).sum()
		return a1,a2

	def get_L1_L2_loss(self,test_ids,batchsize,test_str=' test', num_samples = 100):
		ids_helper = np.array(test_ids)
		Total_loss1,Total_loss2=0,0
		for i in range(int(np.ceil(len(ids_helper)/num_samples))):
			X_val, Y_val = DataLoading.load_data(ids_helper[(i*num_samples): ((i+1)*num_samples)])
			y_pred = self.model.predict(X_val,batch_size=batchsize)
			a,b = self.get_L1_Loss(y_pred,Y_val)
			Total_loss1+=a
			Total_loss2+=b
		print("Total_loss1,2:",Total_loss1/(len(test_ids)),Total_loss1/(len(test_ids)))
		print("\ntop",top_pred_for,":",top_n_pred,"for ",len(test_ids),test_str+" IDs and ",len(speaker_video_embeddings),"Embeddings")
		return Total_loss1,Total_loss2


	def load_weights(self, path):
		self.model.load_weights(path, by_name=True)



