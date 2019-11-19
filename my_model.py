import tensorflow as tf
import pydot
import graphviz
def AudioOnlyModel(audio_shape = [598,257]):

    ip = tf.keras.layers.Input(shape =(audio_shape[0],audio_shape[1],2))

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(ip)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=1,padding="VALID",activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    #check 2-1 ratio
    x = tf.keras.layers.MaxPool2D( pool_size=[2,1], strides=(2,1))(x)
    
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

from tensorflow.keras.utils import plot_model

my_model = AudioOnlyModel()

def model_summary(model_vars = my_model):
    plot_model(model_vars, to_file='model.png')
    model_vars.summary()

model_summary()


