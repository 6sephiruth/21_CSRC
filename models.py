import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import *

from tensorflow.keras import Model, Sequential

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# https://www.tensorflow.org/addons/tutorials/layers_weightnormalization
class cifar10_CNN(Model):
    def __init__(self):
        super(cifar10_CNN, self).__init__()
        self.model = self.build_model()

    def build_model(self):

        # WeightNorm ConvNet
        wn_model = tf.keras.Sequential([
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(6, 5, activation='relu')),
            tf.keras.layers.MaxPooling2D(2, 2),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(16, 5, activation='relu')),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tfa.layers.WeightNormalization(tf.keras.layers.Dense(120, activation='relu')),
            tfa.layers.WeightNormalization(tf.keras.layers.Dense(84, activation='relu')),
            tfa.layers.WeightNormalization(tf.keras.layers.Dense(10, activation='softmax')),
        ])

        return wn_model

class cifar10_KAGGLE(Model):
    def __init__(self):
        super(cifar10_KAGGLE, self).__init__()
        self.call_model = self.build_model()

    def build_model(self):

        cnn_model = tf.keras.Sequential([
            Conv2D(32, (3, 3), padding='same',input_shape=(32, 32, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # CONV => RELU => CONV => RELU => POOL => DROPOUT
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            # FLATTERN => DENSE => RELU => DROPOUT
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            # a softmax classifier
            Dense(10, activation='softmax'),
        ])
        return cnn_model