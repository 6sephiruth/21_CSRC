import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import *

from tensorflow.keras import Model, Sequential


# https://www.tensorflow.org/addons/tutorials/layers_weightnormalization
class cifar10_CNN(Model):
    def __init__(self):
        super(cifar10_CNN, self).__init__()
        self.model = self.build_model()
        self.wow = 4
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


