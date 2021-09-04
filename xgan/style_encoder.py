import tensorflow as tf
from tensorflow.keras import layers

class StyleEncoder(layers.Layer):
    def __init__(self):
        super(StyleEncoder, self).__init__()

        self.conv1 = layers.Conv2D(32, (2,2), strides=(1,1), padding='same', activation='relu')
        self.bnorm1 = layers.BatchNormalization()


        self.conv2 = layers.Conv2D(64, (2,2), strides=(1,1), padding='same', activation='relu')
        self.bnorm2 = layers.BatchNormalization()

    def call(self, input):
        x = self.bnorm1(self.conv1(input))
        x = self.bnorm2(self.conv2(x))
        return x