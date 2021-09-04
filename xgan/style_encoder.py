import tensorflow as tf
from tensorflow.keras import layers

class StyleEncoder(layers.Layer):
    def __init__(self):
        super(StyleEncoder, self).__init__()

        self.conv1 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')


        self.conv2 = layers.Conv2D(64, (4,4), strides=(2,2), padding='same', activation='relu')

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x