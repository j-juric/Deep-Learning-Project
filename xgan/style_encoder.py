import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class StyleEncoder(layers.Layer):
    def __init__(self):
        super(StyleEncoder, self).__init__()

        self.conv1 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same')
        self.bnorm1 = InstanceNormalization()
        self.lrelu1 = LeakyReLU(0.3)

        self.conv2 = layers.Conv2D(64, (4,4), strides=(2,2), padding='same')
        self.bnorm2 = InstanceNormalization()
        self.lrelu2 = LeakyReLU(0.3)

    def call(self, input):
        x = self.lrelu1(self.bnorm1(self.conv1(input)))
        x = self.lrelu2(self.bnorm2(self.conv2(x)))
        return x