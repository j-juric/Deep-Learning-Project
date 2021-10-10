import tensorflow as tf
from tensorflow.keras import layers

class Denoiser(layers.Layer):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.conv1 = layers.Conv2D(64,(3,3),padding='same', activation='relu', input_shape=(64,64,3))
        self.maxpool = layers.MaxPool2D(padding='same')
        self.conv2 = layers.Conv2D(64,(3,3),padding='same', activation='relu')
        self.upsample = layers.UpSampling2D()
        self.conv3 = layers.Conv2D(3,(3,3),padding='same',activation='tanh')


    def call(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)

        return x