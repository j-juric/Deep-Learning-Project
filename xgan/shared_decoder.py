import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class SharedDecoder(layers.Layer):
    def __init__(self):
        super(SharedDecoder, self).__init__()

        self.fc = layers.Dense(2*2*1024, input_shape=(1024,))
        self.reshape = layers.Reshape((2, 2, 1024))
        
        self.deconv1 = layers.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same')
        self.bnorm1 = InstanceNormalization()
        self.lrelu1=LeakyReLU(0.3)

        self.deconv2 = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')
        self.bnorm2 = InstanceNormalization()
        self.lrelu2=LeakyReLU(0.3)

    def call(self, shared_embedding):

        x = self.reshape(self.fc(shared_embedding))
        x = self.lrelu1(self.bnorm1(self.deconv1(x)))
        x = self.lrelu2(self.bnorm2(self.deconv2(x)))
        return x