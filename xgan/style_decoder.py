import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class StyleDecoder(layers.Layer):
    def __init__(self):
        super(StyleDecoder, self).__init__()

        self.deconv3 = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')
        self.bnorm3 = InstanceNormalization()
        self.lrelu1 = LeakyReLU(0.3)
        
        self.deconv4 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')
        self.bnorm4 = InstanceNormalization()
        self.lrelu2 = LeakyReLU(0.3)
        
        self.deconv5 = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', activation='tanh')
        
    def call(self, input):
        x = self.lrelu1(self.bnorm3(self.deconv3(input)))
        x = self.lrelu2(self.bnorm4(self.deconv4(x)))
        return self.deconv5(x)