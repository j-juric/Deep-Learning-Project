import tensorflow as tf
from tensorflow.keras import layers

class StyleDecoder(layers.Layer):
    def __init__(self):
        super(StyleDecoder, self).__init__()

        self.deconv3 = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', activation='relu')
        self.bnorm3 = layers.BatchNormalization()
        
        self.deconv4 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', activation='relu')
        self.bnorm4 = layers.BatchNormalization()
        
        self.deconv5 = layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', activation='tanh')
        
    def call(self, input):
        x = self.bnorm3(self.deconv3(input))
        x = self.bnorm4(self.deconv4(x))
        return self.deconv5(x)