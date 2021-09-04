import tensorflow as tf
from tensorflow.keras import layers

class StyleDecoder(layers.Layer):
    def __init__(self):
        super(StyleDecoder, self).__init__()

        self.deconv3 = layers.Conv2DTranspose(128, (2,2), strides=(1,1), padding='same', activation='relu')
        self.bnorm3 = layers.BatchNormalization()

        self.deconv4 = layers.Conv2DTranspose(64, (2,2), strides=(1,1), padding='same', activation='relu')
        self.bnorm4 = layers.BatchNormalization()

        self.deconv5 = layers.Conv2DTranspose(3, (2,2), strides=(1,1), padding='same', activation='relu')

    def call(self, input):
        x = self.bnorm3(self.deconv3(input))
        x = self.bnorm4(self.deconv4(input))
        return self.deconv5(x)