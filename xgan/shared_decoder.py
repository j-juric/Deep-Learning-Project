import tensorflow as tf
from tensorflow.keras import layers

class SharedDecoder(layers.Layer):
    def __init__(self):
        super(SharedDecoder, self).__init__()

        self.fc = layers.Dense(2*2*1024, input_shape=(1024,))
        self.reshape = layers.Reshape((2, 2, 1024))
        
        self.deconv1 = layers.Conv2DTranspose(512, (2,2), strides=(1,1), padding='same', activation='relu')
        self.bnorm1 = layers.BatchNormalization()

        self.deconv2 = layers.Conv2DTranspose(256, (2,2), strides=(1,1), padding='same', activation='relu')
        self.bnorm2 = layers.BatchNormalization()

    def call(self, shared_embedding):

        x = self.reshape(self.fc(shared_embedding))
        x = self.bnorm1(self.deconv1(x))
        x = self.bnorm2(self.deconv2(x))
        return x