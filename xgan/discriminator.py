import tensorflow as tf
from tensorflow.keras import layers
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class Discriminator(layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(16, (4,4), strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU(0.3))
        #self.drop1 = layers.Dropout(0.2)
        self.conv2 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU(0.3))
        self.bnorm2 = InstanceNormalization()
        self.conv3 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU(0.3))
        self.bnorm3 = InstanceNormalization()
        self.conv4 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation=tf.keras.layers.LeakyReLU(0.3))
        #self.drop4 = layers.Dropout(0.2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1,activation='sigmoid')

    def call(self, input):
        x = self.conv1(input)
        x = self.bnorm2(self.conv2(x))
        x = self.bnorm3(self.conv3(x))
        x = self.fc1(self.flatten(self.conv4(x)))
        return x