import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(16, (4,4), strides=(2,2), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')
        self.conv3 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')
        self.conv4 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1,activation='sigmoid')

    def call(self, input):
        x = self.conv4(self.conv3(self.conv2(self.conv1(input))))
        return self.fc1(self.flatten(x))