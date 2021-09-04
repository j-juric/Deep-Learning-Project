import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(16, (4,4), strides=(2,2), padding='same', activation='relu')
        self.drop1 = layers.Dropout(0.2)
        self.conv2 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')
        self.drop2 = layers.Dropout(0.2)
        self.conv3 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')
        self.drop3 = layers.Dropout(0.2)
        self.conv4 = layers.Conv2D(32, (4,4), strides=(2,2), padding='same', activation='relu')
        self.drop4 = layers.Dropout(0.2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1,activation='sigmoid')

    def call(self, input):
        x = self.drop1(self.conv1(input))
        x = self.drop2(self.conv2(x))
        x = self.drop3(self.conv3(x))
        x = self.drop4(self.conv4(x))
        return x