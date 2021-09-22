import tensorflow as tf
from tensorflow.keras import layers

class SharedEncoder(layers.Layer):
    def __init__(self):
        super(SharedEncoder, self).__init__()

        self.conv3 = layers.Conv2D(128, (4,4), strides=(2,2), padding='same', activation='relu')
        self.bnorm3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(256, (4,4), strides=(2,2), padding='same', activation='relu')
        self.bnorm4 = layers.BatchNormalization()
        
        self.flatten1 = layers.Flatten()

        self.fc1 = layers.Dense(1024,activation='relu')
        self.bnorm5 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.5)

        self.fc2 = layers.Dense(1024,activation='relu')
        self.bnorm6 = layers.BatchNormalization()

    def call(self, input): 
        x = self.bnorm3(self.conv3(input))
        x = self.bnorm4(self.conv4(x))
        x = self.flatten1(x)
        x = self.drop1(self.bnorm5(self.fc1(x)))
        x = self.bnorm6(self.fc2(x))
        return x