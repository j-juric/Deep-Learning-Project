import tensorflow as tf
from tensorflow.keras import layers

class SharedEncoder(layers.Layer):
    def __init__(self):
        super(SharedEncoder, self).__init__()

        self.conv3 = layers.Conv2D(128, (4,4), strides=(2,2), padding='same', activation='relu')
        #self.bnorm3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(256, (4,4), strides=(2,2), padding='same', activation='relu')
        #self.bnorm4 = layers.BatchNormalization()
        
        self.flatten1 = layers.Flatten()

        self.fc1 = layers.Dense(1024,activation='relu')
        self.fc2 = layers.Dense(1024,activation='relu')

    def call(self, input): 
        x = self.conv3(input)
        x = self.conv4(x)
        x = self.flatten1(x)
        x = self.fc2(self.fc1(x))
        return x