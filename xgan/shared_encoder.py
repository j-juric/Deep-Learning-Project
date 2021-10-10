import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class SharedEncoder(layers.Layer):
    def __init__(self):
        super(SharedEncoder, self).__init__()

        self.conv3 = layers.Conv2D(128, (4,4), strides=(2,2), padding='same')
        self.bnorm3 = InstanceNormalization()
        self.lrelu1 = LeakyReLU(0.3)

        self.conv4 = layers.Conv2D(256, (4,4), strides=(2,2), padding='same')
        self.bnorm4 = InstanceNormalization()
        self.lrelu2 = LeakyReLU(0.3)
        
        self.flatten1 = layers.Flatten()

        self.fc1 = layers.Dense(1024)
        self.bnorm5 = InstanceNormalization()
        self.lrelu3 = LeakyReLU(0.3)
        self.drop1 = layers.Dropout(0.4)

        self.fc2 = layers.Dense(1024)
        self.bnorm6 = InstanceNormalization()
        self.lrelu4 = LeakyReLU(0.3)

    def call(self, input): 
        x = self.lrelu1(self.bnorm3(self.conv3(input)))
        x = self.lrelu2(self.bnorm4(self.conv4(x)))
        x = self.flatten1(x)
        x = self.lrelu3(self.drop1(self.bnorm5(self.fc1(x))))
        x = self.lrelu4(self.bnorm6(self.fc2(x)))
        return x