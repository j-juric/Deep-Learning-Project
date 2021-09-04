import tensorflow as tf
from tensorflow.keras import layers

class Cdann(layers.Layer):

    def __init__(self):
        super().__init__()
        
        self.d0 = layers.Dense(512, activation='relu')
        self.drop0 = layers.Dropout(0.3)
        self.d1 = layers.Dense(256,activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.d2 = layers.Dense(64,activation='relu')
        self.drop2 = layers.Dropout(0.1)
        self.d3 = layers.Dense(1,activation='relu')

    def call(self, shared_embedding):
        x = self.drop0(self.d0(shared_embedding))
        x = self.drop1(self.d1(x))
        x = self.drop2(self.d2(x))
        return self.d3(x)
