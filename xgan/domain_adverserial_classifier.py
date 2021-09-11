import tensorflow as tf
from tensorflow.keras import layers

class Cdann(layers.Layer):

    def __init__(self):
        super().__init__()
        
        self.d0 = layers.Dense(512, activation='relu')
        self.d1 = layers.Dense(256,activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.d2 = layers.Dense(128,activation='relu')
        self.d3 = layers.Dense(64,activation='relu')
        self.drop3 = layers.Dropout(0.3)
        self.d4 = layers.Dense(32,activation='relu')
        self.d5 = layers.Dense(16,activation='relu')
        self.drop5 = layers.Dropout(0.3)
        self.d6 = layers.Dense(1,activation='sigmoid')

    def call(self, shared_embedding):
        x = self.d0(shared_embedding)
        x = self.drop1(self.d1(x))
        x = self.drop3(self.d3(self.d2(x)))
        x = self.drop5(self.d5(self.d4(x)))
        return self.d6(x)
