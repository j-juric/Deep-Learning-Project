import tensorflow as tf
from tensorflow.keras import layers

class Cdann(layers.Layer):

    def __init__(self):
        super().__init__()
        
        self.d1 = layers.Dense(256,activation='relu')
        self.d2 = layers.Dense(64,activation='relu')
        self.d3 = layers.Dense(1,activation='relu')

    def call(self, shared_embedding):
        return self.d3(self.d2(self.d1(shared_embedding)))
