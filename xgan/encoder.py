import tensorflow as tf
from tensorflow.keras import layers
from xgan.style_encoder import StyleEncoder
from xgan.shared_encoder import SharedEncoder
from xgan.style_enum import Style

class Encoder(layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        self.style_A_encoder = StyleEncoder()
        self.style_B_encoder = StyleEncoder()

        self.shared_encoder = SharedEncoder()

    def call(self, input, style):
        if style == Style.A: 
            return self.shared_encoder(self.style_A_encoder(input))
        else:
            return self.shared_encoder(self.style_B_encoder(input))
