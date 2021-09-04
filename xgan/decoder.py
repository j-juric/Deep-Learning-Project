import tensorflow as tf
from tensorflow.keras import layers
from xgan.style_decoder import StyleDecoder
from xgan.shared_decoder import SharedDecoder

class Decoder(layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        self.style_A_decoder = StyleDecoder()
        self.style_B_decoder = StyleDecoder()

        self.shared_decoder = SharedDecoder()

    def call(self, input):
        shared_result = self.shared_decoder(input)
        img_A = self.style_A_decoder(shared_result)
        img_B = self.style_B_decoder(shared_result)
        return img_A, img_B