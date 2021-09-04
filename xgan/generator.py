import tensorflow as tf
from tensorflow.keras import layers
from xgan.encoder import Encoder
from xgan.decoder import Decoder


#************************************************************************************************************************
#********************************************************GENERATOR*******************************************************
#************************************************************************************************************************
#                       ENCODER                                                             DECODER
#     _________________________________________________               ________________________________________________
#    |                                                |              |                                                |              
# -->| Style A Encoder ---------|                     |    shared    |                     |-------- Style A Decoder  |-->
#    |                          |                     |  embedding   |                     |                          | 
#    |                          |----> Shared Encoder |>>>>>>>>>>>>>>|  Shared Decoder --> |                          |
#    |                          |                     |              |                     |                          | 
#    |                          |                     |              |                     |                          | 
# -->| Style B Encoder ---------|                     |              |                     |-------- Style B Decoder  |-->
#    |________________________________________________|              |________________________________________________|
#
#************************************************************************************************************************
#************************************************************************************************************************

class Generator(layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, input, style):
        shared_emedding = self.encoder(input,style)
        img_A, img_B = self.decoder(shared_emedding)
        return {'img_A': img_A,'img_B':img_B, 'shared_embedding': shared_emedding}