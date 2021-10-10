import tensorflow as tf
from tensorflow.keras import layers
#from keras.engine import Layer
from keras.engine.base_layer import Layer
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
# def reverse_gradient(X, hp_lambda):
#     '''Flips the sign of the incoming gradient during training.'''
#     try:
#         reverse_gradient.num_calls += 1
#     except AttributeError:
#         reverse_gradient.num_calls = 1

#     grad_name = "GradientReversal%d" % reverse_gradient.num_calls

#     @tf.RegisterGradient(grad_name)
#     def _flip_gradients(op, grad):
#         return [tf.negative(grad) * hp_lambda]

#     g = K.get_session().graph
#     with g.gradient_override_map({'Identity': grad_name}):
#         y = tf.identity(X)

#     return y

# class GradientReversal(Layer):
#     '''Flip the sign of gradient during training.'''
#     def __init__(self, hp_lambda, **kwargs):
#         super(GradientReversal, self).__init__(**kwargs)
#         self.supports_masking = False
#         self.hp_lambda = hp_lambda

#     def build(self, input_shape):
#         self._trainable_weights = []

#     def call(self, x, mask=None):
#         return reverse_gradient(x, self.hp_lambda)

#     def get_output_shape_for(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = {'hp_lambda': self.hp_lambda}
#         base_config = super(GradientReversal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def flip_gradient(x, l=1.0):
	positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
	negative_path = -x * tf.cast(l, tf.float32)
	return positive_path + negative_path

class Cdann(layers.Layer):

    def __init__(self):
        super().__init__()

        #self.flip = GradientReversal(0.3)

        
        self.d0 = layers.Dense(512, activation=LeakyReLU(0.3))
        self.d1 = layers.Dense(256,activation=LeakyReLU(0.3))
        self.drop1 = layers.Dropout(0.4)
        self.d2 = layers.Dense(128,activation=LeakyReLU(0.3))
        self.d3 = layers.Dense(64,activation=LeakyReLU(0.3))
        self.drop3 = layers.Dropout(0.4)
        self.d4 = layers.Dense(32,activation=LeakyReLU(0.3))
        self.d5 = layers.Dense(16,activation=LeakyReLU(0.3))
        self.drop5 = layers.Dropout(0.4)
        self.d6 = layers.Dense(1,activation='sigmoid')

    def call(self, shared_embedding):
        x = flip_gradient(shared_embedding, 1.0)
        x = self.d0(x)
        x = self.drop1(self.d1(x))
        x = self.drop3(self.d3(self.d2(x)))
        x = self.drop5(self.d5(self.d4(x)))

        return self.d6(x)
