
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers.advanced_activations import LeakyReLU
from settings import *
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


class Generator(layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        ##################################
        ###########Congestion#############
        ##################################

        self.conv1 = layers.Conv2D(64,kernel_size=4, strides=2, padding='same', use_bias=False, activation=LeakyReLU(0.3))

        self.conv2 = layers.Conv2D(128,kernel_size=4, strides=2, padding='same', use_bias=False)
        self.norm2 = InstanceNormalization()
        self.lerelu2 = LeakyReLU(0.3)


        self.conv3 = layers.Conv2D(256,kernel_size=4, strides=2, padding='same', use_bias=False)
        self.norm3 = InstanceNormalization()
        self.lerelu3 = LeakyReLU(0.3)


        self.conv4 = layers.Conv2D(512,kernel_size=4, strides=2, padding='same', use_bias=False)
        self.norm4 = InstanceNormalization()
        self.lerelu4 = LeakyReLU(0.3)

        self.conv5 = layers.Conv2D(100,kernel_size=4, strides=2, padding='valid', use_bias=False)
        self.norm5 = InstanceNormalization()
        self.lerelu5 = LeakyReLU(0.3)

        ##################################
        ############Expansion#############
        ##################################

        self.deconv1 = layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', use_bias=False)
        self._norm1 = InstanceNormalization()
        self._lerelu1 = LeakyReLU(0.3)

        self.deconv2 = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)
        self._norm2 = InstanceNormalization()
        self._lerelu2 = LeakyReLU(0.3)

        self.deconv3 = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)
        self._norm3 = InstanceNormalization()
        self._lerelu3 = LeakyReLU(0.3)

        self.deconv4 = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)
        self._norm4 = InstanceNormalization()
        self._lerelu4 = LeakyReLU(0.3)

        self.deconv5 = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')


    def call(self, x):

        x = self.conv1(x)

        x= self.lerelu2(self.norm2(self.conv2(x)))
        x= self.lerelu3(self.norm3(self.conv3(x)))
        x= self.lerelu4(self.norm4(self.conv4(x)))
        x= self.lerelu5(self.norm5(self.conv5(x)))

        x= self._lerelu1(self._norm1(self.deconv1(x)))
        x= self._lerelu2(self._norm2(self.deconv2(x)))
        x= self._lerelu3(self._norm3(self.deconv3(x)))
        x= self._lerelu4(self._norm4(self.deconv4(x)))

        x= self.deconv5(x)

        return x


class Discriminator(layers.Layer):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = layers.Conv2D(64,kernel_size=4, strides=2, padding='same', use_bias=False, activation=LeakyReLU(0.2))

        self.conv2 = layers.Conv2D(128,kernel_size=4, strides=2, padding='same', use_bias=False)
        self.norm2 = InstanceNormalization()
        self.lerelu2 = LeakyReLU(0.2)


        self.conv3 = layers.Conv2D(256,kernel_size=4, strides=2, padding='same', use_bias=False)
        self.norm3 = InstanceNormalization()
        self.lerelu3 = LeakyReLU(0.2)


        self.conv4 = layers.Conv2D(512,kernel_size=4, strides=2, padding='same', use_bias=False)
        self.norm4 = InstanceNormalization()
        self.lerelu4 = LeakyReLU(0.2)

        self.conv5 = layers.Conv2D(1,kernel_size=4, strides=2, padding='same', use_bias=False, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)

        x= self.lerelu2(self.norm2(self.conv2(x)))
        x= self.lerelu3(self.norm3(self.conv3(x)))
        x= self.lerelu4(self.norm4(self.conv4(x)))
        x= self.conv5(x)

        return x


class DiscoGAN(tf.keras.Model):

    def __init__(self):
        super(DiscoGAN,self).__init__()

        #-----------MODEL------------

        self.G_AB = Generator()
        self.G_BA = Generator()

        self.DA = Discriminator()
        self.DB = Discriminator()

        #-----------METRICS----------

        self.batch_step=0
        self.epoch_step=0

        self.g_ab_loss_metric = tf.keras.metrics.Mean('generator_ab_loss', dtype=tf.float32)
        self.d_a_loss_metric = tf.keras.metrics.Mean('discriminator_a_loss', dtype=tf.float32)
        self.g_ba_loss_metric = tf.keras.metrics.Mean('generator_ba_loss', dtype=tf.float32)
        self.d_b_loss_metric = tf.keras.metrics.Mean('discriminator_b_loss', dtype=tf.float32)


    def compile(self, generator_optimizer, discriminator_optimizer, loss_function):
        super(DiscoGAN,self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss = loss_function


    #***************************************
    #******LOSS FUNCTION CALCULATIONS*******
    #***************************************

    def generator_loss(self, real, fake, reconstruction, is_A_to_B=True):

        l_const = tf.reduce_mean(tf.losses.mean_squared_error(real, reconstruction)) 
        
        discriminator_output = self.DA(fake) if is_A_to_B else self.DB(fake)

        #l_gan = tf.reduce_sum(tf.square(discriminator_output-1))/2
        l_gan = self.wasserstein_loss(discriminator_output, tf.ones_like(discriminator_output))
        
        return 2*l_gan + 5*l_const

    def discriminator_loss(self, real, fake, is_A=True):

        real_loss = self.DA(real) if is_A else self.DB(real)
        fake_loss = self.DA(fake) if is_A else self.DB(fake)

        #result = self.loss(tf.ones_like(real_loss), real_loss)
        #result += self.loss(tf.zeros_like(fake_loss), fake_loss)

        real_valid = tf.random.uniform([1,BATCH_SIZE//2], 0.7, 1.0)
        fake_valid = tf.random.uniform([1,BATCH_SIZE//2], 0.0, 0.3)

        result = self.wasserstein_loss(real_valid, real_loss)
        result += self.wasserstein_loss(fake_valid, fake_loss)

        return result

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true*y_pred)


    #***************************************
    #*******MODEL TRAINING FUNCTIONS********
    #***************************************
    # 
    def on_training_begin(self, logs=None):
        self.epoch_step=0 

    def on_epoch_begin(self,logs=None):
        self.batch_step=0
        self.epoch_step+=1

    def on_batch_step(self,logs=None):
        self.batch_step+=1

    
    @tf.function
    def train_step(self,images):

        #----------Prepare data-------------

        domain_A_images = images[:, 0:BATCH_SIZE//2]
        domain_B_images = images[:, BATCH_SIZE//2:]

        domain_A_images = domain_A_images[0]
        domain_B_images = domain_B_images[0]


        with tf.GradientTape() as g_ab_tape, tf.GradientTape() as d_a_tape, tf.GradientTape() as g_ba_tape, tf.GradientTape() as d_b_tape:
            #---------Generate images-----------

            fake_A = self.G_BA(domain_B_images)
            fake_B = self.G_AB(domain_A_images)

            rec_A = self.G_BA(fake_B)
            rec_B = self.G_AB(fake_A)

            #----------Calculate loss-----------

            g_ab_loss = self.generator_loss(domain_A_images, fake_A, rec_A, True)
            g_ba_loss = self.generator_loss(domain_B_images, fake_B, rec_B, False)

            d_a_loss = self.discriminator_loss(domain_A_images, fake_A, True)
            d_b_loss = self.discriminator_loss(domain_B_images, fake_B, False)

            g_loss = g_ab_loss + g_ba_loss
            d_loss = d_a_loss + d_b_loss

            #---------Update weights------------

            g_a_gradient = g_ab_tape.gradient(g_loss, self.G_AB.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(g_a_gradient, self.G_AB.trainable_variables))

            g_b_gradient = g_ba_tape.gradient(g_loss, self.G_BA.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(g_b_gradient, self.G_BA.trainable_variables))

            if d_loss > 0.5:
                d_a_gradient = d_a_tape.gradient(d_loss, self.DA.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(d_a_gradient, self.DA.trainable_variables))

                d_b_gradient = d_b_tape.gradient(d_loss, self.DB.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(d_b_gradient, self.DB.trainable_variables))

        #-----------Training output---------

        # with self.train_summary_writer.as_default():
        #     tf.summary.scalar('generator_loss', self.g_ab_loss_metric.result(), step=self.batch_step)
        #     tf.summary.scalar('discriminator_loss', self.d_a_loss_metric.result(), step=self.batch_step)
        #     tf.summary.scalar('generator_loss', self.g_ba_loss_metric.result(), step=self.batch_step)
        #     tf.summary.scalar('discriminator_loss', self.d_b_loss_metric.result(), step=self.batch_step)


        result = {
            "Generator AB loss": g_ab_loss,
            "Discriminator A loss": d_a_loss,
            "Generator BA loss": g_ba_loss,
            "Discriminator B loss": d_b_loss,
        }

        return result

    def make_prediction(self, data):
        return self.G_AB(data, training=False)

