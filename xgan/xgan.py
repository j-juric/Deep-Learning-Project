import tensorflow as tf
from tensorflow.keras import layers
from settings import *
from xgan.generator import Generator
from xgan.discriminator import Discriminator
from xgan.domain_adverserial_classifier import Cdann
from xgan.style_enum import Style

import gc

class XGAN(tf.keras.Model):
    
    def __init__(self):
        super(XGAN,self).__init__()

        #************MODEL************

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.cdann = Cdann()

        #***********METRICS***********

        self.g_loss_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.d_loss_metric = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
        self.sem_con_loss_metric = tf.keras.metrics.Mean('semantic_consistency_loss', dtype=tf.float32)
        self.rec_loss_metric = tf.keras.metrics.Mean('reconstruction_loss', dtype=tf.float32)
        self.dom_adv_loss_metric = tf.keras.metrics.Mean('domain_adverserial_loss', dtype=tf.float32)
        self.obj_loss_metric = tf.keras.metrics.Mean('objective_loss', dtype=tf.float32)

        self.train_summary_writer = tf.summary.create_file_writer('./train_summary')

        self.batch_step=0


    def compile(self, generator_optimizer, discriminator_optimizer, cdann_optimizer, loss_function):
        super(XGAN,self).compile()

        self.generator_optimizer = generator_optimizer #tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1])
        self.discriminator_optimizer = discriminator_optimizer #tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1])
        self.cdann_optimizer = cdann_optimizer
        self.loss = loss_function #tf.keras.losses.BinaryCrossentropy(from_logits= True)
        
        
    #***************************************
    #******LOSS FUNCTION CALCULATIONS*******
    #***************************************

    def semantic_consistency_loss(self, output_A, output_B):

        x = self.generator( output_A['img_B'], style=Style.B)
        x = tf.reduce_mean(tf.abs(output_A['shared_embedding'] - x['shared_embedding']))

        y = self.generator( output_B['img_A'], style= Style.A)
        y = tf.reduce_mean(tf.abs(output_B['shared_embedding'] - y['shared_embedding']))
        return x+y

    @tf.function
    def reconstruction_loss(self, input_A, input_B, output_A_to_A, output_B_to_B): #auto encoder loss

        rec_loss_A = tf.reduce_mean(tf.losses.mean_squared_error(input_A, output_A_to_A))
        rec_loss_B = tf.reduce_mean(tf.losses.mean_squared_error(input_B, output_B_to_B))
        return rec_loss_A + rec_loss_B

    def domain_adversarial_loss(self, output_A, output_B):

        cdann_A = self.cdann(output_A['shared_embedding'])
        cdann_B = self.cdann(output_B['shared_embedding'])

        r_val = tf.random.uniform(shape=[])

        loss_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cdann_B, labels=tf.ones_like(cdann_B))) #self.loss(tf.zeros_like(cdann_A), cdann_A)
        loss_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cdann_A, labels=tf.zeros_like(cdann_A))) #self.loss(tf.ones_like(cdann_B), cdann_B) #

        return r_val * loss_A + (1-r_val)* loss_B


    def objective_loss(self, input_B):

        #fake_output = self.discriminator(self.generator(input_A, Style.A)['img_B'])
        real_output = self.discriminator(input_B)

        real_loss = self.loss(tf.ones_like(real_output), real_output)
        #fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)

        return real_loss 


    def generator_loss(self, input_A, input_B, output_A, output_B):

        #***CALCULATE LOSS FUNCTIONS***
        L_sem = self.semantic_consistency_loss(output_A, output_B)
        L_rec = self.reconstruction_loss(input_A, input_B, output_A['img_A'], output_B['img_B'])
        L_dann = self.domain_adversarial_loss(output_A, output_B)
        L_gan = self.objective_loss(input_B) #if d_loss<1.0 else 0.0
        
        #********SAVE METRICS**********
        self.sem_con_loss_metric(L_sem)
        self.rec_loss_metric(L_rec)
        self.dom_adv_loss_metric(L_dann)
        self.obj_loss_metric(L_gan)

        #***FINAL LOSS CALCULATION*****
        w_d, w_s, w_g = 1.0, 0.5 , 1.0

        L_xgan = L_rec  + (w_s*L_sem) + (w_g*L_gan) + (w_d * L_dann)

        return L_xgan, L_dann

    def discriminator_loss(self, output_A_to_B, input_B):

        real_output = self.discriminator(input_B)
        fake_output = self.discriminator(output_A_to_B)

        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        
        return 0.5* (real_loss + fake_loss)

    #***************************************
    #*******MODEL TRAINING FUNCTIONS********
    #***************************************    

    def on_epoch_begin(self,logs=None):
        self.batch_step=0

    def on_batch_step(self,logs=None):
        self.batch_step+=1


    # @tf.custom_gradient
    # def reverse_gradient(self, x):
    #     y = tf.identity(x)
    #     def custom_grad(dy):
    #         return -dy
    #     return y, custom_grad

    @tf.function
    def train_step(self,images):

        domain_A_images = images[:, 0:BATCH_SIZE//2]
        domain_B_images = images[:, BATCH_SIZE//2:]

        domain_A_images = domain_A_images[0]
        domain_B_images = domain_B_images[0]

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as cdann_tape:

            result_from_A = self.generator(domain_A_images, style=Style.A)
            result_from_B = self.generator(domain_B_images, style=Style.B)

            g_loss, cdann_loss = self.generator_loss(domain_A_images, domain_B_images, result_from_A, result_from_B)
            d_loss = self.discriminator_loss(result_from_A['img_B'], domain_B_images)
        
            if g_loss > 0.2:
                g_gradient = generator_tape.gradient(g_loss, self.generator.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
            
            if d_loss > 0.35:
                d_gradient = discriminator_tape.gradient(d_loss, self.discriminator.trainable_variables)
                self.discriminator_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
            
            cdann_gradient = cdann_tape.gradient(cdann_loss, self.cdann.trainable_variables)
            self.cdann_optimizer.apply_gradients(zip(cdann_gradient, self.cdann.trainable_variables))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('generator_loss', self.g_loss_metric.result(), step=self.batch_step)
            tf.summary.scalar('discriminator_loss', self.d_loss_metric.result(), step=self.batch_step)
            tf.summary.scalar('semantic_consistency_loss', self.sem_con_loss_metric.result(), step=self.batch_step)
            tf.summary.scalar('reconstruction_loss', self.rec_loss_metric.result(), step=self.batch_step)
            tf.summary.scalar('domain_adverserial_loss', self.dom_adv_loss_metric.result(), step=self.batch_step)
            tf.summary.scalar('objective_loss', self.obj_loss_metric.result(), step=self.batch_step)
        
        result = {
            "Generator loss": g_loss,
            "Discriminator loss": d_loss,
            "Semantic consistency loss": self.sem_con_loss_metric.result(),
            "Reconstruction loss": self.rec_loss_metric.result(),
            "Domain adverserial loss": self.dom_adv_loss_metric.result(),
            "Objective loss": self.obj_loss_metric.result(),
        }

        gc.collect()
        #tf.keras.backend.clear_session()

        return result
  