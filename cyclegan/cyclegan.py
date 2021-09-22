import tensorflow as tf
from tensorflow import keras

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
from tensorflow.python.keras.layers import normalization
from tqdm import tqdm

class CycleGAN():
    def __init__(self,model_name = None) -> None:

        self.model_name = model_name

        self.domain_A_data = None
        
        self.optimizewr= Adam(2e-4,0.5)

        #-------------------------
        #-----Discriminators------
        #-------------------------

        self.D_A = self.build_discriminator()
        self.D_B = self.build_discriminator()

        self.D_A.trainable = False
        self.D_B.trainable = False

        #-------------------------
        #-------Generators--------
        #-------------------------

        self.G_AB = self.build_generator()
        self.G_BA = self.build_generator()

        #-------------------------
        #-----Combined Model------
        #-------------------------

        imgs_A = Input(shape=(64,64,3))
        imgs_B = Input(shape=(64,64,3))

        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        id_A = self.G_BA(imgs_A)
        id_B = self.G_AB(imgs_B)


        self.combined_model = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B, id_A, id_B])

        #-------------------------
        #----Model compilation----
        #-------------------------

        optimizer = Adam(2e-4, 0.5)

        self.D_A.compile(loss = 'mse', optimizer=optimizer, metrics=['accuracy'])
        self.D_B.compile(loss = 'mse', optimizer=optimizer, metrics=['accuracy'])

        self.combined_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], optimizer=optimizer, loss_weights=[1.0, 1.0, 10.0, 10.0, 1.0, 1.0])
    

    def build_discriminator(self)->Model:

        def d_layer(input_layer, filters, kernel_size=4, padding='same', activation = LeakyReLU(alpha=0.2), normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding=padding, activation = activation)(input_layer)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        input = Input(shape=(64,64,3))

        d1 = d_layer(input, 64, normalization=False)
        d2 = d_layer(d1, 128)
        d3 = d_layer(d2, 256)
        d4 = d_layer(d3, 512)

        output = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(input, output)


    def build_generator(self)->Model:

        def conv2d(input_layer, filters, kernel_size = 4, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)):
            c = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, activation= activation)(input_layer)
            c = InstanceNormalization()(c)
            return c

        def deconv2d(input_layer, skip_layer, filters, kernel_size = 4, strides=1, padding='same', activation='relu'):
            d = UpSampling2D(size=2)(input_layer)
            d = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(d)
            d = InstanceNormalization()(d)
            d = Concatenate()([d, skip_layer])
            return d

        input = Input(shape=(64,64,3))

        c1 = conv2d(input, 32)
        c2 = conv2d(c1, 64)
        c3 = conv2d(c2, 128)
        c4 = conv2d(c3, 256)

        d1 = deconv2d(c4, c3, 128)
        d2 = deconv2d(d1, c2, 64)
        d3 = deconv2d(d2, c1, 32)

        d4 = UpSampling2D(size=2)(d3)
        output = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(d4)

        return Model(input, output)


    def train(self, data, epochs=100, batch_size=32):

        data_A = data[:,0:batch_size//2]
        data_B = data[:,batch_size//2:]

        data_A = np.reshape(data_A,(data_A.shape[0]* data_A.shape[1], 1, data_A.shape[2], data_A.shape[3], data_A.shape[4]))
        data_B = np.reshape(data_B,(data_B.shape[0]* data_B.shape[1], 1, data_B.shape[2], data_B.shape[3], data_B.shape[4]))

        num_of_batches = len(data)

        print('Dataset data_A:',data_A.shape, sep=' ')

        self.domain_A_data = data_A

        valid = np.ones((batch_size//2,)+(4,4,1))
        fake = np.zeros((batch_size//2,)+(4,4,1))

        half_batch = batch_size//2 

        if(self.model_name!=None):
            os.mkdir(f'./training_progression/gifs/cyclegan_{self.model_name}')
       
        for epoch in range(epochs):

            G_loss, D_loss, D_A_loss, D_B_loss = 0, 0, 0, 0

            imgs_A, imgs_B = None, None

            for i in tqdm(range(1000)):

                imgs_A = data_A[i]
                imgs_B = data_B[i]

                fake_B = self.G_AB.predict(imgs_A)
                fake_A = self.G_BA.predict(imgs_B)

                #------------------------------
                #----Discriminator training----
                #------------------------------

                loss_A_real = self.D_A.train_on_batch(imgs_A, valid)
                loss_A_fake = self.D_A.train_on_batch(fake_A, fake)

                loss_B_real = self.D_B.train_on_batch(imgs_B, valid)
                loss_B_fake = self.D_B.train_on_batch(fake_B, fake)

                D_A_loss = 0.5 * np.add(loss_A_fake, loss_A_real)
                D_B_loss = 0.5 * np.add(loss_B_fake, loss_B_real)

                D_loss = 0.5 * np.add(D_A_loss, D_B_loss)

                #------------------------------
                #------Generator training------
                #------------------------------
                G_loss = self.combined_model.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B,  imgs_A, imgs_B])

            print(f'Epoch {epoch+1}: G loss: {G_loss}, D loss: {D_loss}, D_A loss: {D_A_loss}, D_B loss: {D_B_loss}')
            if self.model_name != None:
                self.save_images(epoch)

    
    def save_images(self, epoch):
        
        images = self.domain_A_data[0:16]
        
        result = [self.G_AB.predict(x) for x in images]

        fig = plt.figure(figsize=(32,32))

        for i in range(16):
            fig.add_subplot(4,8,i+1)
            img = (images[i][0]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        for i in range(16,32):
            fig.add_subplot(4,8,i+1)
            img = (result[i-16][0]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig(f'./training_progression/gifs/cyclegan_{self.model_name}/image_{epoch}.png')
