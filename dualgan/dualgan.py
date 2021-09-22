import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

class DualGAN(tf.keras.Model):
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

        imgs_A = Input(shape=(64*64*3))
        imgs_B = Input(shape=(64*64*3))

        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        self.combined_model = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])

        #-------------------------
        #----Model compilation----
        #-------------------------

        optimizer = Adam(2e-4, 0.5)

        self.D_A.compile(loss = self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.D_B.compile(loss = self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        self.combined_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'], optimizer=optimizer, loss_weights=[1,1,50,50])
    

    def build_discriminator(self)->Model:
        input = Input(shape=(64*64*3,))

        model = Sequential()

        model.add(Dense(512, activation=LeakyReLU(0.2)))
        model.add(Dense(256, activation=LeakyReLU(0.2)))
        model.add(BatchNormalization())
        model.add(Dense(1))

        return Model(input, model(input))


    def build_generator(self)->Model:
        model = Sequential()

        input = Input(shape=(64*64*3,))

        model.add(Dense(256, activation=LeakyReLU(0.2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(512, activation=LeakyReLU(0.2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(1024, activation=LeakyReLU(0.2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(64*64*3, activation='tanh'))

        return Model(input, model(input))


    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true*y_pred)


    def generate_random_sample(self, data, sample_size):
        x = np.random.randint(0, data.shape[0], sample_size)
        return data[x]


    def clip_weights(self, model:Model):
        clip_val = 1e-2
        for layer in model.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -clip_val, clip_val) for w in weights]
            layer.set_weights(weights)


    def train(self, data, epochs=100, batch_size=32):

        print('*'*75)
        print('Dataset shape:',data.shape, sep=' ')
        data = np.reshape(data, (len(data),len(data[0]), 64*64*3))
        print('Dataset reshape:',data.shape, sep=' ')
        print('*'*75)

        data_A = data[:,0:16]
        data_B = data[:,16:]
        print('Dataset data_A:',data_A.shape, sep=' ')

        data_A = np.reshape(data_A, (data_A.shape[0]*data_A.shape[1], data_A.shape[2]))
        data_B = np.reshape(data_B, (data_B.shape[0]*data_B.shape[1], data_B.shape[2]))

        print('Dataset data_A:',data_A.shape, sep=' ')
        self.domain_A_data = data_A

        valid = -np.ones((batch_size//2))
        fake = np.ones((batch_size//2))

        n_critic = 4

        half_batch = batch_size//2 

        if(self.model_name!=None):
            os.mkdir(f'./training_progression/gifs/dualgan_{self.model_name}')
       
        for epoch in range(epochs):

            for i in range(100):

                D_A_loss, D_B_loss = 0, 0

                # Discriminator training

                imgs_A, imgs_B = None, None


                for _ in range(n_critic):

                    imgs_A = self.generate_random_sample(data_A, half_batch)
                    imgs_B = self.generate_random_sample(data_B, half_batch)

                    fake_B = self.G_AB.predict(imgs_A)
                    fake_A = self.G_BA.predict(imgs_B)

                    tf.keras.backend.clear_session()
                    loss_A_real = self.D_A.train_on_batch(imgs_A, valid)
                    loss_A_fake = self.D_A.train_on_batch(fake_A, fake)
                    self.clip_weights(self.D_A)

                    tf.keras.backend.clear_session()
                    loss_B_real = self.D_B.train_on_batch(imgs_B, valid)
                    loss_B_fake = self.D_B.train_on_batch(fake_B, fake)
                    self.clip_weights(self.D_B)

                    D_A_loss = 0.5 * np.add(loss_A_fake, loss_A_real)
                    D_B_loss = 0.5 * np.add(loss_B_fake, loss_B_real)

                # Generator training
                tf.keras.backend.clear_session()
                G_loss = self.combined_model.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

                print(f'Epoch {epoch+1}: G loss: {G_loss}, D_A loss: {D_A_loss}, D_B loss: {D_B_loss}')
                if self.model_name != None:
                    self.save_images(epoch)

    
    def save_images(self, epoch):
        
        images = self.domain_A_data[0:16]

        
        result = self.G_AB.predict(images)
        images = np.reshape(images, (16,64,64,3))
        result = np.reshape(result, (16,64,64,3))

        fig = plt.figure(figsize=(32,32))

        for i in range(16):
            fig.add_subplot(4,8,i+1)
            img = (images[i]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        for i in range(16,32):
            fig.add_subplot(4,8,i+1)
            img = (result[i-16]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig(f'./training_progression/gifs/dualgan_{self.model_name}/image_{epoch}.png')
