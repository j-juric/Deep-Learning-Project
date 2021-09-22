#from cyclegan.cyclegan import CycleGAN
import tensorflow as tf
from data_preprocessor import DatasetPreprocessor

from xgan.xgan import XGAN
from xgan.style_enum import Style

#from dualgan.dualgan import DualGAN
import matplotlib.pyplot as plt
from settings import *
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')

dataset = DatasetPreprocessor(CELEBRITY_DATASET, CARTOON_DATASET).xgan_merge() if False else np.load('./data.npy')
xgan = XGAN()

import random
import string

def get_random_string():
    # choose from all lowercase letter
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(10))
    return result_str

model_name = get_random_string()
os.mkdir(f'./training_progression/gifs/xgan_{model_name}')
#***********************************************
#****************CALLBACK***********************
#***********************************************
import gc
import random 
class ResultGrid(tf.keras.callbacks.Callback):

    def __init__(self, dataset):
        super().__init__()
        #self.digit = random.randint(0,600)
        r = random.randint(0,len(dataset)-1)
        self.images = np.array(dataset[r,0:16])
        
    
    def on_epoch_end(self, epoch, logs=None):
        
        #images = dataset[0,0:16]
        if epoch%10 !=0:
            return
        result = xgan.generator(self.images, Style.A, training=False)
        # input_domain = result['img_A']
        target_domain = np.array(result['img_B'])

        fig = plt.figure(figsize=(32,32))      

        for i in range(16):
            fig.add_subplot(4,8,i+1)
            img = (self.images[i]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        for i in range(16,32):
            fig.add_subplot(4,8,i+1)
            img = (target_domain[i-16]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig(f'./training_progression/gifs/xgan_{model_name}/image_{epoch}.png')
        plt.clf()
        plt.close(fig)
        gc.collect()



#***********************************************
#******************MAIN*************************
#***********************************************
def main():
    
    print('Compiling model...')
    xgan.compile(
        generator_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=BETA[0], beta_2=BETA[1]),
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=BETA[0], beta_2=BETA[1]),
        cdann_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=BETA[0], beta_2=BETA[1]),
        loss_function= tf.keras.losses.BinaryCrossentropy(from_logits= False)
    )
    tf.compat.v1.get_default_graph().finalize()
    print('Training model...')
    xgan.fit(dataset, epochs=500, callbacks=[ResultGrid(dataset)])

    # dualgan_model_name = get_random_string()
    # dualgan = DualGAN(dualgan_model_name)
    # dualgan.train(dataset, epochs=100)

    # from discogan.discogan import DiscoGAN
    # cyclegan_model_name = get_random_string()
    # cyclegan = DualGAN(cyclegan_model_name)
    # cyclegan.train(dataset, epochs=100)

if __name__ == '__main__':
    main()