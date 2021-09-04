import tensorflow as tf
from data_preprocessor import DatasetPreprocessor
from xgan.xgan import XGAN
from xgan.style_enum import Style
import matplotlib.pyplot as plt
from settings import *
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')

dataset = DatasetPreprocessor(CELEBRITY_DATASET, CARTOON_DATASET).xgan_merge() if True else np.load('./data.npy')
xgan = XGAN(batch_size = BATCH_SIZE)

import random
import string

def get_random_string():
    # choose from all lowercase letter
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(10))
    return result_str

model_name = get_random_string()
os.mkdir(f'./training_progression/gifs/{model_name}')
#***********************************************
#****************CALLBACK***********************
#***********************************************

class GifProducer(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        images = dataset[16:32]
        result = xgan.generator(images, Style.A, training=False)
        images = np.array(images)
        # input_domain = result['img_A']
        target_domain = np.array(result['img_B'])

        fig = plt.figure(figsize=(6,12))

        for i in range(8):
            fig.add_subplot(8,4,i+1)
            img = (images[i]+0.5)* 255.0
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        for i in range(8,16):
            fig.add_subplot(8,4,i+1)
            img = (target_domain[i-8]+0.5)* 255.0
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig(f'./training_progression/gifs/{model_name}/image_{epoch}.png')


model_cp = tf.keras.callbacks.ModelCheckpoint(f'./model/{model_name}.h5', verbose=1, save_best_only=True)

#***********************************************
#******************MAIN*************************
#***********************************************
def main():
    
    xgan.compile(
        generator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
        cdann_optimizer=tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
        loss_function= tf.keras.losses.BinaryCrossentropy(from_logits= True)
    )

    xgan.fit(dataset, epochs=5, callbacks=[GifProducer(),model_cp])

if __name__ == '__main__':
    main()