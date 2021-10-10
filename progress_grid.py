import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import gc
import random
import string

def get_random_string():
    # choose from all lowercase letter
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(10))
    return result_str


class ProgressGrid(tf.keras.callbacks.Callback):

    def __init__(self, dataset, model, architecture, num_of_batches):
        #self.digit = random.randint(0,600)
        r = random.randint(0,len(dataset)-1)
        self.images = np.array(dataset[5,0:16]) #10
        self.model = model
        self.architecture = architecture
        self.model_name = get_random_string()
        self.num_of_batches = num_of_batches
        os.mkdir(f'./training_progression/gifs/{self.architecture}_{self.num_of_batches}_{self.model_name}')
    
    def on_epoch_end(self, epoch, logs=None):
        
        #images = dataset[0,0:16]
        if epoch%10 !=0:
            return
        target_domain = self.model.make_prediction(self.images)
        target_domain = np.array(target_domain)

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

        plt.savefig(f'./training_progression/gifs/{self.architecture}_{self.num_of_batches}_{self.model_name}/image_{epoch}.png')
        plt.clf()
        plt.close(fig)
        gc.collect()
