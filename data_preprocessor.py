import tensorflow as tf
import numpy as np
from settings import *
from tqdm import tqdm


class DatasetPreprocessor():
    def __init__(self, domain_A_dataset_path, domain_B_dataset_path):

        #*******READ DATA*******
        self.domain_A_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            domain_A_dataset_path,
            image_size = IMAGE_SIZE,
            batch_size= BATCH_SIZE//2,
            label_mode=None
        )

        self.domain_B_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            domain_B_dataset_path,
            image_size = IMAGE_SIZE,
            batch_size= BATCH_SIZE//2,
            label_mode=None
        )

        #******SCALE DATA*******
        self.domain_A_dataset = self.domain_A_dataset.map(self.normalize)
        self.domain_B_dataset = self.domain_B_dataset.map(self.normalize)

    
    def normalize(self, image):
        image = tf.cast(image, tf.float32)
        image = (image-127.5) / 127.5
        return image

    def xgan_merge(self):

        if (len(self.domain_A_dataset) != len(self.domain_B_dataset)):
            raise "Datasets lengths not equal."

        iter_a = self.domain_A_dataset.as_numpy_iterator()
        iter_b = self.domain_B_dataset.as_numpy_iterator()

        xgan_dataset = None

        print('Dataset length:' , len(self.domain_B_dataset), sep=' ')

        for _ in tqdm(range(len(self.domain_A_dataset))): #range(len(self.domain_A_dataset)):
            data_a = iter_a.next()
            data_b = iter_b.next()
            data = np.array([np.concatenate((data_a, data_b))])
            if xgan_dataset is None:
                xgan_dataset = data
            else:
                xgan_dataset = np.vstack((xgan_dataset,data))

        np.save('./data.npy', arr=xgan_dataset)
            
        return np.array(xgan_dataset)
