import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import gc
from tqdm import tqdm

import shutil



def log_result(data, model, architecture, start_index, end_index, is_test_set = True):

    print('Logging results...')
    data_set = "test" if is_test_set else "train"
    shutil.rmtree(f'./results/{architecture}/{data_set}', ignore_errors=True)
    os.mkdir(f'./results/{architecture}/{data_set}')
    for k in tqdm(range(start_index,end_index)):

        images = data[k,0:16]

        target_domain = model.make_prediction(images)
        target_domain = np.array(target_domain)

        fig = plt.figure(figsize=(32,32))

        for i in range(16):
                fig.add_subplot(4,8,i+1)
                img = (images[i]+1.0)* 127.5
                img = img.astype(np.uint8)
                plt.imshow(img)
                plt.axis('off')

        for i in range(16,32):
            fig.add_subplot(4,8,i+1)
            img = (target_domain[i-16]+1.0)* 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        plt.savefig(f'./results/{architecture}/{data_set}/{k}.png')
        plt.clf()
        plt.close(fig)
        gc.collect()
    