import tensorflow as tf
from data_preprocessor import DatasetPreprocessor
from discogan.discogan2 import DiscoGAN
from xgan.xgan import XGAN
from settings import *
import os
import numpy as np
from progress_grid import ProgressGrid
from result_output import log_result

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')

#***********************************************
#******************MAIN*************************
#***********************************************
def main():
    
    print("Loading data")
    dataset = np.load('./data.npy') if PRELOADED_DATA==True else DatasetPreprocessor(CELEBRITY_DATASET, CARTOON_DATASET).prepare_dataset()
    num_of_batches = 300

    #******************XGAN*************************
    if arch == 'xgan':
        
        xgan = XGAN()
        if mode == "train":
            print('Compiling XGAN model...')
            xgan.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=X_G_LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=X_D_LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
                cdann_optimizer=tf.keras.optimizers.Adam(learning_rate=X_C_LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
                loss_function= tf.keras.losses.BinaryCrossentropy(from_logits= False)
            )
            print('Training XGAN model...')
            xgan.fit(dataset[:num_of_batches], epochs=X_EPOCHS, callbacks=[ProgressGrid(dataset, xgan, 'xgan', num_of_batches)])
            xgan.save_weights('./model/my_checkpoint/xgan_001')
        else:
            print('Loading XGAN model...')
            xgan.load_weights('./model/my_checkpoint/xgan_001')
            log_result(dataset, xgan, "xgan", 0, 20, False)
            log_result(dataset, xgan, "xgan", 550, 570, True)



    #***************DiscoGAN************************
    if arch == 'disco':
        
        disco = DiscoGAN()
        if mode == "train":
            print('Compiling DiscoGAN model...')
            disco.compile(
                generator_optimizer=tf.keras.optimizers.Adam(learning_rate=D_G_LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
                discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=D_D_LEARN_RATE, beta_1=BETA[0], beta_2=BETA[1]),
                loss_function= tf.keras.losses.BinaryCrossentropy(from_logits= False)
            )
            print('Training DiscoGAN model...')
            disco.fit(dataset[:num_of_batches], epochs=D_EPOCHS, callbacks=[ProgressGrid(dataset, disco, 'disco', num_of_batches)])
            disco.save_weights('./model/my_checkpoint/disco_001')
        else:
            print('Loading DiscoGAN model...')
            disco.load_weights('./model/my_checkpoint/disco_001')
            log_result(dataset, disco, "discogan", 0, 20, False)
            log_result(dataset, disco, "discogan", 550, 570, True)


if __name__ == '__main__':
    main()