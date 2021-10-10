#*******************************************
#*************DATASET SETTINGS**************
#*******************************************

CARTOON_DATASET = './cartoon_dataset/'
CELEBRITY_DATASET = './celebrity_dataset/'

PRELOADED_DATA = True # use preloaded data, if loading new data it should be set to False
BATCH_SIZE = 32
IMAGE_SIZE = (64,64)

#*******************************************
#************MODEL PARAMETERS***************
#*******************************************

BETA =  (0.5,0.999)

#XGAN
X_EPOCHS = 131
X_G_LEARN_RATE = 3e-4
X_D_LEARN_RATE = 3e-4
X_C_LEARN_RATE = 3e-4

#DiscoGAN
D_EPOCHS = 101
D_G_LEARN_RATE = 2e-4
D_D_LEARN_RATE = 2e-4

#*******************************************
#**************CONFIGURATIONS***************
#*******************************************

arch="disco" # Choose 'disco' for DiscoGAN or 'xgan' for XGAN
mode="load" # 'train' or 'load'

