import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(layers.Layer):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.dense1 = layers.Dense(512)
        self.lrelu1 = layers.LeakyReLU(alpha=0.2)

        self.dense2 = layers.Dense(256)
        self.lrelu2 = layers.LeakyReLU(alpha=0.2)
        self.bn2 = layers.BatchNormalization(momentum=0.8)

        self.dense3 = layers.Dense(1)

    def call(self, input):

        x = self.lrelu1(self.dense1(input))
        x = self.bn2(self.lrelu2(self.dense2(x)))

        return self.dense3(x)


class Generator(layers.Layer):
    def __init__(self, input_dim, batch_size=16):
        super(Generator,self).__init__()

        self.dense1 = layers.Dense(256)
        self.lrelu1 = layers.LeakyReLU(alpha=0.2)
        self.bn1 = layers.BatchNormalization(momentum=0.8)
        self.drop1 = layers.Dropout(0,4)

        self.dense2 = layers.Dense(512)
        self.lrelu2 = layers.LeakyReLU(alpha=0.2)
        self.bn2 = layers.BatchNormalization(momentum=0.8)
        self.drop2 = layers.Dropout(0,4)

        self.dense3 = layers.Dense(1024)
        self.lrelu3 = layers.LeakyReLU(alpha=0.2)
        self.bn3 = layers.BatchNormalization(momentum=0.8)
        self.drop3 = layers.Dropout(0,4)

        self.dense4 = layers.Dense(input_dim, activation='tanh')

    def call(self, input):

        x = self.drop1(self.bn1(self.lrelu1(self.dense1(input))))
        x = self.drop2(self.bn2(self.lrelu2(self.dense2(x))))
        x = self.drop3(self.bn3(self.lrelu3(self.dense3(x))))

        return self.dense4(x)



class DUALGAN(tf.keras.Model):
    def __init__(self, batch_size=16):
        super(DUALGAN,self).__init__()

        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()

        self.generator_AB = Generator()
        self.generator_BA = Generator()

        self.g_loss_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.d_loss_metric = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)

        self.train_summary_writer = tf.summary.create_file_writer('./train_summary')

