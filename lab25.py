from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(3)
tf.random.set_seed(3)

#  generator
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))

#  discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

#  GAN model
generator_input = Input(shape=(100,))
discriminator_output = discriminator(generator(generator_input))
gan = Model(generator_input, discriminator_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

#import os
#if not os.path.exists("./gan_images"):
#    os.makedirs("./gan_images")

#  train
def gan_train(epoch, batch_size, saving_interval):
    

    # MNIST data
    (x_train, _), (_, _) = mnist.load_data()  # only train data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_train = (x_train - 127.5) / 127.5  # -1 ~ +1
    # X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

    true = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    output_dir = 'output'
 

    for i in range(epoch):
      # train discriminator with real images
      idx = np.random.randint(0, x_train.shape[0], batch_size)
      images = x_train[idx]
      discriminator_loss_real = discriminator.train_on_batch(images, true)

      # train discriminator with fake images
      noise = np.random.normal(0, 1, (batch_size, 100))
      gen_images = generator.predict(noise, verbose=0)
      discriminator_loss_fake = discriminator.train_on_batch(gen_images, fake)

      # discriminaor loss
      discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

      # train gan with fake images - gan loss
      generator_loss = gan.train_on_batch(noise, true) # note that output is true

      if i % 1000 == 0:
          print('epoch:%d' % i, ' d_loss:%.4f' % discriminator_loss, ' g_loss:%.4f' % generator_loss)
            #  Saving images
      if i % saving_interval == 0:
          # r, c = 5, 5
          noise = np.random.normal(0, 1, (25, 100))
          gen_images = generator.predict(noise)

            # Rescale images 0 - 1
          gen_images = 0.5 * gen_images + 0.5

          fig, axs = plt.subplots(5, 5)
          count = 0
          for j in range(5):
              for k in range(5):
                  axs[j, k].imshow(gen_images[count, :, :, 0], cmap='gray')
                  axs[j, k].axis('off')
                  count += 1
          fig.savefig("output%d.png" % i)
gan_train(40001, 32, 1000)
