from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Activation, Reshape
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np


class Generator:
    def __init__(self):
        super(Generator, self).__init__()
        # activations
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')

        self.dense3 = Dense(512)
        self.dense4 = Dense(7 * 7 * 64)
        self.reshape = Reshape((7, 7, 64))
        self.convt1 = Conv2DTranspose(32, 2, 2, padding='valid')
        self.convt2 = Conv2DTranspose(1, 2, 2, padding='valid')

    def __call__(self, x, *args, **kwargs):
        y = self.relu(self.dense3(x))
        y = self.relu(self.dense4(y))
        y = self.reshape(y)
        y = self.relu(self.convt1(y))
        y = self.convt2(y)
        return self.sigmoid(y)


class Discriminator(Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        # activations
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')

        self.conv1 = Conv2D(32, 3, 2, 'valid')
        self.conv2 = Conv2D(64, 3, 2, 'valid')
        self.flatten = Flatten()
        self.dense1 = Dense(512)
        self.dense2 = Dense(1)  # classify as real

    def __call__(self, x, *args, **kwargs):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.flatten(y)
        y = self.relu(self.dense1(y))
        y = self.relu(self.dense2(y))
        return y


class GANModel(Model):

    def __init__(self):
        super(GANModel, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()

    def __call__(self, x, *args, **kwargs):
        # TODO check later, might be wrong
        y_gen = self.generator(x)
        y_disc = self.discriminator(x)
        return y_gen, y_disc


def batch_data(x, y, batches=30):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    sx = x[indices]
    sy = y[indices]
    batch_size = indices.shape[0] // batches
    x_batches = list()
    y_batches = list()
    for i in range(batches):
        y_batch = np.array(sy[i * batch_size: (i + 1) * batch_size])
        x_batch = np.array(sx[i * batch_size: (i + 1) * batch_size])
        x_batch = x_batch.astype(np.float32)
        mins = np.min(x_batch, axis=(1, 2))
        mins = mins[..., np.newaxis, np.newaxis]
        x_batch = x_batch - mins
        maxes = x_batch.max(axis=(1, 2))
        maxes = maxes[..., np.newaxis, np.newaxis]
        x_batch = x_batch / maxes
        if len(x_batch) < batch_size:
            x_batch = np.concatenate(x_batch, sx[: batch_size - len(x_batch)])
            y_batch = np.concatenate(y_batch, sy[: batch_size - len(x_batch)])
        x_batches.append(x_batch[..., np.newaxis])
        y_batches.append(y_batch)
    return x_batches, y_batches


def discriminator_loss(real_output, fake_output, loss_function):
    real_loss = loss_function(tf.ones_like(real_output), real_output)
    fake_loss = loss_function(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(fake_output, loss_function):
    return loss_function(tf.ones_like(fake_output), fake_output)


def train_step(batch, batch_noise, generator, discriminator, gen_optimizer, disc_optimizer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(batch_noise)

        real_output = discriminator(batch)
        fake_output = discriminator(generated_images)

        gen_loss = generator_loss(fake_output, generator_loss)
        disc_loss = discriminator_loss(real_output, fake_output, discriminator_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss)
    gradients_of_discriminator = disc_tape.gradient(disc_loss)

    gen_optimizer.apply_gradients(gradients_of_generator)
    disc_optimizer.apply_gradients(gradients_of_discriminator)


def q3(epochs=1000):

    discriminator = Discriminator()
    generator = Generator()
    gan = GANModel()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    training_x_batches, training_y_batches = batch_data(x_train, y_train)
    test_x_batches, test_y_batches = batch_data(x_test, y_test)
    batch_size = len(training_x_batches[0])
    noise_dim = 100

    for epoch in range(epochs):
        for batch in training_x_batches:
            noise = tf.random.normal([batch_size, noise_dim])
            train_step(batch, noise, generator, discriminator, generator_optimizer, discriminator_optimizer, )


