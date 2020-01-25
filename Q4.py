from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Activation, Reshape
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class GLO(Model):

    def __init__(self):
        super(GLO, self).__init__()
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')

        self.dense2 = Dense(10)
        self.dense3 = Dense(512)
        self.dense4 = Dense(7 * 7 * 64)
        self.reshape = Reshape((7, 7, 64))
        self.convt1 = Conv2DTranspose(32, 2, 2, padding='valid')
        self.convt2 = Conv2DTranspose(1, 2, 2, padding='valid')

    def __call__(self, x, *args, **kwargs):
        y = self.relu(self.dense2(x))
        y = self.relu(self.dense3(y))
        y = self.relu(self.dense4(y))
        y = self.reshape(y)
        y = self.relu(self.convt1(y))
        y = self.convt2(y)
        return self.sigmoid(y)


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


# @tf.function
def train_step(batch, noise, generator, generator_optimizer, loss_func):

    with tf.GradientTape() as gen_tape:
        project_noise = project(noise)
        reconstructions = generator(project_noise)
        loss = loss_func(batch, reconstructions)

    gradients = gen_tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))


def glo_loss(batch, reconstructions):
    return tf.reduce_mean(tf.square(batch - reconstructions))


def project(batch, is_numpy=False):
    if is_numpy:
        return batch / np.sqrt(np.sum(batch ** 2, axis=1))[:, np.newaxis]
    return batch / tf.sqrt(tf.reduce_sum(tf.square(batch), axis=1, keepdims=True))


def q4(epochs=1000):
    glo = GLO()
    glo_optimizer = tf.keras.optimizers.Adam(1e-4)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    training_x_batches, training_y_batches = batch_data(x_train, y_train)
    latent_dim = 10
    for epoch in range(1, epochs + 1):

        if epoch == 1 or (epoch % 10 == 0):
            print("epoch " + str(epoch) + " out of " + str(epochs))

        for batch in training_x_batches:
            noise = tf.random.normal([batch.shape[0], latent_dim])
            train_step(batch, noise, glo, glo_optimizer, glo_loss)

    test_sample = tf.random_normal([1, 10])

    pred = glo(test_sample)
    plt.imshow(pred[0, :, :, 0], cmap='gray')
    plt.show()

if __name__ == '__main__':
    q4(epochs=1)








