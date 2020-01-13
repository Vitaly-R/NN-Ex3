import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from Model import Ex3Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def batch_data(x, batches=30):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    sx = x[indices]
    batch_size = indices.shape[0] // batches
    x_batches = list()
    for i in range(batches):
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
        x_batches.append(x_batch[..., np.newaxis])
    return x_batches


# @tf.function
def training_step(batch, model, training_loss, optimizer, loss_func):
    with tf.GradientTape() as tape:
        predictions = model(batch)
        # loss = loss_func(batch, predictions)
        loss = tf.reduce_sum(tf.square(batch - predictions), axis=(1, 2))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    training_loss(loss)


# @tf.function
def test_step(batch, model, test_loss):
    predictions = model(batch, training=False)
    test_loss(batch, predictions)


def q1(epochs=1000):
    (x_train, _), (x_test, _) = mnist.load_data()
    training_batches = batch_data(x_train)
    test_batches = batch_data(x_test)
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    model = Ex3Model()
    training_losses = list()
    test_losses = list()

    for i in range(1, epochs + 1):
        j = 0
        for batch in training_batches:
            training_step(batch, model, training_loss, optimizer, loss_func)
            training_losses.append(training_loss.result())
            j += 1
            if (j == 1) or (not (j % 10)):
                print('epoch', i, 'round', j, '- training loss:', training_losses[-1])

        for batch in test_batches:
            test_step(batch, model, test_loss)
            test_losses.append(test_loss.result())

    image = test_batches[-1][0]
    image = image[np.newaxis, ...]
    prediction = model(image)
    plt.figure()
    plt.title('original image')
    plt.imshow(image[0, :, :, 0], cmap='gray')

    plt.figure()
    plt.title('predicted image')
    plt.imshow(prediction[0, :, :, 0], cmap='gray')

    # latent_vectors = model.encode(test_batches[0])
    # pca = PCA(n_components=2)
    # points = pca.fit_transform(latent_vectors)
    # plt.figure()
    # plt.title('Resulting clusters PCA on latent vectors')
    # plt.scatter(points[:, 0], points[:, 1], color='navy')
    plt.show()
