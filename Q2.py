import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from Model import AEModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pic_fsize = (10, 10)
plot_fsize = (20, 15)
fcolor = 'white'


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
        x_batch = x_batch / 255
        x_batch += np.random.normal(0, 0.02, x_batch.shape)
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
def training_step(batch, model, training_loss, optimizer, loss_func):
    with tf.GradientTape() as tape:
        predictions = model(batch)
        # loss = loss_func(batch, predictions)
        loss = tf.reduce_sum(tf.square(batch - predictions), axis=(1, 2))
        # loss = tf.square(tf.norm(batch - predictions, axis=(1, 2)))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    training_loss(loss)


# @tf.function
def test_step(batch, model, test_loss):
    predictions = model(batch, training=False)
    loss = tf.reduce_sum(tf.square(batch - predictions), axis=(1, 2))
    test_loss(loss)


def q2(epochs=1000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    training_x_batches, _ = batch_data(x_train, y_train)

    plt.figure()
    plt.imshow(training_x_batches[0][0, :, :, 0], cmap='gray')
    plt.show()

    # test_x_batches, test_y_batches = batch_data(x_test, y_test)
    # training_loss = tf.keras.metrics.Mean(name='training_loss')
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # optimizer = tf.keras.optimizers.Adam()
    # loss_func = tf.keras.losses.CategoricalCrossentropy()
    # model = AEModel()
    # training_losses = list()
    # test_losses = list()

    # for i in range(1, epochs + 1):
    #     for batch in training_x_batches:
    #         training_step(batch, model, training_loss, optimizer, loss_func)
    #         training_losses.append(training_loss.result())
    #
    #     for batch in test_x_batches:
    #         test_step(batch, model, test_loss)
    #         test_losses.append(test_loss.result())
    #
    #      print('epoch', i, '- training loss:', training_losses[-1], '- test loss:', test_losses[-1])

    # next = False
    # for i in range(10):
    #     for j in range(len(test_y_batches)):
    #         for k in range(test_y_batches[j].shape[0]):
    #             if test_y_batches[j][k] == i:
    #                 image = test_x_batches[j][k]
    #                 image = image[np.newaxis, ...]
    #                 prediction = model(image)
    #                 plt.figure(figsize=pic_fsize, facecolor=fcolor)
    #                 plt.title('Original image \nClass={} \n{} epochs'.format(i, epochs))
    #                 plt.imshow(image[0, :, :, 0], cmap='gray')
    #                 plt.figure(figsize=pic_fsize, facecolor=fcolor)
    #                 plt.title('Reconstructed image \nClass={} \n{} epochs'.format(i, epochs))
    #                 plt.imshow(prediction[0, :, :, 0], cmap='gray')
    #                 next = True
    #                 break
    #         if next:
    #             next = False
    #             break

    # plt.figure(figsize=plot_fsize, facecolor=fcolor)
    # plt.title('Training loss\n{} epochs'.format(epochs))
    # plt.xlabel('training iteration')
    # plt.ylabel('sum of squared differences error')
    # plt.plot(training_losses)
    #
    # plt.figure(figsize=plot_fsize, facecolor=fcolor)
    # plt.title('Test loss\n{} epochs'.format(epochs))
    # plt.xlabel('test iteration')
    # plt.ylabel('sum of squared differences error')
    # plt.plot(test_losses)

    # pca = PCA(n_components=2)
    # latent_points = list()
    # for batch in test_x_batches[:5]:
    #     latent_vectors = model.encode(batch)
    #     points = pca.fit_transform(latent_vectors)
    #     latent_points.append(points)
    #
    # plt.figure(figsize=plot_fsize, facecolor=fcolor)
    # plt.title('Resulting clusters from PCA on latent vectors')
    # colors = {0:'blue', 1:'navy', 2:'green', 3:'red', 4:'lime', 5:'magenta', 6:'cyan', 7:'orange', 8:'yellow', 9:'slategray'}
    # batch = 0
    # for points in latent_points:
    #     for label in range(10):
    #         label_pts = points[test_y_batches[batch] == label]
    #         plt.scatter(label_pts[:, 0], label_pts[:, 1], color=colors[label])


q2(1)
