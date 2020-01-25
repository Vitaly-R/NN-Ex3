"""
This module contains the NN Model classes, and functions/data structures used by multiple parts in the exercise or by the main function of the main module.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Activation, Reshape


colors = {0: 'blue',
          1: 'red',
          2: 'navy',
          3: 'green',
          4: 'lime',
          5: 'yellow',
          6: 'magenta',
          7: 'pink',
          8: 'cyan',
          9: 'orange'}


def plot_losses(training_loss, test_loss, title=''):
    plt.figure()
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(training_loss, label='training loss')
    plt.plot(test_loss, label='test loss')
    plt.legend()


def project_2d(model, images, labels, title=''):
    pca = PCA(n_components=2)
    plt.figure()
    plt.title(title)
    projected_points = pca.fit_transform(model.encoder(images))
    for label in range(10):
        pts = projected_points[labels == label]
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[label])


def load_datasets():
    (training_images, _), (test_images, test_labels) = mnist.load_data()
    training_images = ((training_images.astype('float32')) / 255.0)[..., np.newaxis]
    np.random.shuffle(training_images)
    test_images = ((test_images.astype('float32')) / 255.0)[..., np.newaxis]
    indices = np.arange(test_images.shape[0])
    np.random.shuffle(indices)
    test_images = test_images[indices]
    test_labels = test_labels[indices]
    return training_images, test_images, test_labels


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(32, 3, (2, 3), padding='valid', activation='relu')
        self.conv2 = Conv2D(64, 3, (2, 2), padding='valid', activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(10, activation='relu')

    def __call__(self, x, **kwargs):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.flatten(y)
        y = self.dense1(y)
        return self.dense2(y)


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(3136, activation='relu')
        self.reshape = Reshape((7, 7, 64))
        self.tconv1 = Conv2DTranspose(32, 3, (2, 2), activation='relu', padding='same')
        self.tconv2 = Conv2DTranspose(1, 3, (2, 2), activation='sigmoid', padding='same')

    def __call__(self, x, **kwargs):
        y = self.dense1(x)
        y = self.dense2(y)
        y = self.reshape(y)
        y = self.tconv1(y)
        return self.tconv2(y)


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x, **kwargs):
        y = self.encoder(x)
        return self.decoder(y)
