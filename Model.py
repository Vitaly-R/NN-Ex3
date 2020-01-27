"""
This module contains the NN Model classes, functions, and data structures used by multiple parts in the exercise or by the main function of the main module.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras import Model, Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, LeakyReLU, Dropout

""" A color dictionary for scattering points in 2D """
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


def plot_losses(training_loss, test_loss, title='', training_loss_label='training loss', test_loss_label='test loss'):
    """
    Plots training and test losses as a function of epochs.
    (Can also be used to plot any two losses)
    :param training_loss: A list of training loss values
    :param test_loss: A list of test loss values
    :param title: A title for the plot.
    :param training_loss_label: Label for the training_loss plot.
    :param test_loss_label: Label for the test_loss plot.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(training_loss, label=training_loss_label)
    plt.plot(test_loss, label=test_loss_label)
    plt.legend()


def project_2d(model, images, labels, title=''):
    """
    Projects the latent vectors created by the given model for the given images on a 2D plane using PCA.
    Each projected point is colored according to the label corresponding to the image it came from.
    :param model: An autoencoder model with an encoder component.
    :param images: The images from which to create the latent vectors.
    :param labels: The labels of the images (0-9).
    :param title: A title for the figure.
    """
    pca = PCA(n_components=2)
    plt.figure()
    plt.title(title)
    projected_points = pca.fit_transform(model.encoder(images))
    for label in range(10):
        pts = projected_points[labels == label]
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[label])


def load_datasets():
    """
    Loads the mnist dataset, scales the pictures into 0-1 range, and shuffles the datasets.
    :return: The training images, the test images, and the test labels.
    """
    (training_images, _), (test_images, test_labels) = mnist.load_data()
    training_images = ((training_images.astype('float32')) / 255.0)[..., np.newaxis]
    np.random.shuffle(training_images)
    test_images = ((test_images.astype('float32')) / 255.0)[..., np.newaxis]
    indices = np.arange(test_images.shape[0])
    np.random.shuffle(indices)
    test_images = test_images[indices]
    test_labels = test_labels[indices]
    return training_images, test_images, test_labels


def get_generator_model():
    """
    Creates a sequential Generator model which receives a noise vector and generates an image from it.
    """
    model = Sequential()
    model.add(Dense(12544, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


def get_discriminator_model():
    """
    Creates a sequential Discriminator model, which determines weather an input image is real, or artificial.
    """
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1))
    return model


class Encoder(Model):
    """
    A convolutional encoder model, which encodes images into a 10-D space.
    """
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
    """
    A convolutional decoder model, which maps 10-D vectors into a 28x28x1 image.
    """
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
    """
    An autoencoder model, which encodes images to 10-D space, and then reconstructs an image of the same size.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x, **kwargs):
        y = self.encoder(x)
        return self.decoder(y)
