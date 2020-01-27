import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Model import Autoencoder, plot_losses, project_2d


def display_reconstruction_examples(model, images):
    """
    Displays an image reconstruction example of the given images by the given model.
    :param model: An autoencoder model which reconstructs images.
    :param images: An array of images which will be used as an example.
    """
    fig = plt.figure()
    plt.title('Image Reconstruction Example')
    plt.ylabel('Reconstructed                Original      ')
    plt.xticks([])
    plt.yticks([])
    i = 1
    for image in images:
        fig.add_subplot(2, len(images), i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image[:, :, 0], cmap='gray')
        reconstruction = model(image[np.newaxis, ...])
        fig.add_subplot(2, len(images), i + len(images))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(reconstruction[0, :, :, 0], cmap='gray')
        i += 1


def q1(training_images, test_images, test_labels, epochs=1000, batch_size=500):
    """
    Main function of question 1.
    Trains the an autoencoder model to reconstruct images for the given number of epochs over the given dataset.
    Then, plots the training and test losses, and generates a reconstruction example.
    :param training_images: Set of training images.
    :param test_images: Set of test images.
    :param test_labels: Set of labels corresponding to the test images.
    :param epochs: Number of epochs to train.
    :param batch_size: Size of each batch of images for training and testing.
    """
    model = Autoencoder()
    # Defining the optimizer and metrics for the model.
    optimizer = tf.keras.optimizers.Adam()
    training_loss_metric = tf.keras.metrics.Mean()
    test_loss_metric = tf.keras.metrics.Mean()
    training_loss = list()
    test_loss = list()
    # Batching the data.
    training_batches = tf.data.Dataset.from_tensor_slices(training_images).batch(batch_size)
    test_batches = tf.data.Dataset.from_tensor_slices(test_images).batch(batch_size)
    for i in range(1, epochs + 1):
        # Training loop.
        for batch in training_batches:
            with tf.GradientTape() as tape:
                predictions = model(batch)
                loss = tf.reduce_mean(tf.square(batch - predictions), axis=(1, 2))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            training_loss_metric(loss)
        training_loss.append(training_loss_metric.result())
        # Testing loop.
        for batch in test_batches:
            predictions = model(batch)
            loss = tf.reduce_mean(tf.square(batch - predictions), axis=(1, 2))
            test_loss_metric(loss)
        test_loss.append(test_loss_metric.result())
        print("epoch {}: training loss - {} | test loss - {}".format(i, training_loss[-1], test_loss[-1]))
    # Plotting losses, displaying an example, and showing projection of latent vectors to 2D.
    plot_losses(training_loss, test_loss, 'Training and Test Loss\nafter {} epochs'.format(epochs))
    display_reconstruction_examples(model, test_images[:10])
    project_2d(model, test_images, test_labels, 'Projection of Latent Vectors of Image Reconstruction to 2D With PCA\nafter {} epochs'.format(epochs))
