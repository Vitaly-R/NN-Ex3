import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Model import Autoencoder, plot_losses, project_2d


def display_denoising_examples(model, noisy_images, original_images):
    """
    Displays an image denoising example of the given images by the given model.
    :param model: An autoencoder model which preforms image denoising.
    :param noisy_images: An array of noisy images which will be used as an example.
    :param original_images: An array of the original images, for a visual comparison with the denoising results.
    """
    fig = plt.figure()
    plt.title('Image Denoising Examples')
    plt.ylabel("Denoised                          Noisy                          Original")
    plt.xticks([])
    plt.yticks([])
    i = 1
    for noisy_image, original_image in list(zip(noisy_images, original_images)):
        denoised = model(noisy_image[np.newaxis, ...])
        fig.add_subplot(3, len(noisy_images), i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(original_image[:, :, 0], cmap='gray')
        fig.add_subplot(3, len(noisy_images), i + len(noisy_images))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(noisy_image[:, :, 0], cmap='gray')
        fig.add_subplot(3, len(noisy_images), i + 2 * len(noisy_images))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(denoised[0, :, :, 0], cmap='gray')
        i += 1


def q2(training_images, test_images, test_labels, epochs=1000, batch_size=500):
    """
    Main function of question 2.
    Trains the an autoencoder model to denoise images for the given number of epochs over the given dataset.
    Then, plots the training and test losses, and generates a reconstruction example.
    :param training_images: Set of training images.
    :param test_images: Set of test images.
    :param test_labels: Set of labels corresponding to the test images.
    :param epochs: Number of epochs to train.
    :param batch_size: Size of each batch of images for training and testing.
    """
    model = Autoencoder()
    # Creating the noisy images dataset.
    noisy_training_images = np.clip(training_images + np.random.random(size=training_images.shape), 0, 1).astype('float32')
    noisy_test_images = np.clip(test_images + np.random.random(size=test_images.shape), 0, 1).astype('float32')
    # Defining the optimizer and metrics for the model.
    optimizer = tf.keras.optimizers.Adam()
    training_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()
    training_losses = list()
    test_losses = list()
    # Batching the data.
    training_batches = tf.data.Dataset.from_tensor_slices((noisy_training_images, training_images)).batch(batch_size)
    test_batches = tf.data.Dataset.from_tensor_slices((noisy_test_images, test_images)).batch(batch_size)
    for i in range(1, epochs + 1):
        # Training loop.
        for noisy_batch, original_batch in training_batches:
            with tf.GradientTape() as tape:
                predictions = model(noisy_batch)
                loss = tf.reduce_mean(tf.square(original_batch - predictions), axis=(1, 2))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            training_loss(loss)
        training_losses.append(training_loss.result())
        # Testing loop.
        for noisy_batch, original_batch in test_batches:
            predictions = model(noisy_batch)
            loss = tf.reduce_mean(tf.square(original_batch - predictions), axis=(1, 2))
            test_loss(loss)
        test_losses.append(test_loss.result())
        print("epoch {}: training loss - {} | test loss - {}".format(i, training_losses[-1], test_losses[-1]))
    # Plotting losses, displaying an example, and showing projection of latent vectors to 2D.
    plot_losses(training_losses, test_losses, 'Training and Test Loss\nafter {} epochs'.format(epochs))
    display_denoising_examples(model, noisy_test_images[:10], test_images[:10])
    project_2d(model, noisy_test_images, test_labels, 'Projection of Latent Vectors of Image Denoising to 2D With PCA\nafter {} epochs'.format(epochs))
