import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Model import plot_losses, project_2d


def display_denoising_examples(model, noisy_images, original_images):
    fig = plt.figure(figsize=(20, 6), facecolor='white')
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


def q2(epochs, model, training_images, test_images, test_labels, batch_size=500):
    noisy_training_images = np.clip(training_images + np.random.random(size=training_images.shape), 0, 1).astype('float32')
    noisy_test_images = np.clip(test_images + np.random.random(size=test_images.shape), 0, 1).astype('float32')
    optimizer = tf.keras.optimizers.Adam()
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    training_losses = list()
    test_losses = list()
    training_batches = tf.data.Dataset.from_tensor_slices((noisy_training_images, training_images)).batch(batch_size)
    test_batches = tf.data.Dataset.from_tensor_slices((noisy_test_images, test_images)).batch(batch_size)
    for i in range(1, epochs + 1):
        for noisy_batch, original_batch in training_batches:
            with tf.GradientTape() as tape:
                predictions = model(noisy_batch)
                loss = tf.reduce_mean(tf.square(original_batch - predictions), axis=(1, 2))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            training_loss(loss)
        training_losses.append(training_loss.result())
        for noisy_batch, original_batch in test_batches:
            predictions = model(noisy_batch)
            loss = tf.reduce_mean(tf.square(original_batch - predictions), axis=(1, 2))
            test_loss(loss)
        test_losses.append(test_loss.result())
        print("epoch {}: training loss - {} | test loss - {}".format(i, training_losses[-1], test_losses[-1]))

    plot_losses(training_losses, test_losses, 'Training and Test Loss\nafter {} epochs'.format(epochs))
    display_denoising_examples(model, noisy_test_images[:10], test_images[:10])
    project_2d(model, noisy_test_images, test_labels, 'Projection of Latent Vectors of Image Denoising to 2D With PCA\nafter {} epochs'.format(epochs))
    plt.show()
