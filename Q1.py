import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Model import plot_losses, project_2d


def display_reconstruction_examples(model, images):
    fig = plt.figure(figsize=(20, 4))
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


def q1(epochs, model, training_images, test_images, test_labels, batch_size=500):
    optimizer = tf.keras.optimizers.Adam()
    training_loss = tf.keras.metrics.Mean(name='training_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    training_losses = list()
    test_losses = list()
    training_batches = tf.data.Dataset.from_tensor_slices(training_images).batch(batch_size)
    test_batches = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
    for i in range(1, epochs + 1):
        for batch in training_batches:
            with tf.GradientTape() as tape:
                predictions = model(batch)
                loss = tf.reduce_mean(tf.square(batch - predictions), axis=(1, 2))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            training_loss(loss)
        training_losses.append(training_loss.result())
        for (batch, _) in test_batches:
            predictions = model(batch)
            loss = tf.reduce_mean(tf.square(batch - predictions), axis=(1, 2))
            test_loss(loss)
        test_losses.append(test_loss.result())
        print("epoch {}: training loss - {} | test loss - {}".format(i, training_losses[-1], test_losses[-1]))

    plot_losses(training_losses, test_losses, 'Training and Test Loss\nafter {} epochs'.format(epochs))
    display_reconstruction_examples(model, test_images[:10])
    project_2d(model, test_images, test_labels, 'Projection of Latent Vectors of Image Reconstruction to 2D With PCA\nafter {} epochs'.format(epochs))
