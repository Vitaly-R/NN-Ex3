from Model import get_discriminator_model, get_generator_model, plot_losses
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def display_generating_examples(generator, noise_dim):
    """
    Displays an example of generating images from the generator of a GAN model.
    :param generator: A generator model which generates images from random noise vectors.
    :param noise_dim: Dimension of the noise vectors.
    """
    fig = plt.figure()
    plt.title('Image Generation Example by GAN')
    plt.xticks([])
    plt.yticks([])
    noise = np.random.normal(size=(25, noise_dim)).astype('float32')
    images = generator(noise)
    for i in range(1, 26):
        fig.add_subplot(5, 5, i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i - 1, :, :, 0], cmap='gray')


def q3(training_images, epochs=200, noise_dim=100, batch_size=256):
    """
    The main function of question 3.
    Trains a GAN to map random noise vectors to mnist digits such that a discriminator model will mistake the images as real.
    :param training_images: An array of images to train over.
    :param epochs: The number of epochs to train.
    :param noise_dim: Dimension of the noise vectors which the generator gets as an input.
    :param batch_size: Size of a single training batch.
    """
    # Shifting the images to range [-1, 1] and batching.
    train_dataset = tf.data.Dataset.from_tensor_slices(2 * (training_images - 0.5)).batch(batch_size)
    # Getting the discriminator and generator models.
    discriminator = get_discriminator_model()
    generator = get_generator_model()
    # Defining the optimizers, loss function, and metrics.
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    discriminator_loss_metric = tf.keras.metrics.Mean()
    generator_loss_metric = tf.keras.metrics.Mean()
    discriminator_loss = list()
    generator_loss = list()
    # Training loop.
    for epoch in range(1, epochs + 1):
        for image_batch in train_dataset:
            noise = tf.random.normal((batch_size, noise_dim))
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)
                generator_loss_val = cross_entropy(tf.ones_like(fake_output), fake_output)
                discriminator_loss_val = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
            gradients_of_generator = gen_tape.gradient(generator_loss_val, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss_val, discriminator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            discriminator_loss_metric(discriminator_loss_val)
            generator_loss_metric(generator_loss_val)
        discriminator_loss.append(discriminator_loss_metric.result())
        generator_loss.append(generator_loss_metric.result())
        print("epoch {} | Discriminator training loss - {} | Generator training loss - {}".format(epoch, discriminator_loss[-1], generator_loss[-1]))
        # Visualizing the progress with 5 images
        if (epoch in [1, epochs]) or (epochs < 10) or not (epoch % (epochs // 10)):
            fig = plt.figure()
            plt.title('GAN Training Progress Visualization \nepoch {}'.format(epoch))
            plt.xticks([])
            plt.yticks([])
            noise = np.random.normal(size=(5, noise_dim)).astype('float32')
            images = generator(noise)
            for i in range(1, 6):
                fig.add_subplot(1, 5, i)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(images[i - 1, :, :, 0], cmap='gray')
    # Plotting losses, and displaying generation examples.
    plot_losses(discriminator_loss, generator_loss, "Discriminator and Generator Training Loss \nafter {} epochs".format(epochs), "discriminator loss", "generator loss")
    display_generating_examples(generator, noise_dim)
