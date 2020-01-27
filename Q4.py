from Model import Decoder, plot_losses
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def display_generating_examples(model, noise_vectors):
    """
    Displays an example of generating images from a GLO model.
    :param model: The glo model which produces the images.
    :param noise_vectors: The noise vectors which are mapped to mnist images.
    """
    fig = plt.figure()
    plt.title('Image Generating Examples by a GLO model')
    plt.xticks([])
    plt.yticks([])
    images = model(noise_vectors)[:5, :, :, :]
    for i in range(1, len(noise_vectors) + 1):
        fig.add_subplot(1, len(noise_vectors), i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i - 1, :, :, 0], cmap='gray')


def q4(training_images, epochs=1000, latent_dim=10, batch_size=500):
    """
    The main function of question 4.
    Trains a GLO model to map vectors from the unit sphere in latent_dim-dimensions to mnist digits.
    :param training_images: A training set of mnist digits whose distribution we attempt to learn.
    :param epochs: Number of training epochs.
    :param latent_dim: Dimension of the latent vectors in which we wish to learn the distribution.
    :param batch_size: Size of a single batch in training.
    """
    model = Decoder()
    # Creating noise vectors in latent_dim dimensions, and projecting them on the unit sphere.
    noise_data = np.random.normal(size=(training_images.shape[0], latent_dim)).astype('float32')
    noise_data = noise_data / (np.linalg.norm(noise_data, axis=1)[..., np.newaxis])
    # Creating training batches.
    training_batches = tf.data.Dataset.from_tensor_slices((training_images, noise_data)).batch(batch_size)
    # Defining optimizers, and a loss metric.
    z_optimizer = tf.keras.optimizers.Adam(1e-2)
    glo_optimizer = tf.keras.optimizers.Adam(1e-3)
    training_loss_metric = tf.keras.metrics.Mean()
    training_loss = list()
    shown = False
    # Training loop.
    for epoch in range(1, epochs + 1):
        for original_images_batch, noise_batch in training_batches:
            # Representing the noise batch as a tensorflow variable so we could take the derivative of the loss w.r.t the batch.
            noise_vector_batch = tf.Variable(noise_batch)
            with tf.GradientTape() as z_tape, tf.GradientTape() as glo_tape:
                generated_images_batch = model(noise_vector_batch)
                z_loss = tf.square(tf.norm(original_images_batch - generated_images_batch, axis=1))
                glo_loss = tf.reduce_mean(z_loss)
            z_gradients = z_tape.gradient(z_loss, [noise_vector_batch, ])
            z_optimizer.apply_gradients(zip(z_gradients, [noise_vector_batch, ]))
            glo_gradients = glo_tape.gradient(glo_loss, model.trainable_variables)
            glo_optimizer.apply_gradients(zip(glo_gradients, model.trainable_variables))
            training_loss_metric(glo_loss)
            # After applying the gradients, we need to update the original batch, which we want to train. We also project the resulting vectors to the unit sphere.
            noise_batch = noise_vector_batch / (tf.norm(noise_vector_batch, axis=1)[..., tf.newaxis])
            # Visualization of the progress is done using training data, and thus placed in the loop.
            if not shown:
                if (epoch in [1, epochs]) or (epochs < 5) or not (epoch % (epochs // 5)):
                    fig = plt.figure()
                    plt.title('GLO Training Progress Visualization \nepoch {}'.format(epoch))
                    plt.xticks([])
                    plt.yticks([])
                    noise = noise_batch[: 3, :]
                    images = model(noise)
                    for i in range(1, 4):
                        fig.add_subplot(1, 3, i)
                        plt.xticks([])
                        plt.yticks([])
                        plt.imshow(images[i - 1, :, :, 0], cmap='gray')
                    shown = True
        training_loss.append(training_loss_metric.result())
        shown = False
        print("epoch {}: training loss - {} ".format(epoch, training_loss[-1]))

    plot_losses(training_loss, [])
    for original_images_batch, noise_batch in training_batches:
        display_generating_examples(model, noise_batch[:5, :])
        break
