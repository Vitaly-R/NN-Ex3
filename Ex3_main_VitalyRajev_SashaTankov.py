from Model import Autoencoder, load_datasets
from Q1 import q1
from Q2 import q2


def main():
    training_images, test_images, test_labels = load_datasets()
    print("Question 1")
    reconstruction_model = Autoencoder()
    q1(1, reconstruction_model, training_images, test_images, test_labels)
    print("Question 2")
    denoising_model = Autoencoder()
    q2(1, denoising_model, training_images, test_images, test_labels)


if __name__ == '__main__':
    main()
