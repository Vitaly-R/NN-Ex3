"""
Main file for ex3 in Introduction to Neural Networks.
Vitaly Rajev
Sasha Tankov
"""
import matplotlib.pyplot as plt
from Model import load_datasets
from Q1 import q1
from Q2 import q2
from Q3 import q3
from Q4 import q4


def main():
    training_images, test_images, test_labels = load_datasets()
    print("Question 1")
    q1(training_images, test_images, test_labels)
    print("Question 2")
    q2(training_images, test_images, test_labels)
    print("Question 3")
    q3(training_images)
    print("Question 4")
    q4(training_images)
    plt.show()


if __name__ == '__main__':
    main()
