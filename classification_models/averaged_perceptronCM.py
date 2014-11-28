# -*- coding: utf-8 -*-
from classification_model import ClassificationModel
from preprocess import OneHotEncoder
import numpy as np


class AveragedPerceptronCM(ClassificationModel):
    """ Multi-class averaged perceptron.

    Kwargs:
        max_iter (int):  Number of iterations the classifier should be trained (default 100)

    Attributes:
        w (float np-array):       Normalized feature weights
        b (float np-array):       Normalized bias
        encoder (OneHotEncoder):  Translator from categorical labels to one hot format

    """

    def __init__(self, max_iter=100):
        ClassificationModel.__init__(self)
        self.max_iter = max_iter                                # Maximum number of iterations
        self.w = None                                           # Normalized weight matrix
        self.b = None                                           # Normalized bias vector
        self.encoder = OneHotEncoder()                          # One-hot encoder

    def _train_classifier(self, labels, data):

        oh_labels = self.encoder.encode(labels, 1, -1)          # Encode labels with values 1 and -1

        _, k = np.shape(oh_labels)                              # Extract numbers of parameters
        n, m = np.shape(data)                                   # k for classes, n for samples, m for features

        w = np.zeros((m, k))                                    # Instantiate weight matrix
        u = np.zeros((m, k))                                    # Instantiate weighted weight matrix
        b = np.zeros(k)[np.newaxis]                             # Instantiate bias vector
        beta = np.zeros(k)[np.newaxis]                          # Instantiate weighted bias vector
        c = 1                                                   # Instantiate weight counter

        for _ in range(self.max_iter):                          # For number of iterations
            for i in range(n):                                  # For every sample
                t = np.array(oh_labels[i, :])[np.newaxis]       # t is the ith label
                d = np.array(data[i, :])[np.newaxis]            # d is the ith data sample

                if np.argmax(d.dot(w) + b) != np.argmax(t):     # If t is not the most likely label for d
                    w += d.T.dot(t)                             # Update the weight matrix
                    b += t                                      # Update the bias vector
                    u += d.T.dot(t) * c                         # Update the weighted weight matrix
                    beta += t * c                               # Update the weighted bias vector
                c += 1                                          # Update weight counter

        self.w = w - (1/c) * u                                  # Store normalized weight matrix
        self.b = b - (1/c) * beta                               # Store normalized bias vector

    def _classify_data(self, data):
        soft_one_hot = (data.dot(self.w) + self.b)                  # Predicted label values
        predicted_labels = self.encoder.decode_soft(soft_one_hot)   # Most likely predicted label values
        return predicted_labels
