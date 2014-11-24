# -*- coding: utf-8 -*-
from classification_model import ClassificationModel
from preprocess import OneHotEncoder
import numpy as np


class AveragedPerceptronCM(ClassificationModel):
    def __init__(self):
        ClassificationModel.__init__(self)        
        self.w = None
        self.b = None
        self.encoder = None

    def _train_classifier(self, labels, data):
        iterations = 100

        self.encoder = OneHotEncoder()
        oh_labels = self.encoder.encode(labels, 1, -1)

        _, k = np.shape(oh_labels)
        n, m = np.shape(data)

        w = np.zeros((m, k))
        u = np.zeros((m, k))
        b = np.zeros(k)[np.newaxis]
        beta = np.zeros(k)[np.newaxis]
        c = 1
        for _ in range(iterations):
            for i in range(n):
                t = np.array(oh_labels[i, :])[np.newaxis]
                d = np.array(data[i, :])[np.newaxis]

                if np.argmax(d.dot(w) + b) != np.argmax(t):
                    w += d.T.dot(t)
                    b += t
                    u += d.T.dot(t) * c
                    beta += t * c
                c += 1
        self.w = w - (1/n) * u
        self.b = b - (1/n) * beta

    def _classify_data(self, data):
        soft_one_hot = (data.dot(self.w) + self.b)
        predicted_labels = self.encoder.decode_soft(soft_one_hot)
        return predicted_labels
