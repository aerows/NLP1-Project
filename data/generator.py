import numpy.random as rnd
import numpy as np
from data import Data

mean_low = -10.
mean_high = 10
div_low = 2
div_high = 5


class Generator(Data):

    def __init__(self, k=10, n=100, m=50):
        Data.__init__(self)
        self.k = k
        self.m = m
        self.means = rnd.uniform(mean_low, mean_high, (self.k, self.m))
        self.divs = rnd.uniform(div_low, div_high, (self.k, self.m))
        self.data, self.labels = self.generate_data(n)

    def _data(self):
        return self.data

    def _labels(self):
        return self.labels

    def number_of_samples(self):
        return len(self.labels)

    def generate_data(self, n):
        data = np.zeros((n, self.m))
        labels = np.zeros(n)
        for i in range(n):
            label = rnd.randint(0, self.k)
            labels[i] = label
            for j in range(self.m):
                data[i,j] = rnd.normal(self.means[label, j], self.divs[label, j],1)
        return data, labels
