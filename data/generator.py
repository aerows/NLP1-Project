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
        self.means = rnd.uniform(mean_low, mean_high, (k, m))
        self.divs = rnd.uniform(div_low, div_high, (k, m))
        self.data, self.labels = self._generateData(k, n, m, self.means, self.divs)

    def _data(self):
        return self.data

    def _labels(self):
        return self.labels

    def _number_of_samples(self):
        return len(self.labels)

    def _generateData(self, k, n, m,  means, divs):
        data = np.zeros((n, m))
        labels = np.zeros(n)
        for i in range(n):
            label = rnd.randint(0, k)
            labels[i] = label
            for j in range(m):
                data[i,j] = rnd.normal(means[label, j], divs[label, j],1)
        return data, labels
