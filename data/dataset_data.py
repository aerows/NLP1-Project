import numpy.random as rnd
import numpy as np
from data import Data
from datasets import mysql_dataset

class DatasetData(Data):

    def __init__(self,dataset,features):
        Data.__init__(self)
        self.data, self.labels = self._compute_features_dataset(dataset)
    def _data(self):
        return self.data

    def _labels(self):
        return self.labels

    def _number_of_samples(self):
        return len(self.labels)

    def _compute_features_dataset(self,dataset):
        self.data = []
        self.labels = []
        for feature in self.features:
             self.data = [self.data, feature.quantize_feature(dataset.all_texts())]
        self.labels = dataset.all_author_ids()
