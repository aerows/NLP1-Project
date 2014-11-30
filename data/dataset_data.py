import numpy.random as rnd
import numpy as np
from data import Data
from datasets import mysql_dataset

class MysqlDatasetData(Data):

    def __init__(self,dataset,features):
        Data.__init__(self)
        self.features = features
        self.dataset = dataset
        self._compute_features_dataset(dataset)
    def fold(self):
        # TODO: generalize this for all dataset models
        n = self.number_of_samples()

        assert n % 3  == 0, "Every author should have 3 documents!"

        training_indexes = []
        testing_indexes = []

        # We want every 3th article to be a testing article (assuming that every author has exactly 3 documents!)
        for i in range(n):
            if i % 3 == 0:
                testing_indexes.append(i)
            else:
                training_indexes.append(i)
        return (self._data())[training_indexes], (self._labels())[training_indexes], self._data()[testing_indexes], self._labels()[testing_indexes]
    def _data(self):
        return self.data

    def _labels(self):
        return self.labels

    def number_of_samples(self):
        return len(self.labels)

    def _compute_features_dataset(self,dataset):
        self.data = np.zeros((dataset.dataset_size(),0))
        self.labels = []
        for feature in self.features:
            # TODO: Properly concatenate the feature matrices
            feature_matrix = feature.quantize_feature(dataset.all_texts())
            self.data =np.concatenate((self.data, feature_matrix),1)
        self.labels = dataset.all_author_ids()
