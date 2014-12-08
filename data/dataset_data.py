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
        n = self.number_of_samples()
        labels = self.labels
        n_test  = 2

        training_indexes = []
        testing_indexes = []

        # We want every 3rd article to be a testing article (assuming that every author has exactly 3 documents!)
        for label in set(labels):
            indexes_test = [i for i,x in enumerate(labels) if x == label][-n_test:]
            indexes_train = [i for i,x in enumerate(labels) if x == label][0:-n_test]
            for i in indexes_test:
                testing_indexes.append(i)
            for i in indexes_train:
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
        self.labels = dataset.all_author_ids()
        for feature in self.features:
            # TODO: use a subset of the texts
            all_text = dataset.all_texts()
            feature_matrix = feature.quantize_feature(all_text,self.labels)
            self.data =np.concatenate((self.data, feature_matrix),1)

