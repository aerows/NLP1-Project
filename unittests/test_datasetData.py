from unittest import TestCase
from datasets.mysql_dataset import MysqlDataset
from data.dataset_data import MysqlDatasetData
from feature_extractors.factor_stop_words import FactorStopWordsFE
from feature_extractors.ngram_freq import NGramFreq
from feature_extractors.word_freq import WordFreqFE
__author__ = 'bas'


class TestDatasetData(TestCase):
    def setUp(self):
        self.data = MysqlDatasetData(MysqlDataset("small_article"),[FactorStopWordsFE(),WordFreqFE(400)])

    def test__data(self):
        self.assertEqual(len(self.data._data()),2)
        pass

    def test__labels(self):
        # self.fail()
        pass

    def test_number_of_samples(self):
        self.assertEqual(self.data.number_of_samples(),33)

    def test__compute_features_dataset(self):
        # self.fail()
        pass