__author__ = 'Daniel'
from unittest import TestCase
import feature_lib.helper_functions as hf
import feature_extractors.spectral_clustering as sc
from datasets.mysql_dataset import MysqlDataset
from data.dataset_data import MysqlDatasetData
from feature_extractors.factor_stop_words import FactorStopWordsFE
from feature_extractors.word_freq import WordFreqFE


class TestSpectralClustering(TestCase):
    def setUp(self):
        self.data = MysqlDatasetData(MysqlDataset("small_article"), [FactorStopWordsFE(), WordFreqFE(400)])

    def testFitSpectralClustering(self):
        # Create vocab
        vocab = hf.vocabulary()
        # Init SpectralClustering
        # Extract features from data
        # Train classifier
        #


