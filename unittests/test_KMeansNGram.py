from unittest import TestCase
from feature_extractors.kmeans_ngram import KMeansNGram
from datasets.mysql_dataset import MysqlDataset
__author__ = 'bas'


class TestKMeansNGram(TestCase):
    def setUp(self):
        self.dataset = MysqlDataset("small_article")
        self.texts = self.dataset.all_texts()
        self.feature = KMeansNGram(self.texts,n=16,step_size=1,k=400)

    def test_extract_patches(self):
        combined_patch_matrix,_ = self.feature.extract_patches()
        self.assertEqual(combined_patch_matrix.shape[1],16,"Should be a x by n matrix")

    def test_quantize_feature(self):
        feature_matrix = self.feature.quantize_feature(self.texts)
        print  feature_matrix
        self.assertEqual(feature_matrix.shape,(33,400))

