from unittest import TestCase
from datasets.mysql_dataset import MysqlDataset
__author__ = 'bas'


class TestMysqlDataset(TestCase):
    def setUp(self):
        self.dataset = MysqlDataset("small_article")

    def test_all_author_ids(self):
        self.fail()

    def test_dataset_size(self):
        size = self.dataset.dataset_size()
        self.assertEqual(size, 33)

    def test_all_texts(self):
        self.fail()