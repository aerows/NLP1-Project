from unittest import TestCase
from datasets.mysql_dataset import MysqlDataset
__author__ = 'bas'


class TestMysqlDataset(TestCase):
    def setUp(self):
        self.dataset = MysqlDataset("small_article")

    def test_all_author_ids(self):
        author_ids = self.dataset.all_author_ids()
        self.assertEqual(author_ids[0], 1)
        self.assertEqual(author_ids[32], 9)

    def test_dataset_size(self):
        size = self.dataset.dataset_size()
        self.assertEqual(size, 33)

    def test_all_texts(self):
        texts = self.dataset.all_texts()
        self.assertIn("Warm, sunny, and clear",texts[0])
        self.assertIn("When America experienced a downfall on September 11",texts[32])