import MySQLdb as mdb

from data import Data

INDEX_ID = 0
INDEX_TOPIC_ID = 1
INDEX_TEXT = 2
INDEX_AUTHOR_ID = 3


class SmallArticle(Data):

    def __init__(self):
        Data.__init__(self)
        self.train_corpus = []
        self.test_corpus = []
        self._generateData()

    def _data(self):
        return []

    def _labels(self):
        return []

    def _number_of_samples(self):
        return []

    def training_corpus(self):
        return [row[INDEX_TEXT] for row in self.train_corpus], [row[INDEX_AUTHOR_ID] for row in self.train_corpus]

    def testing_corpus(self):
        return [row[INDEX_TEXT] for row in self.test_corpus], [row[INDEX_AUTHOR_ID] for row in self.test_corpus]

    def _generateData(self):
        con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
        cur = con.cursor()
        cur.execute("SELECT * FROM `small_article`")
        results = cur.fetchall()

        train_corpus = []
        test_corpus = []
        for row in results:
            row_id, author_id, topic_id, text, is_train = row
            if int(topic_id) is not 3:
                train_corpus.append((row_id, topic_id, text.decode('utf-8'), author_id))
            else:
                test_corpus.append((row_id, topic_id, text.decode('utf-8'), author_id))
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
