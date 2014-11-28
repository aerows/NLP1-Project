import numpy.random as rnd
import numpy as np
import MySQLdb as mdb

class MysqlDataset(object):
    def __init__(self, table_name):
        self.table_name = table_name
        self.con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
        self.cur = self.con.cursor()

    def all_author_ids(self):
        self.cur.execute("SELECT author_id FROM %s" % self.table_name)
        results = self.cur.fetchall()
        return [doc[0] for doc in results]

    def dataset_size(self):
        self.cur.execute("SELECT COUNT(*) FROM %s" % self.table_name)
        results = self.cur.fetchall()
        return results[0][0]

    def all_texts(self):
        self.cur.execute("SELECT text FROM %s" % self.table_name)
        results = self.cur.fetchall()
        return [doc[0] for doc in results]
