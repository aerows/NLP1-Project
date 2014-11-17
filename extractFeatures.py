
import numpy as np
import MySQLdb as mdb
from sklearn import ensemble
import pickle
from feature_lib.helper_functions import *
from feature_extractors.word_freq import WordFreqFE

INDEX_ID        = 0
INDEX_TOPIC_ID  = 1
INDEX_TEXT      = 2
INDEX_AUTHOR_ID = 3

def texts_from_corpus(corpus):
    return [d[INDEX_TEXT] for d in corpus]



con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
cur = con.cursor()
cur.execute("SELECT * FROM `small_article`")
results = cur.fetchall()
# Parameters:
num_words = 300

train_corpus = []
test_corpus = []
for row in results:
    id,author_id,topic_id,_,text,is_train = row
    if int(topic_id) is not 3:
        train_corpus.append((id,topic_id,text.decode('utf-8'),author_id))
    else:
        test_corpus.append((id,topic_id,text.decode('utf-8'),author_id))


word_freq_fe = WordFreqFE(num_words=num_words)
train_word_freq_matrix = word_freq_fe.quantize_feature(texts_from_corpus(train_corpus))
test_word_freq_matrix = word_freq_fe.quantize_feature(texts_from_corpus(test_corpus))

train_author_ids = [d[INDEX_AUTHOR_ID] for d in train_corpus]
test_author_ids = [d[INDEX_AUTHOR_ID] for d in test_corpus]
# Saving the objects:
with open('word_freq.pickle', 'w') as f:
    pickle.dump([train_word_freq_matrix, test_word_freq_matrix, word_freq_fe, train_author_ids,test_author_ids], f)







