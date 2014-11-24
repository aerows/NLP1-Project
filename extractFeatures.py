
import numpy as np
import MySQLdb as mdb
from sklearn import ensemble
import pickle
from feature_lib.helper_functions import *
from feature_extractors.word_freq import WordFreqFE
from feature_extractors.factor_stop_words import FactorStopWordsFE

INDEX_ID        = 0
INDEX_TOPIC_ID  = 1
INDEX_TEXT      = 2
INDEX_AUTHOR_ID = 3

def texts_from_corpus(corpus):
    return [d[INDEX_TEXT] for d in corpus]


# Import data
con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
cur = con.cursor()
cur.execute("SELECT * FROM `small_article`")
results = cur.fetchall()

# Parameters:
num_words = 10

# Format Data
train_corpus = []
test_corpus = []
for row in results:
    id,author_id,topic_id,_,text,is_train = row
    if int(topic_id) is not 3:
        train_corpus.append((id,topic_id,text.decode('utf-8'),author_id))
    else:
        test_corpus.append((id,topic_id,text.decode('utf-8'),author_id))

# Create Features extractors
word_freq_fe = WordFreqFE(num_words=num_words)
stop_words_fe = FactorStopWordsFE()

# Create Classifiers
classifiers = [stop_words_fe]

train_features = np.zeros((len(train_corpus),0))
test_features = np.zeros((len(test_corpus),0))

# Extract features
for classifier in classifiers:
    train_features = np.concatenate((train_features, classifier.quantize_feature(texts_from_corpus(train_corpus))),axis=1)
    test_features = np.concatenate((test_features, classifier.quantize_feature(texts_from_corpus(test_corpus))),axis=1)

train_author_ids = [d[INDEX_AUTHOR_ID] for d in train_corpus]
test_author_ids = [d[INDEX_AUTHOR_ID] for d in test_corpus]

# Saving the objects:
with open('word_freq.pickle', 'w') as f:
    pickle.dump([train_features, test_features, word_freq_fe, train_author_ids,test_author_ids], f)
