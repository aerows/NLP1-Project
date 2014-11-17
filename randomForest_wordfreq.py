import nltk
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import MySQLdb as mdb
from sklearn import ensemble
import pickle

# Getting back the objects:
with open('word_freq.pickle') as f:
    train_word_freq_matrix, test_word_freq_matrix, word_freq_fe, train_author_ids,test_author_ids = pickle.load(f)

randomforest = ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy')
randomforest.fit(train_word_freq_matrix,train_author_ids)

correct = 0
for index,data_point in enumerate(test_word_freq_matrix):
    prediction = randomforest.predict(data_point)
    print prediction, test_author_ids[index]
    if prediction == test_author_ids[index]:
        correct += 1

print correct, len(test_author_ids)