import numpy as np
import MySQLdb as mdb
from classification_models.randomForestCM import RandomForestCM
from classification_models.averaged_perceptronCM import *
from datasets.mysql_dataset import MysqlDataset
from data.dataset_data import MysqlDatasetData
from feature_extractors.factor_stop_words import FactorStopWordsFE
from feature_extractors.ngram_freq import NGramFreq
from feature_extractors.words_per_sentence import WordsPerSentanceFE
from feature_extractors.word_freq import WordFreqFE
from feature_extractors.kmeans_ngram import KMeansNGram
import pickle
import collections
n=5
try:
    pkl_file = open("kmeansngramdemo_hn_%d.pickle" % n, 'rb')
    dataset = pickle.load(pkl_file)
except:
    k = 400
    kmeans_args = dict(n_clusters=k,n_jobs=1,max_iter=200,n_init=10,init="random",verbose=True)
    # Import data
    features = [
        KMeansNGram(k=k, n=n, kmeans_args=kmeans_args)
        # WordsPerSentanceFE(), # Not implemented properly yet!
        # NGramFreq(2,400)
    ]
    dataset = MysqlDatasetData(MysqlDataset("long_comments"),features)
    dataset.dataset = None
    pkl_file = open("kmeansngramdemo_hn_%d.pickle" % n, 'wb')
    pickle.dump(dataset,pkl_file)

data_train,labels_train,data_test,labels_test = dataset.fold()
print collections.Counter(labels_train)
# Todo: cross validation
# Train model
model = RandomForestCM(n_estimators=1000)
# model = AveragedPerceptronCM(max_iter=200)
model.train_classifier(data_train,labels_train)

# Test model
q, pred_labels = model.test_classifier(data_test,labels_test)
print q
print pred_labels
print labels_test