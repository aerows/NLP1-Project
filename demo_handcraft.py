import numpy as np
import MySQLdb as mdb
from classification_models.randomForestCM import RandomForestCM
from datasets.mysql_dataset import MysqlDataset
from data.dataset_data import MysqlDatasetData
from feature_extractors.factor_stop_words import FactorStopWordsFE
from feature_extractors.ngram_freq import NGramFreq
from feature_extractors.words_per_sentence import WordsPerSentanceFE
from feature_extractors.word_freq import WordFreqFE
from feature_extractors.kmeans_ngram import KMeansNGram
import pickle
import collections
import itertools as it


def run_ngram(dataset,n,k,n_test,n_estimators, num_words):
    """
    :param dataset:
    :param n:               Used for classifier
    :param k:               Not used
    :param n_test:          Used for data folding
    :param n_estimators:    Used for classifier
    :param num_words:       Used for feature extractor
    :return:
    """
    result = ""
    ngram_args = dict(n_clusters=k,n_jobs=1,max_iter=200,n_init=5,verbose=True)
    pkl_filename = "pickles/ngramsdemo_%s_n-%d_%d_%s.pickle" % (dataset,n,num_words,str(ngram_args))
    try:
        pkl_file = open(pkl_filename, 'rb')
        dataset = pickle.load(pkl_file)
    except:
        result += "\nngrams arguments: " + str(ngram_args)
        # Import data
        features = [
            # KMeansNGram(k=k, n=n, kmeans_args=kmeans_args)
            # WordsPerSentanceFE(), # Not implemented properly yet!
            NGramFreq(n=n, num_words=num_words),
            # WordFreqFE(num_words=num_words),
            # FactorStopWordsFE()
        ]

        dataset = MysqlDatasetData(MysqlDataset(dataset), features)
        dataset.dataset = None
        pkl_file = open(pkl_filename, 'wb')
        pickle.dump(dataset,pkl_file)
        pkl_file.close()

    data_train,labels_train,data_test,labels_test = dataset.fold(n_test=n_test)

    # Todo: cross validation
    # Train model
    model = RandomForestCM(n_estimators=n_estimators)
    # model = AveragedPerceptronCM(max_iter=200)
    model.train_classifier(data_train,labels_train)

    # Test model
    q, pred_labels = model.test_classifier(data_test,labels_test)
    print q
    print pred_labels
    print labels_test
    result += "\n N-grams random forest with precision: %s \npredicted labels: %s \nactual labels: %s" % (q,str(pred_labels),str(labels_test))
    return result

def run_kmeans_ngram(dataset,n,k,n_test,n_estimators):
    result = ""
    kmeans_args = dict(n_clusters=k,n_jobs=1,max_iter=200,n_init=5,verbose=True)
    pkl_filename = "pickles/kmeansngramdemo_%s_n-%d_%s.pickle" % (dataset,n,str(kmeans_args))
    try:
        pkl_file = open(pkl_filename, 'rb')
        dataset = pickle.load(pkl_file)
    except:
        result += "\nkmeans arguments: " + str(kmeans_args)
        # Import data
        features = [
            KMeansNGram(k=k, n=n, kmeans_args=kmeans_args)
            # WordsPerSentanceFE(), # Not implemented properly yet!
            # NGramFreq(2,400)
        ]
        dataset = MysqlDatasetData(MysqlDataset(dataset),features)
        dataset.dataset = None
        pkl_file = open(pkl_filename, 'wb')
        pickle.dump(dataset,pkl_file)
        pkl_file.close()

    data_train,labels_train,data_test,labels_test = dataset.fold(n_test=n_test)

    # Todo: cross validation
    # Train model
    model = RandomForestCM(n_estimators=n_estimators)
    # model = AveragedPerceptronCM(max_iter=200)
    model.train_classifier(data_train,labels_train)

    # Test model
    q, pred_labels = model.test_classifier(data_test,labels_test)
    print q
    print pred_labels
    print labels_test
    result += "\n K-means random forest with precision: %s \npredicted labels: %s \nactual labels: %s" % (q,str(pred_labels),str(labels_test))
    return result

# experiment = run_kmeans_ngram # Function to call
experiment = run_ngram
parameter_sets = [
    #dict(dataset=["long_comments"],n=[3,5,7,9], k=[100],n_test=[1],n_estimators=[10000],num_words=[100,400,1000,2500])
    dict(dataset=["small_article"],n=[3,4,5,6,10,14], k=[100],n_test=[1],n_estimators=[10000],num_words=[100,400,1000,2500])
]

# Call the experiment for all the combinations of parameters and log the results
for parameters in parameter_sets:
    # Log the results
    log_file = open("log/results-%s-%s.log"%(experiment.func_name,str(parameters)),"w")
    for args in it.product(*parameters.values()):
        args_dict = {key: args[i] for i,key in enumerate(parameters.keys())}
        print "Running experiment: %s with parameters: %s" % (experiment.func_name, args_dict)
        log_file.write("\n\n--------\n\nRunning experiment: %s with parameters: %s" % (experiment.func_name, args_dict))
        log_file.flush()
        results = experiment(**args_dict)
        log_file.write(results)
        log_file.flush()
    log_file.close()