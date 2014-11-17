import nltk
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import MySQLdb as mdb
from sklearn import ensemble
import pickle

INDEX_ID        = 0
INDEX_TOPIC_ID  = 1
INDEX_TEXT      = 2
INDEX_AUTHOR_ID = 3


def argsort(array):
    return sorted(range(len(array)), key=lambda k: array[k])

def full_corpus_text(corpus):
    full_text = ""
    for document in corpus:
        _,_,text,_ = document
        full_text += text
    return full_text

def non_stop_word_count(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    non_stop_tokens = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    return Counter(non_stop_tokens)


def most_common_vocabulary(corpus,num_words=None):
    full_text = full_corpus_text(corpus)
    count_dict = non_stop_word_count(full_text)
    common_vocabulary = [i[0] for i in count_dict.most_common(num_words)]
    return common_vocabulary

def word_freq(corpus, num_words, common_vocabulary=None):
    # Assumption: we have at least num_words unique words in the full_text
    if common_vocabulary is None:
        common_vocabulary = most_common_vocabulary(corpus,num_words)

    word_freq_matrix = np.zeros((len(corpus),num_words))
    for index,document in enumerate(corpus):
        count_dict = non_stop_word_count(document[INDEX_TEXT])
        # TODO: Normalize by number of words in documents
        word_freq_matrix[index,:] = [count_dict.get(key_word,0) for key_word in common_vocabulary]

    return word_freq_matrix, common_vocabulary


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

train_word_freq_matrix, common_vocabulary = word_freq(train_corpus, num_words)
test_word_freq_matrix, _ = word_freq(test_corpus,num_words,common_vocabulary)

train_author_ids = [d[INDEX_AUTHOR_ID] for d in train_corpus]
test_author_ids = [d[INDEX_AUTHOR_ID] for d in test_corpus]
# Saving the objects:
with open('word_freq.pickle', 'w') as f:
    pickle.dump([train_word_freq_matrix, test_word_freq_matrix, common_vocabulary, train_author_ids,test_author_ids], f)







