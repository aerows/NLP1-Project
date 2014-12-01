import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import numpy as np
import itertools

def full_corpus_text(texts):
    full_text = ""
    for document in texts:
        full_text += document
    return full_text

def non_stop_word_count(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    non_stop_tokens = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    return Counter(non_stop_tokens)


def most_common_vocabulary(texts,num_words=None):
    full_text = full_corpus_text(texts)
    count_dict = non_stop_word_count(full_text)
    common_vocabulary = [i[0] for i in count_dict.most_common(num_words)]
    return common_vocabulary
    
def num_words_in_document(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    numwords = len(tokens)
    return numwords
    
def num_stop_words(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    stop_tokens = [w for w in tokens if w in nltk.corpus.stopwords.words('english')]
    return len(stop_tokens)
    
def num_sentences(text):
    tokenizer = RegexpTokenizer(r' ([A-Z][^\.!?]*[\.!?])')
    sentences = tokenizer.tokenize(text)
    return len(sentences)
    
def average_sentence_length(text):
    tokenizer = RegexpTokenizer(r' ([A-Z][^\.!?]*[\.!?])')
    sentences = tokenizer.tokenize(text)
    s = np.zeros(len(sentences))
    for inds, sentence in enumerate(sentences):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        s[inds] = len(tokens)
    return s, np.mean(s), np.std(s)
    
def n_gram_vocabulary(text, n=2 , num_words = None):
    tokenizer = RegexpTokenizer(r' ([A-Z][^\.!?]*[\.!?])')
    if isinstance(text,list):
        text = " ".join(text)
    sentences = tokenizer.tokenize(text)
    grams = ()
    for ind, sentence in enumerate(sentences):
        sentence = sentence.split()
        grams  = grams + tuple([tuple(sentence[i:i+n]) for i in xrange(len(sentence)-n)])
    return Counter(grams).most_common(num_words)
    
def num_n_grams_in_document(text, n=2):
    tokenizer = RegexpTokenizer(r' ([A-Z][^\.!?]*[\.!?])')
    sentences = tokenizer.tokenize(text)
    grams = ()
    for ind, sentence in enumerate(sentences):
        sentence = sentence.split()
        grams  = grams + tuple([tuple(sentence[i:i+n]) for i in xrange(len(sentence)-n)])
    return len(grams)
    
        