import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import numpy as np
import itertools
import re
import pickle


CHAR = re.compile(r'\w')


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
    
def text_to_vector(X):
    """"vectorizes a given string"""
    chars = CHAR.findall(X)
    return Counter(chars)


def cosine_sim(X,Y):
    """takes two strings and returns their cosine simularity"""
    vecX = text_to_vector(X.lower())
    vecY = text_to_vector(Y.lower())
    intersect = set(vecX.keys()) & set(vecY.keys())
    numerator = sum([vecX[i] * vecY[i] for i in intersect])
        
    sumX = sum([vecX[k]**2 for k in vecX.keys()])
    sumY = sum([vecY[k]**2 for k in vecY.keys()])
    denom = np.sqrt(sumX) * np.sqrt(sumY)
    
    if not denom:
        return 0.0
    else:
        return float(numerator)/denom
    
def n_sim(X,Y):
    """takes two strings, X and Y, and returns it's n-similarity score as
    defined by Kondrak in 2005"""
    
    return 0
    
def dice_coefficient(a,b):
    a = a.lower()
    b = b.lower()
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0
 
    """ use python list comprehension, preferred over list.append() """
    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]
 
    a_bigram_list.sort()
    b_bigram_list.sort()
 
    # assignments to save function calls
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1
 
    score = float(matches)/float(lena + lenb)
    return score


def vocabulary(texts):
    voc = set()
    for text in texts:
        voc.union(nltk.set(text))
    return voc


def pos_tag_hist(text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    tagged_words = nltk.pos_tag(words)
    tags = [x[-1] for x in tagged_words]
    counted = Counter(tags)
    tagset = get_tagset()
    hist = {pos_tag: counted[pos_tag] for pos_tag in tagset}
    return hist

def get_tagset():
    tagset = pickle.load(open("tagset.pickle", 'r'))
    return tagset