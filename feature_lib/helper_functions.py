import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer

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