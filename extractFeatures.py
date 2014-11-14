import nltk
import numpy as np
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer


TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)

file_name = 'datasets/mensuckatfriendships.txt'
with open (file_name, "r") as file:
    data=file.read()
data = data.replace('<p>',"\n");
data = remove_tags(data)
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(data)


non_stop_tokens = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
# print np.bincount(tokens)
counts = Counter(non_stop_tokens)
print counts

counts = Counter(tokens)
print counts