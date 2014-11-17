import numpy as np
import re
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import codecs
import os
import io
import MySQLdb as mdb
dataset_location = 'datasets/small_article_dataset/'

def add_text(author_id, topic_id, author, text):
    try:
        cur.execute("""INSERT INTO `small_article` (`id`, `author_id`, `topic_id`, `author`, `text`, `is_train`) VALUES (NULL, %s, %s, %s, %s, 1);
""", (author_id, topic_id, author, text))
    except mdb.OperationalError, e: print repr(e)
    con.commit()

con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
cur = con.cursor()
for dirname, dirnames, filenames in os.walk(dataset_location):
    for filename in filenames:
        print filename
        with io.open (os.path.join(dirname, filename), "r",encoding="iso-8859-1") as file:
            text=file.read()
        author_id = re.findall(r'\d+',dirname)[0]
        topic_id = re.findall(r'\d+',filename)[-1]
        author = "Author" + author_id
        add_text(author_id, topic_id, author, text)
# with codecs.open (file_name, "r", "utf-8") as file:
#     data=file.read()
# data = data.replace('<p>',"\n");