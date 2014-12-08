import MySQLdb as mdb
import HTMLParser
import re
__author__ = 'bas'
con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
cur = con.cursor()
cur.execute("SELECT id,comment_text FROM long_comments")
result = cur.fetchall()
html_parser = HTMLParser.HTMLParser()


TAG_RE = re.compile(r'<[^>]+>')

A_RE = re.compile(r'<a[^>]+>[^<]*</a>')

QUOTE_RE = re.compile(r'^>.*$',re.MULTILINE)
for row in result:
    id = int(row[0])
    text = row[1]
    text = text.replace("<p>", "\n")
    text = A_RE.sub('',text)
    text = TAG_RE.sub('',text)
    text = QUOTE_RE.sub('',text)

    cur.execute("UPDATE long_comments SET comment_text=%s WHERE id = %s",(text,id))
con.commit()

cur.execute("SELECT distinct(author) FROM long_comments")
result = cur.fetchall()

for author_id, row in enumerate(result):
    author = row[0]
    cur.execute("UPDATE long_comments SET author_id = %s WHERE author = %s",(author_id, author))
con.commit()
