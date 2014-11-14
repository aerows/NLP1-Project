from ijson import parse
import MySQLdb as mdb
import HTMLParser
def flush_item():
    if doc_offset > last_doc_offset:
        try:
            cur.execute("""INSERT INTO `comments` (`id`, `hn_object_id`,`doc_offset`, `created_at`, `story_id`, `parent_id`, `author`, `points`, `comment_text`)
                VALUES (NULL, %s, %s, STR_TO_DATE(%s, '%%Y-%%m-%%dT%%H:%%i:%%sZ' ) , %s, %s, %s, %s, %s)""",
                (hn_object_id, doc_offset, created_at, story_id, parent_id, author, points, html_parser.unescape(comment_text)))
        except mdb.OperationalError, e: print repr(e)
        con.commit()
f =  open('/Users/bas/Documents/datasets/HNCommentsAll.json','r')
parser = parse(f)
con = mdb.connect('127.0.0.1', 'root', '', 'nlpcorpus')
cur = con.cursor()
cur.execute("SELECT max(`doc_offset`) FROM `comments`")
html_parser = HTMLParser.HTMLParser()
last_doc_offset = cur.fetchall()[0][0]
skipping = True
if last_doc_offset is None:
    last_doc_offset = 0
    skipping = False
doc_offset = -1

for prefix, event, value in parser:
    # print "p",prefix
    # print "e",event
    # print "v",value
    if prefix == 'item.hits.item.created_at':
        doc_offset = doc_offset + 1
        if doc_offset % 10000 == 0:
            print doc_offset
        if doc_offset >= last_doc_offset:
            flush_item()
            skipping = False
            created_at = value
        # while prefix is not 'item.hits.item.comment_text'
    if not skipping:
        if prefix == 'item.hits.item.comment_text':
            comment_text = value
        if prefix == 'item.hits.item.story_id':
            story_id = value
        if prefix == 'item.hits.item.parent_id':
            parent_id = value
        if prefix == 'item.hits.item.author':
            author = value
        if prefix == 'item.hits.item.points':
            points = value
        if prefix == 'item.hits.item.objectID':
            hn_object_id = value
        if doc_offset > last_doc_offset + 100000:
            break 

