class Corpus:
    """A class to represent corpus and corpus information"""
    def __init__(self, author, path):
            self.author = author
            self.path = path
                
    def getPath(self):
        return self.path
        
    def getAuthor(self):
        return self.author
        
    def data(self):
        with open(self.path, 'r') as content_file:
            data = content_file.read()
        return data