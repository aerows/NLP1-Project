import numpy as np


class OneHotEncoder(object):
    def __init__(self):
        self.code_words = None
        self.on_encoding = 1
        self.off_encoding = 0

    def encode(self, labels, on_encoding=1, off_encoding=0):
        self.code_words = np.array(np.unique(labels))                       # store original label codewords
        self.on_encoding, self.off_encoding = on_encoding, off_encoding     # store on- and off-encoding
        n, k = len(labels), len(self.code_words)                            # dimensions of one hot encoding
        one_hot = np.ones((n, k)) * off_encoding                            # init with off_encoding
        one_hot += [(label == self.code_words) *                            # replace with on_encoding
                    (on_encoding - off_encoding) for label in labels]       # where condition is true

        return one_hot                                                      # return encoding

    def decode(self, one_hot):
        _, label_indexes = np.where(one_hot == self.on_encoding)            # get indexes of on_encoding elements
        labels = self.code_words[label_indexes]                             # get original labels
        return labels

    def decode_soft(self, soft_one_hot):
        label_indexes = np.argmax(soft_one_hot, axis=1)                     # get indexes of max_elements
        labels = self.code_words[label_indexes]                             # get original labels
        return labels