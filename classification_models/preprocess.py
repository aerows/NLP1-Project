import numpy as np


class OneHotEncoder(object):
    def __init__(self):
        self.code_words = None
        self.on_encoding = 1
        self.off_encoding = 0

    def encode(self, labels, on_encoding=1, off_encoding=0):
        """ Encode list of labels on 'One hot'-format

        Args:
            labels: a one dimensional vector of size n
            on_encoding: value for index belonging to encoded label
            off_encoding: value for index not belonging to encoded label
        Returns:
            one_hot: a one_hot matrix, with size n by k, where k is the number of unique labels
        Raises:
            AssertionError: if labels is not a one-dimensional matrix.

        """
        assert len(np.shape(labels)) == 1, ":labels should be a vector"

        self.code_words = np.array(np.unique(labels))                       # store original label codewords
        self.on_encoding, self.off_encoding = on_encoding, off_encoding     # store on- and off-encoding
        n, k = len(labels), len(self.code_words)                            # dimensions of one hot encoding
        one_hot = np.ones((n, k)) * off_encoding                            # init with off_encoding
        one_hot += [(label == self.code_words) *                            # replace with on_encoding
                    (on_encoding - off_encoding) for label in labels]       # where condition is true

        return one_hot                                                      # return encoding

    def decode(self, one_hot):
        """ Decodes one_hot matrix to list with original labels.

        Args:
            one_hot: a one_hot matrix, with size n by k, where k is the number of unique labels
        Returns:
            labels: a one dimensional vector of size n
        Raises:
            AssertionError: if one_hot is not a two-dimensional matrix.
            AssertionError: if one_hot does not have #columns equal to #unique labels
        """
        assert len(np.shape(one_hot)) == 2, ":one_hot should be a matrix"
        assert np.shape(one_hot)[1] == len(self.code_words), ":one_hot should have #columns equal to #unique labels"

        _, label_indexes = np.where(one_hot == self.on_encoding)            # get indexes of on_encoding elements
        labels = self.code_words[label_indexes]                             # get original labels
        return labels

    def decode_soft(self, soft_one_hot):
        """ Decodes soft_one_hot matrix to list with original labels with highest ranking.

        Args:
            soft_one_hot: a ranking matrix, with size n by k, where k is the number of unique labels
        Returns:
            labels: a one dimensional vector of size n
        Raises:
            AssertionError: if one_hot is not a two-dimensional matrix.
            AssertionError: if one_hot does not have #columns equal to #unique labels
        """
        assert len(np.shape(soft_one_hot)) == 2, ":one_hot should be a matrix"
        assert np.shape(soft_one_hot)[1] == len(self.code_words), ":one_hot should have #columns equal to #unique labels"

        label_indexes = np.argmax(soft_one_hot, axis=1)                     # get indexes of max_elements
        labels = self.code_words[label_indexes]                             # get original labels
        return labels