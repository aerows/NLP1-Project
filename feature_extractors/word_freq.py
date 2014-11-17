from feature_extractor import FeatureExtractor
from feature_lib.helper_functions import *
import numpy as np

class WordFreqFE(FeatureExtractor):
    def __init__(self,num_words,common_vocabulary=None):
        self.num_words = num_words
        self.common_vocabulary = common_vocabulary
        FeatureExtractor.__init__(self)

    def quantize_feature(self,texts):
        if self.common_vocabulary is None:
            self.common_vocabulary = most_common_vocabulary(texts,self.num_words)

        word_freq_matrix = np.zeros((len(texts),self.num_words))
        for index,document in enumerate(texts):
            count_dict = non_stop_word_count(document)
            # TODO: Normalize by number of words in documents
            word_freq_matrix[index,:] = [count_dict.get(key_word,0) for key_word in self.common_vocabulary]

        return word_freq_matrix