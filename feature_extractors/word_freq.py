from feature_extractor import FeatureExtractor
from feature_lib.helper_functions import *
import numpy as np

class WordFreqFE(FeatureExtractor):
    def __init__(self,num_words,common_vocabulary=None):
        self.num_words = num_words
        self.common_vocabulary = common_vocabulary
        FeatureExtractor.__init__(self)

    def quantize_feature(self, texts, labels):
        if self.common_vocabulary is None:
            self.common_vocabulary = most_common_vocabulary(texts,self.num_words)

        word_freq_matrix = np.zeros((len(texts),self.num_words))
        for index,text in enumerate(texts):
            count_dict = non_stop_word_count(text)
            # TODO: Normalize by number of words in documents
            word_freqs = np.double(np.array([count_dict.get(key_word,0) for key_word in self.common_vocabulary]))
            # Normalize it
            word_freqs = self._normalize_freq(word_freqs,text)
            word_freq_matrix[index,:] = word_freqs

        return word_freq_matrix

    def _normalize_freq(self, word_freqs,text):
        try:
            return np.divide(word_freqs,num_words_in_document(text))
        except:
            print "whut"