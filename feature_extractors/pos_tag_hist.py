from feature_extractor import FeatureExtractor
from feature_lib.helper_functions import *
import numpy as np

class PosTagHistFE(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)

    def quantize_feature(self, texts, labels):

        hist_freq_matrix = np.zeros((len(texts),len(get_tagset())))
        for index,text in enumerate(texts):
            hist = self._normalize_freq(pos_tag_hist(text).values(),text)
            hist_freq_matrix[index,:] = hist

        return hist_freq_matrix

    def _normalize_freq(self, word_freqs,text):
        try:
            return np.divide(word_freqs, float(num_words_in_document(text)))
        except:
            print "whut"