from feature_extractor import FeatureExtractor
from feature_lib.helper_functions import *
import numpy as np

class FactorStopWordsFE(FeatureExtractor):
    def __init__(self):
        FeatureExtractor.__init__(self)

    def quantize_feature(self, texts, labels):
        feature_matrix = np.zeros((len(texts),1))
        for index,text in enumerate(texts):
             feature_matrix[index,:] = float(num_stop_words(text)) / num_words_in_document(text)
        return feature_matrix
