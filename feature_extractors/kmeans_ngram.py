from feature_extractor import FeatureExtractor
from feature_lib.helper_functions import *
import numpy as np
from sklearn.cluster import KMeans

class KMeansNGram(FeatureExtractor):
    def __init__(self,texts=None,n=16,step_size=1,k=100,kmeans_args = None):
        self.n = n
        self.step_size = step_size
        self.k = k
        self.kmeans=None
        self.kmeans_args = kmeans_args
        FeatureExtractor.__init__(self)

    def initialise_kmeans(self,texts,labels):
        num_texts = 1
        unique_labels = set(labels)

        vocab_texts = []
        for label in unique_labels:
            indexes = [i for i,x in enumerate(labels) if x == label][0:num_texts]
            for i in indexes:
                vocab_texts.append(texts[i])

        if self.kmeans == None:
            combined_patch_matrix,_ = self.extract_patches(vocab_texts)
            self.kmeans = self.compute_kmeans(combined_patch_matrix)

    def normalize_hist(self, hist):
        return np.divide(hist, np.sum(hist).astype(float))

    def quantize_feature(self, texts, labels):
        self.initialise_kmeans(texts,labels)

        texts_centroids, texts_one_hot = self.compute_centroids(texts)
        texts_hist = []
        for assignments in texts_one_hot:
            print '.',
            # Sum up the assignments (which are one-hot) to get a histogram over the clusters
            hist = np.sum(assignments,0)

            assert(hist.shape == (self.k,))

            # Normalize the histogram and append to list
            texts_hist.append(self.normalize_hist(hist))
        texts_hist = np.array(texts_hist)
        assert texts_hist.shape == (len(texts),self.k), "Should be n by k"
        return texts_hist

    def compute_centroids(self, texts):
        _,text_patches = self.extract_patches(texts)

        texts_centroids = []
        texts_one_hot = []
        for patches in text_patches:
            distances = self.kmeans.transform(patches)

            # Compute the most likely cluster based on cluster distances (argmin)
            centroids = []
            for i in range(0,len(distances)):
                centroids.append(np.argmin(distances[i,:]))
            texts_centroids.append(centroids)

            # Convert distances to one-hot notation
            for i in range(0,len(distances)):
                minimum = np.min(distances[i,:])
                distances[i,:] = [dst == minimum for dst in distances[i,:]]
            texts_one_hot.append(distances)


        return texts_centroids, texts_one_hot
    def compute_kmeans(self,combined_patch_matrix):
        # TODO: Look at initialisation options
        print "Computing k_means..."
        if self.kmeans_args == None:
            self.kmeans_args = dict(n_clusters=self.k,n_jobs=1,max_iter=50,verbose=True,n_init=2)
        kmeans = KMeans(**self.kmeans_args)
        kmeans.fit(combined_patch_matrix)

        return kmeans

    def extract_patches(self,texts):
        """
        Create a matrix of all patches in all texts
        :param texts: list of texts
        :return: combined_patch_matrix: a matrix of all patches
        """
        n = self.n
        step_size = self.step_size
        assert isinstance(texts[0], unicode), "Text must be unicode"


        # Convert each text(string) to an array of unicode ints
        char_texts = []
        for text in texts:
            char_texts.append([ord(c) for c in text])

        texts_patches = []

        # Take n-grams of text
        for text in char_texts:
            amount_of_patches = (len(text)-n)/step_size

            patch_matrix = np.zeros((amount_of_patches,n)) # patch per row.
            for i in range(0,len(text)-n,step_size):
                patch_matrix[i,:] = text[i:i+n]

            texts_patches.append(patch_matrix)

        # Concatenate the matrices vertically
        combined_patch_matrix = np.concatenate(texts_patches,0)

        return combined_patch_matrix,texts_patches