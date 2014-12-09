__author__ = 'Daniel'

from feature_extractor import FeatureExtractor
from sklearn import neighbors
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
import numpy as np
import feature_lib.helper_functions as hf


class SpectralClustering(FeatureExtractor):
    def __init__(self, k, similarity_function):
        self.vocabulary = None
        self.k = k
        self.sim_function = similarity_function
        FeatureExtractor.__init__(self)
        self.featuresVec

    def setupFeatureSpace:

    def similarity_matrix(self):
        assert len(np.shape(self.vocabulary)) == 1, "Vocabulary should be a vector"
        assert callable(self.sim_function), "sim_func should be a function"
        data = self.vocabulary
        n = len(data)                                                    # Length of data vector
        sim_matrix = np.zeros((n, n))                                    # Initialize matrix
        for i in range(n):                                               # For each row
            for j in range(n):                                           # For each column
                sim_matrix[i, j] = self.sim_function(data[i], data[j])   # Use sim_func to calculate the i,jth entry
        return sim_matrix                                                # Return sim_matrix

    def compute_featurespace(self, sim_matrix, k):
        w = k_nearest_neighbor_graph(sim_matrix, k)
        lap = laplacian(w, normed=False)           # Compute the unnormalized Laplacian L.
        _, u = np.linalg.eig(lap)                  # Compute the first k eigenvectors u1, . . . , uk of L.
        feature_space = u[:, 0:k]                  # Let U in Rnxk be the matrix containing the vectors u1 uk as columns
        return feature_space

    def k_nearest_neighbor_graph(simmatrix, k):
        nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(simmatrix)
        return nbrs.kneighbors_graph(simmatrix).toarray()

    def quantize_feature(self, texts, labels):
        n = len(self.vocabulary)
        d = len(texts)
        word_count = np.zeros((d, n))
        dict_list = [hf.non_stop_word_count(texts[i]) for i in range(d)]    # Count all word freqs
        for i in range(d):
            for j in range(n):
                word_count[i, j] = dict_list[i].get(self.vocabulary[j])     # Count occurrences of word j in dict i
            word_count[i, :] = word_count[i, :] / np.sum(word_count[i, :])  # Normalize the counts

        features = word_count.dot(self.feature_vectors)                     # Compute the features

        return features


# def unnormalized_spectral_clustering(simmatrix, k):
#     """ Input: Similarity matrix S in Rnxn, number k of clusters to construct.
#     :param simmatrix:
#     :param k:
#     :return:
#     """
#     # Construct a similarity graph by one of the ways described in Section 2. Let W be its weighted adjacency matrix.
#     w = k_nearest_neighbor_graph(simmatrix, k)
#     # Compute the unnormalized Laplacian L.
#     lap = laplacian(w, normed=False)
#     # Compute the first k eigenvectors u1, . . . , uk of L.
#     _, u = np.linalg.eig(lap)
#     # Let U in Rn x k be the matrix containing the vectors u1 , . . . , uk as columns.
#     U = u[:, 0:k]
#     # For i = 1,...,n, let yi in Rk be the vector corresponding to the i-th row of U.
#     Y = U
#     # Cluster the points (yi)i=1,...,n in Rk with the k-means algorithm into clusters C1,...,Ck.
#
#     kmeans_args = dict(n_clusters=k,n_jobs=1,max_iter=50,verbose=True,n_init=2)
#     k_means = KMeans(**kmeans_args)
#     # return: Clusters A1,...,Ak with Ai = {j| yj in Ci}.
#     ret = k_means.fit(Y)
#     # return ret



# k = 3
# n = 10
# simmatrix = np.random.randint(1, 7, (n, n))
#
# A = unnormalized_spectral_clustering(simmatrix, k)
# print A
