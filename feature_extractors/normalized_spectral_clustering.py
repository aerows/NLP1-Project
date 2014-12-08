__author__ = 'Daniel'

from sklearn import neighbors
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
import numpy as np


def unnormalized_spectral_clustering(simmatrix, k):
    """ Input: Similarity matrix S in Rnxn, number k of clusters to construct.
    :param simmatrix:
    :param k:
    :return:
    """
    # Construct a similarity graph by one of the ways described in Section 2. Let W be its weighted adjacency matrix.
    w = k_nearest_neighbor_graph(simmatrix, k)
    # Compute the unnormalized Laplacian L.
    lap = laplacian(w, normed=False)
    # Compute the first k eigenvectors u1, . . . , uk of L.
    _, u = np.linalg.eig(lap)
    # Let U in Rn x k be the matrix containing the vectors u1 , . . . , uk as columns.
    U = u[:, 0:k]
    # For i = 1,...,n, let yi in Rk be the vector corresponding to the i-th row of U.
    Y = U
    # Cluster the points (yi)i=1,...,n in Rk with the k-means algorithm into clusters C1,...,Ck.

    kmeans_args = dict(n_clusters=k,n_jobs=1,max_iter=50,verbose=True,n_init=2)
    k_means = KMeans(**kmeans_args)
    # return: Clusters A1,...,Ak with Ai = {j| yj in Ci}.
    ret = k_means.fit_transform(Y)
    return ret


def k_nearest_neighbor_graph(simmatrix, k):
    nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(simmatrix)
    return nbrs.kneighbors_graph(simmatrix).toarray()


# k = 3
# n = 10
# simmatrix = np.random.randint(1, 7, (n, n))
#
# A = unnormalized_spectral_clustering(simmatrix, k)
# print A
