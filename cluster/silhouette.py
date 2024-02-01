import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        sil = np.zeros(shape=X.shape[0])
        labels = np.unique(y)

        # build dictionary of data per cluster, indexed by label
        clusters = {}
        for lab in labels:
            clusters[lab] = X[np.argwhere(y == lab)]

        # iterate through observations and labels
        sil_index = 0
        for i, C_I in zip(X, y):

            # Check if cluster size is <= 1
            if len(clusters[C_I]) <= 1:
                sil[sil_index] = 0
                sil_index += 1
                continue

            # compute intra-cluster distance
            dist_sum = 0
            for j in clusters[C_I]:
                if np.array_equal(i, j):
                    continue
                else:
                    dist_sum += self._dist(i, j)
            
            a = (1 / (len(clusters[C_I]) - 1)) * dist_sum

            # compute inter-cluster distance
            b = np.inf
            for C_J in labels:
                if C_J == C_I: 
                    continue
                else:
                    dist_sum = 0
                    for j in clusters[C_J]:
                        dist_sum += self._dist(i, j)
                    new_dist = (1 / len(clusters[C_J])) * dist_sum
                    if new_dist < b:
                        b = new_dist
                        
            # Handle case of only one cluster
            if b == np.inf:
                b = 0

            # compute silhouette score
            sil[sil_index] = (b - a) / np.max([a, b])
            sil_index += 1

        return sil


    def _dist(self, x, y):
        """
        Compute the distance between two observations.
        """

        return np.sqrt(np.sum((x - y)**2))

