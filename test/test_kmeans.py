# Write your k-means unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans

def test_kmeans():
    rng = np.random.default_rng(62)
    fit_data = rng.uniform(size=(500,5))
    pred_data = rng.uniform(size=(100,5))
    for k in range(10):
        print("Testing KMeans for k ==", k)
        if k == 0:
            with pytest.raises(ValueError, match=r'k must be at least 1'):
                km = KMeans(k=k)
            continue
        km = KMeans(k=k, tol=1e-6, max_iter=100)
        km.fit(fit_data)

        assert km.get_error() >= 0, 'Error must be nonnegative.'
        assert len(km.get_centroids()) == k, 'Number of centroids must be k.'

        pred = km.predict(pred_data)
        
        assert len(pred) == pred.shape[0], 'Predicted labels must be length of number of data points'


