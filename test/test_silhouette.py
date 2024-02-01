# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_samples

def test_silhouette():
    rng = np.random.default_rng(62)
    pred_data = rng.uniform(size=(100,5))

    for k in range(2, 10):
        print("Testing Silhouette for k ==", k)
        labels = rng.integers(0, k, size=100)

        sil = Silhouette()
        sil_score = sil.score(pred_data, labels)

        sil_sklearn = silhouette_samples(pred_data, labels)
        assert np.allclose(sil_score, sil_sklearn, atol=1e-10), "Silhouette score does not match sklearn's silhouette_samples."

    
