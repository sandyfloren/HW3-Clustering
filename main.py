import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)


def main():

    # create tight clusters
    clusters, labels = make_clusters(scale=0.3)
    plot_clusters(clusters, labels, filename="figures/tight_clusters.png")

    # create loose clusters
    clusters, labels = make_clusters(scale=2)
    plot_clusters(clusters, labels, filename="figures/loose_clusters.png")

    """
    uncomment this section once you are ready to visualize your kmeans + silhouette implementation
    """
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)

    pred = km.predict(clusters)


    km3 = KMeans(k=2)
    km3.fit(np.array([[0,1,2,3,4,5,6],
                     [2,3,4,5,6,7,8],
                     [0.1,1.1,2.1,3.1,4.1,5.1,6.1],
                     [2.3,3.3,4.3,5.3,6.3,7.3,8.3]]))

    pred3 = km3.predict(np.array([[0,1,2,3,4,5,6],
                     [2,3,4,5,6,7,8]
                     ]))


    km4 = KMeans(k=4)
    km4.fit(np.random.rand(5))


    data = np.random.rand(100)
    pred4 = km4.predict(data)


    #print(km.get_error())


    from sklearn.metrics import silhouette_samples
    from sklearn.cluster import KMeans as skmeans
   #scores = silhouette_score(clusters, np.argmax(pred, axis=1))
    scores=np.random.randint(0,4, 500)
    scores = Silhouette().score(clusters, pred)
    #print(scores)
    plot_multipanel(clusters, labels, pred, scores)

    km2=skmeans(4, init='random', max_iter=100, tol=1e-6, n_init=10)
    km2.fit(clusters)
    pred2 = km2.predict(clusters)
    scores2 = silhouette_samples(clusters, pred2)
    plot_multipanel(clusters, labels, pred2, scores2)


    rng = np.random.default_rng(62)
    pred_data = rng.uniform(size=1000)

    for k in range(1, 10):
        print("Testing Silhouette for k ==", k)
        labels = rng.integers(0, k+1, size=1000)
        print(labels)
        sil = Silhouette()
        sil_score = sil.score(pred_data, labels)
        print(sil_score)
    

if __name__ == "__main__":
    main()
