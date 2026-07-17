from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def optimize_k(max_k:int, data:NDArray, diagnostic_panel:bool):
    """
    Finds the optimal combination between model inertia and silhoutte score to pick a best k
    inputs:
        k_range: The highest k value to test starting from 2
        data: The data you are clustering on, the original feature matrix
        diagnostic_panel: If true, this method will save to the working directory a plot of model metrics
    outputs:
        k: a single, most optimal k value to use for clustering 
    """
    
    if max_k <= 2:
        raise ValueError("Max k value should be above two")

    silhouette_scores = []
    inertias = []
    davies = []
    calinskis = []
    ks = np.arange(2, max_k + 1)
    
    for k in range(2, max_k + 1):
        logger.debug(f'Attempting to cluster using k={k}')
        model = KMeans(
            n_clusters=k,
        )
        labels = model.fit_predict(data)
        s_score = silhouette_score(data, labels)
        d_score = davies_bouldin_score(data, labels)
        c_score = calinski_harabasz_score(data, labels)
        silhouette_scores.append(s_score)
        davies.append(d_score)
        calinskis.append(c_score)
        inertias.append(model.inertia_)

    silhouette_scores = np.array(silhouette_scores)
    davies = np.array(davies)
    calinskis = np.array(calinskis)
    inertias = np.array(inertias)
    velocities = np.diff(inertias)
    accelerations = np.diff(velocities)

    if diagnostic_panel:
        fig, ax = plt.subplots(2, 2, figsize=(12,8))
        ax[0][0].plot(ks, silhouette_scores)
        ax[0][0].set_title('Model Silhoutte Scores')
        ax[0][1].plot(ks, inertias)
        ax[0][1].set_title('Model Inertia Tracking')
        ax[1][0].plot(ks, davies)
        ax[1][0].set_title('Davies-Bouldin Score')
        ax[1][1].plot(ks, calinskis)
        ax[1][1].set_title('Calinski-Harabasz Score')

        plt.savefig('kfinder_metrics.png')

    optimal_inertia_k = np.argmax(accelerations) + 3
    optimal_silhoutte_k = np.argmax(silhouette_scores) + 2
    optimal_c_k = np.argmax(calinskis) + 2
    optimal_d_k = np.argmin(davies) + 2
    
    opt = Counter([optimal_inertia_k, optimal_silhoutte_k, optimal_c_k, optimal_d_k]).most_common()
    print(f'K-Finder has selected the set of k={opt} as valid k-values. Output and inspect the graph for finer details')
    return opt

def produce_dummy_set():
    X, y = make_blobs(
        n_samples=1000,
        centers=4
    )
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.savefig('centroids.png')
    return X, y

def main():
    data, labels = produce_dummy_set()
    optimize_k(8, data, True)

if __name__ == '__main__':
    main()