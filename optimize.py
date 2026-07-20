from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def optimize_k(data:NDArray, max_k:int = 10, diagnostic_panel:bool = True) -> set[int]:
    """Finds the optimal set of k values using an ensemble of clustering metrics.

    Args:
        data (array-like of shape (n_samples, n_features)): 
            The input data matrix to cluster. Must be 2D, strictly numeric 
            (int/float), free of NaN/inf values, and ideally scaled using 
            StandardScaler or MinMaxScaler.
        max_k (int, optional): The maximum cluster count to test. Defaults to 10.
        diagnostic_panel (boolean, optional): Controls whether or not to output the metrics as a graph, defaults to true
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
            n_init=10
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

    normed_ks = (ks - ks[0]) / (ks[-1] - ks[0])
    normed_inertias = (inertias - inertias.min()) / (inertias.max() - inertias.min())
    points = np.column_stack([normed_ks, normed_inertias])
    p1, p2 = points[0], points[-1]
    line = p2 - p1
    distances = (line[0] * (p1 - points)[:, 1]) - (line[1] * (p1 - points)[:, 0]) / np.linalg.norm(line)
    distances = np.abs(distances)

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

    optimal_inertia_k = np.argmax(distances) + 2
    optimal_silhoutte_k = np.argmax(silhouette_scores) + 2
    optimal_c_k = np.argmax(calinskis) + 2
    optimal_d_k = np.argmin(davies) + 2
    
    votes = Counter([optimal_inertia_k, optimal_silhoutte_k, optimal_d_k, optimal_c_k])
    opt = set([int(optimal_inertia_k), int(optimal_silhoutte_k), int(optimal_c_k), int(optimal_d_k)])
    print(f'K-Finder has selected the set of k={opt} as valid k-values with the following votes')
    print(votes.most_common())
    return opt

def produce_dummy_set():
    X, y = make_blobs(
        n_samples=1000,
        centers=6
    )
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.savefig('centroids.png')
    return X, y

def main():
    data, labels = produce_dummy_set()
    optimize_k(data, 8, True)

if __name__ == '__main__':
    main()