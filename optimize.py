from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def optimize_k(k_range, data):
    """
    Finds the optimal combination between model inertia and silhoutte score to pick a best k
    inputs:
        k_range: The highest k value to test starting from 2
        data: The data you are clustering on, the original feature matrix
    outputs:
        k: a single, most optimal k value to use for clustering 
    """
    
    silhouette_scores = []
    inertias = []
    ks = np.arange(2, k_range + 1)
    
    for k in range(2, k_range + 1):
        print(f'Attempting to cluster using k={k}')
        model = KMeans(
            n_clusters=k,
        )
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
        inertias.append(model.inertia_)
        print(f'Value k={k} yielded a final inertia of: {model.inertia_} and a silhouette score of {score}')
        

    silhouette_scores = np.array(silhouette_scores)
    inertias = np.array(inertias)
    velocities = np.diff(inertias)
    accelerations = np.diff(velocities)

    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(ks, silhouette_scores)
    ax[0][0].set_title('Model Silhoutte Scores')
    ax[0][1].plot(ks, inertias)
    ax[0][1].set_title('Model Inertia Tracking')
    ax[1][0].plot(list(range(len(velocities))), velocities)
    ax[1][0].set_title('Model Inertia First Derivative')
    ax[1][1].plot(list(range(len(accelerations))), accelerations)
    ax[1][1].set_title('Model Inertia Second Derivative')

    optimal_inertia_k = np.argmax(accelerations) + 3
    optimal_silhoutte_k = np.argmax(silhouette_scores) + 2

    if optimal_inertia_k == optimal_silhoutte_k:
        print(f'Optimal k value selected: {optimal_silhoutte_k}')
    else:
        print('There was a disagreement :(')
        
    plt.show()

def produce_dummy_set():
    X, y = make_blobs(
        n_samples=1000
    )
    return X, y

def main():
    data, labels = produce_dummy_set()
    optimize_k(8, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='KOpt - An K-Value Optimizer',
        description='This script is for finding the best possible k-value for the KMeans algorithm when you are unsure of the number of clusters'
    )
    parser.add_argument('-k', help='Range of k values to use as the search space')
    main()