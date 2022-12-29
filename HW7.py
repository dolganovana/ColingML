from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

X, _ = make_blobs(n_samples=1000,
                  centers=2,
                  n_features=2,
                  random_state=42)

plt.figure(figsize=(9,6))
scatter = plt.scatter(X[:,0], X[:,1])

from sklearn.metrics import pairwise_distances_argmin

class KMeans:
    
    def __init__(self, k, centroids=None):
        self.k = k
        self.centroids = centroids or None
    
    def fit(self, x):
        n_features = x.shape[1]

        self.centroids =  np.random.rand(self.k, n_features)

        done = False
        prev_centroids = None

        while not done:
            labels = self.predict(x)
            prev_centroids = self.centroids

            self.centroids = np.array([X[labels == i].mean(0)
                                    for i in range(self.k)])
            if np.allclose(self.centroids, prev_centroids):
                done=True
            



    def predict(self, x):
        assert len(x.shape) == 2 # it's a batch, not a vector
        y = pairwise_distances_argmin(x, self.centroids)
        return y
    
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7, random_state=0).fit(X)
model = KMeans(k=2)
model.fit(X)

y = model.predict(X)

plt.figure(figsize=(9,6))
scatter = plt.scatter(X[:,0], X[:,1], c=y)
plt.scatter(model.centroids[:,0], model.centroids[:,1], c='red', label='centroids')
plt.legend();

np.random.seed(42)
X, _ = make_blobs(n_samples=1000,
                  centers=10,
                  n_features=2,
                  cluster_std=np.random.uniform(low=0.5, high=2, size=(10)),
                  random_state=42)
plt.figure(figsize=(9,6))
scatter = plt.scatter(X[:,0], X[:,1])

import sklearn
from sklearn.cluster import KMeans
hypo_clusters = range(2, 20)
clusters_scores = {}
for n_clusters in hypo_clusters:
    clust = KMeans(n_clusters=n_clusters, random_state=10)
    labels = clust.fit_predict(X)

    silhouette_sc = sklearn.metrics.silhouette_score(X, labels, metric='euclidean')
    clusters_scores[n_clusters]=silhouette_sc
print(clusters_scores)
best_k_num = max(clusters_scores, key=clusters_scores.get)
score_for_best_k_num = max(clusters_scores.values())
print(f'Best number of clusters is {best_k_num}, score:{score_for_best_k_num}')

from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.datasets import load_nfl

model = KMeans(best_k_num, random_state=42)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

visualizer.fit(X)        
visualizer.show()

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons

import matplotlib.pyplot as plt
import numpy as np
X, _ = make_moons(n_samples=100)
plt.figure(figsize=(9,6))
scatter = plt.scatter(X[:,0], X[:,1])

db = DBSCAN(eps=0.1, min_samples=2).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
    
        col = [0, 1]

    class_member_mask = labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=1,
    )

plt.title(f'number of clusters: {n_clusters_}')
plt.show()

np.random.seed(42)
X, _ = make_blobs(n_samples=1000,
                  centers=15,
                  n_features=2,
                  cluster_std=np.random.uniform(low=0.5, high=1.2, size=(15)),
                  random_state=42)
plt.figure(figsize=(9,6))
scatter = plt.scatter(X[:,0], X[:,1])

import pandas as pd
df=pd.DataFrame(X,_)
nn = NearestNeighbors(n_neighbors=11)
neighbors = nn.fit(df)

dist, ind = neighbors.kneighbors(df)
dist = np.sort(dist[:,10], axis=0)
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

i = np.arange(len(distances))
knee = KneeLocator(i, dist, S=1, curve='convex', direction='increasing', interp_method='polynomial')
fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print(f'Optimal eps value is {dist[knee.knee]}')

bscan_cluster = DBSCAN(eps=dist[knee.knee], min_samples=8)
dbscan_cluster.fit(X)


plt.scatter(X[:, 0], 
X[:, 1], 
c=dbscan_cluster.labels_, 
label=_)


labels=dbscan_cluster.labels_
n_clus=len(set(labels))-(1 if -1 in labels else 0)
print(f'Num of clusters: {n_clus}')


n_noise = list(dbscan_cluster.labels_).count(-1)
print(f'Noise points: {n_noise}')
