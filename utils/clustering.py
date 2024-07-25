from sklearn.cluster import KMeans
import pandas as pd

def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    data['cluster'] = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    return data, centroids
