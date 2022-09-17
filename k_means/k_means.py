import numpy as np 
import pandas as pd
from random import uniform
import random
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, n_clusters=2):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.n_clusters = n_clusters
        self.centroids = np.array([])
        self.clusters = np.array([], dtype=int)
        self.points = np.array([])
        #random.seed(10)

    def fit(self, x, better_centroids=False):
        """
        Estimates parameters for the classifier

        Args:
            x (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            better_centroids: Decides whether to use k-means++ or not
        Returns an array of centroids
        """
        self.points = np.array(x)

        # standard k-means of totally random centroids
        if not better_centroids:
            self.centroids = np.array([[uniform(0, 1), uniform(0, 1)] for _ in range(self.n_clusters)])
            return self.centroids

        # K-means++
        centroids = []
        first_centroid = np.array([uniform(0, 1), uniform(0, 1)])
        centroids.append(first_centroid)

        # For each cluster, select the point furthest from any centroid
        for _ in range(self.n_clusters-1):
            dists = []
            for point in self.points:
                dist = euclidean_distance(point, centroids)
                dists.append(min(dist))
            furthest_point = np.argmax(dists)
            centroids.append(self.points[furthest_point])

        self.centroids = np.array(centroids)

        return self.centroids

    def predict(self, x):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            x (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        prev_centroids = []
        a = np.array(x)

        while not np.array_equal(self.centroids, prev_centroids):
            self.clusters = np.array([], dtype=int)
            prev_centroids = self.centroids
            for point in a:
                ed = euclidean_distance(point, self.centroids)
                self.clusters = np.append(self.clusters, np.argmin(ed))
            self.centroids = self.get_centroids()
            for (i, c) in enumerate(self.centroids):
                if np.isnan(c).any():
                    self.centroids[i] = prev_centroids[i]
        return self.clusters

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        n_dims = self.points.shape[1]
        centroids = [[1] * n_dims for _ in range(self.n_clusters)]

        for i in range(self.n_clusters):
            for dim in range(n_dims):
                col = self.points[:,dim]
                m = np.mean(col[np.where(self.clusters == i)])
                centroids[i][dim] = m
        return np.array(centroids)



# --- Some utility functions 


def euclidean_distortion(x, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        x (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    x, z = np.asarray(x), np.asarray(z)
    assert len(x.shape) == 2
    assert len(z.shape) == 1
    assert x.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        xc = x[z == c]
        mu = xc.mean(axis=0)
        distortion += ((xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(abs(x-y), ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(x, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        x (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    x, z = np.asarray(x), np.asarray(z)
    assert len(x.shape) == 2
    assert len(z.shape) == 1
    assert x.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(x), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(x[in_cluster_a], x[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(x)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
