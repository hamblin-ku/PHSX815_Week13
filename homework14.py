# File:     minimize.py
# Author:   Kurt Hamblin
# Description:  Utitlize the Random Class to:
# Simulate dice rolls where the dice weights are sampled from a Rayleigh Distribution

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import my_params
custom_params = my_params.params()
matplotlib.rcParams.update(custom_params)

if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)
    X = X[:, ::-1] # flip axes for better plotting

    kmeans = KMeans(5, random_state=0)
    labels = kmeans.fit(X).predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
    
    
    plt.show()
