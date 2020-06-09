import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.cm as cm

X = pd.read_csv("DataSet.csv")
X = X.drop(["No"], axis=1)

n_clusters = 3
random_state = np.random.RandomState(0)
km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
km.cluster_centers_

for i in range(30):
    if(km.labels_[i] == 0):
        plt.plot(X['x'][i], X['y'][i], 'bo')
    if(km.labels_[i] == 1):
        plt.plot(X['x'][i], X['y'][i], 'ro')
    if(km.labels_[i] == 2):
        plt.plot(X['x'][i], X['y'][i], 'go')

for i in range(3):
    plt.plot(km.cluster_centers_[i][0], km.cluster_centers_[i][1], 'ko')

plt.show()
