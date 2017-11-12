# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import math

# Importing the dataset
dataset = pd.read_csv('TED_Talks_by_ID_plus.csv')
dataset.drop(dataset.columns[[0, 1, 2, 3, 4, 5, 8, 10, 12, 13, 14]], axis=1, inplace=True)
x = dataset.iloc[:, 2:].values
a = x[:, :1]


def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

for i in range(len(x)):
    c = str(a[i][0])
    a[i][0] = get_sec(c)
    a[i][0] = int(a[i][0])
x[:, 2:3] = a

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, :])
x[:, :] = imputer.transform(x[:, :])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#using elbow mwthod to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300,n_init = 20,random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,21),wcss)
plt.title('the elbow methd')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#applying k-means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 8, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
y = kmeans.fit_predict(x)
"""y_pred = kmeans.predict()"""

#x = sc.inverse_transform(x)

#visualising the clusters
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(x[y == 5, 0], x[y == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(x[y == 6, 0], x[y == 6, 1], s = 100, c = 'pink', label = 'Cluster 7')
plt.scatter(x[y == 7, 0], x[y == 7, 1], s = 100, c = 'yellow', label = 'Cluster 8')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()