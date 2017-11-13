# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import math

# Importing the dataset
dataset = pd.read_csv('TED_Talks_by_ID_plus.csv')
#Dropping the unneccessary columns
dataset.drop(dataset.columns[[0, 1, 2, 3, 4, 5, 8, 10, 12, 13, 14]], axis=1, inplace=True)
#Slicing the data into required values columns
x = dataset.iloc[100 :, 2:].values
a = x[:, :1]


#In our dataset, we have the time variable, since it's hard to understand the type of the dataset, we
#changed the time format into the number of seconds
def get_sec(time_str):
    h, m, s = time_str.split(':') #Splitting based on :
    return int(h) * 3600 + int(m) * 60 + int(s) #converting into seconds

#applying the defined function for the each value in the time column in the dataset
for i in range(len(x)):
    c = str(a[i][0])
    a[i][0] = get_sec(c)
    a[i][0] = int(a[i][0])
x[:, :1] = a


# Taking care of missing data
from sklearn.preprocessing import Imputer
#Here we replaced the NaN values with the mean as it;s the best strategy to remove the NaN values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, :])
x[:, :] = imputer.transform(x[:, :])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x) #fitting into the standard scaler method/function

#small test_set
x_test = x[:100, :]

#As we don't the best number of clusters to be made, we use the elbow method strategy
#using elbow mwthod to find optimal number of clusters
from sklearn.cluster import KMeans
#initializing the weighted clusters sum of squares
wcss = []
for i in range(1,11):
    #choosing the initialization method as k-means++ is best instead of selecting the random centroids
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow methd')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show() #Plotting the data to visualize and recognize the best no. of clusters

#applying k-means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 6, init = 'k-means++',max_iter = 300,n_init = 10,random_state = 0)
y = kmeans.fit_predict(x)
y_kmeans = kmeans.predict(x_test)

#output the number of clusters and for the each category using collections library
import collections
x_col = collections.Counter(y)
x_col_test = collections.Counter(y_kmeans)
print('training set : ', x_col)
print('test set : ', x_col_test)

#visualising the clusters
#plotting with X, Y and size = 100mm, color as 'c', and labelling
plt.scatter(x[y == 0, 0], x[y == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y == 2, 0], x[y == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y == 3, 0], x[y == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y == 4, 0], x[y == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(x[y == 5, 0], x[y == 5, 1], s = 100, c = 'brown', label = 'Cluster 6')
plt.title('Clusters of customers')
plt.xlabel('X label')
plt.ylabel('Y label')
plt.legend()
plt.show()


#Building the confusion matrix for the TED Clustering
"""#building a confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true, y_kmeans)
print(cm)
ac = accuracy_score(y_true, y_kmeans)
print(ac)"""
