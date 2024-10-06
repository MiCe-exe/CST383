# -*- coding: utf-8 -*-
"""

Code related to KNN anomaly detection lecture

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
# sns.set_context('notebook')   
# sns.set_context('paper')  # smaller
sns.set_context('talk')   # larger

# change default plot size
rcParams['figure.figsize'] = 8,6

#
# read the college data
#

# note the index_col=0 option
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)
private = df.Private
df.drop(['Private'], axis=1, inplace=True)

# look at only 
df = df[['Outstate', 'F.Undergrad']]

# scale the data
df_raw = df
df = df.apply(zscore)

#
# slides 13 and 14
#


# compute distance between two n-dimensional points
def edist(x,y):
    return np.sqrt(np.sum((x-y)**2))

# return a distance matrix based on columns of float matrix x
def dist(x):
    m = x.shape[0]
    dm = np.zeros((m,m))
    for i in range(m):
        for j in range(i,m):
            dm[i,j] = edist(x[i,:], x[j,:])
            dm[j,i] = dm[i,j]
    return dm

dm = dist(df.values)
dm.round(2)[:3,:6]

sorted_distances = np.apply_along_axis(np.sort, 1, dm)

# values in row 1 are distances to nearest neighbor
k = 3
dist_to_k_nearest = sorted_distances[:,k]

dist_to_k_nearest.round(2)[:5]

# which colleges have greatest distance to kth nearest neighbor?

indices_of_weird_colleges = np.argsort(-dist_to_k_nearest)
df.index.values[indices_of_weird_colleges[:10]]

# plot the colleges and see about the 10 weirdest ones
cols = np.full(len(df), 'black')
cols[indices_of_weird_colleges[:10]] = 'red'
df.plot.scatter('Outstate', 'F.Undergrad',c=cols)

# what about the distribution of distances?
plt.hist(dist_to_k_nearest)
# call a distance > 4*std_dev a big distance
big_dist = 4*np.std(dist_to_k_nearest)
sorted_distances = -np.sort(-dist_to_k_nearest)
number_of_weird_colleges = np.sum(sorted_distances > big_dist)

#
# wrap code up into a function
#

# given an m x n matrix x of scaled instances (each row an instance),
# return a 1D array kdist of length m, where kdist[i] is the distance
# from the instance on row i to its k-nearest neighbor
def knn_anom_detect(x, k):
    dm = dist(x)
    sorted_distances = np.apply_along_axis(np.sort, 1, dm)
    dist_to_k_nearest = sorted_distances[:,k]
    return dist_to_k_nearest

# plot the n biggest anomalies in matrix x (must have exactly two columns)
def plot_anomalies(x, n=10, k=3):
    kdist = knn_anom_detect(x, k)
    weird_ones = np.argsort(-kdist)[:n]
    cols = np.full(kdist.size, 'black')
    cols[weird_ones] = 'red'
    plt.scatter(x[:,0], x[:,1], c=cols, s=30)

# test
temp = knn_anom_detect(df.values, 5)

plot_anomalies(df.values)

plot_anomalies(df.values, n=5, k=5)

#
# special purpose code for colleges
#

# plot the n biggest anomalies in matrix x (must have exactly two columns)
def plot_weird_colleges(df, n=10, k=3):
    x = df.apply(zscore).values
    kdist = knn_anom_detect(x, k)
    weird_ones = np.argsort(-kdist)[:n]
    cols = np.full(kdist.size, 'black')
    cols[weird_ones] = 'red'
    df.plot.scatter('Outstate', 'F.Undergrad',c=cols)

plot_weird_colleges(df_raw, n=20, k=3)


    














