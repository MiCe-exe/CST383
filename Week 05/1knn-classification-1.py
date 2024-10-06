# -*- coding: utf-8 -*-
"""

Code related to KNN lecture 1

"""

import numpy as np
import pandas as pd
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
# slide 11
#

# compute distance between two n-dimensional points
def edist(x,y):
    return np.sqrt(np.sum((x-y)**2))

# test
x1 = np.array([1.0, 2.0])
x2 = np.array([1.0, 2.0])
x3 = np.array([2.0, 1.0])
edist(x1,x2)
edist(x1,x3)

# return a upper triangular distance matrix based on rows of float matrix x
# (i.e. values on the diagonal and to the lower left are all 0)
# 
def dist(x):
    m = x.shape[0]
    dm = np.zeros((m,m))
    for i in range(m):
        for j in range(i,m):
            dm[i,j] = edist(x[i,:], x[j,:])
    return dm



# test
n = 10
x1 = np.random.rand(n)
x2 = np.random.rand(n)
x = np.stack([x1,x2], axis=1)
dist(x)

x3 = np.random.rand(n)
x = np.stack([x1,x2,x3], axis=1)
dist(x)

#
# slide 13
#

# note the index_col=0 option
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)
df.drop(['Private'], axis=1, inplace=True)

# scale 
df = df.apply(zscore)
df = df[['Outstate', 'F.Undergrad']]

df.head(5)

x = df.values
x1 = df.loc['Malone College'].values

k = 3
dists = np.apply_along_axis(lambda x: edist(x, x1), 1, x)
topk = np.argsort(dists)[1:(k+1)]
topk

df.iloc[18]





