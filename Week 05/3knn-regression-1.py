# -*- coding: utf-8 -*-
"""

Code related to KNN lecture 3

"""

import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split

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
private = df.Private == 'Yes'
df.drop(['Private'], axis=1, inplace=True)

# look at only two columns
df = df[['Outstate', 'F.Undergrad']]

# scale the data
df = df.apply(zscore)

# 
# prepare data for sklearn
#

# convert to NumPy
# note that string or boolean labels can be used directly, at least with KNN
X = df.values
y = private.values.astype(int)

# test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#
# knn classification
#

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# for classifiers, score is accuracy
accuracy = knn.score(X_test, y_test)

# verify
predicted = knn.predict(X_test)
actual = y_test
np.mean(actual == predicted)

# baseline
baseline_accuracy = pd.Series(y_train).value_counts().iloc[0] / y_train.size
baseline_accuracy

#
# knn regression
#

# read and prepare data
# we will predict out of state tuition
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)
tuition = df.Outstate
df.drop(['Outstate'], axis=1, inplace=True)
df = df[['Top10perc', 'F.Undergrad', 'S.F.Ratio']]
df = df.apply(zscore)

# convert to NumPy and test/train split
X = df.values
y = tuition.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# knn regression; predict tuition
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
# make predictions
predicted = knn.predict(X_test)
rmse = np.sqrt(np.mean((y_test - predicted)**2))

# verify
predicted = knn.predict(X_test)
actual = y_test
np.mean(actual == predicted)