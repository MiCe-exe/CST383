# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:10:24 2019

@author: Glenn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# read the data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/german-credit.csv")
bad_loan = df['good.loan'] - 1

# use only numeric data, and scale it
df = df[["duration.in.months", "amount", "percentage.of.disposable.income", "at.residence.since", 
              "age.in.years", "num.credits.at.bank"]]
X = df.apply(zscore).values
y = bad_loan.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# see how knn classifier works as training size changes


k = 3
knn = KNeighborsClassifier(n_neighbors=k)
te_errs = []
tr_errs = []
tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
for tr_size in tr_sizes:
  X_train1 = X_train[:tr_size,:]
  y_train1 = y_train[:tr_size]
  
  # train model on a subset of the training data
  knn.fit(X_train1, y_train1)

  # error on subset of training data
  tr_predicted = knn.predict(X_train1)
  err = (tr_predicted != y_train1).mean()
  tr_errs.append(err)
  
  # error on all test data
  te_predicted = knn.predict(X_test)
  err = (te_predicted != y_test).mean()
  te_errs.append(err)

#
# plot the learning curve here
#
#3
plt.figure(figsize=(10, 8))
plt.plot(tr_sizes, tr_errs, label='Training error', color='blue', marker='o')
plt.plot(tr_sizes, te_errs, label='Test error', color='red', marker='x')
plt.xlabel('Training set size')
plt.ylabel('Classification error')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)

#4
k_values = [1, 3, 4, 5]

  
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

# Original compute_errors function logic is moved to the main loop
for i, k in enumerate(k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    te_errs = []
    tr_errs = []
    tr_sizes = np.linspace(100, X_train.shape[0], 10).astype(int)
    for tr_size in tr_sizes:
        X_train1 = X_train[:tr_size, :]
        y_train1 = y_train[:tr_size]

        # Train model on a subset of the training data
        knn.fit(X_train1, y_train1)

        # Error on subset of training data
        tr_predicted = knn.predict(X_train1)
        tr_err = (tr_predicted != y_train1).mean()
        tr_errs.append(tr_err)

        # Error on all test data
        te_predicted = knn.predict(X_test)
        te_err = (te_predicted != y_test).mean()
        te_errs.append(te_err)

    # Plot the learning curve
    axes[i].plot(tr_sizes, tr_errs, label='Training error', color='blue', marker='o')
    axes[i].plot(tr_sizes, te_errs, label='Test error', color='red', marker='x')
    axes[i].set_title(f'Learning Curve for k={k}')
    axes[i].set_xlabel('Training set size')
    axes[i].set_ylabel('Classification error')
    axes[i].legend()
    axes[i].grid(True)


#5
# k = 1 low has a low training error because it can be a result of overfitting since we're using closest neighbor. 
# With the other graphs the error increases because K  become larger. Better closest neighbor samle is used. 

#6
# Low values shows large gaps between the data. It has high variance.
# values of 3 and 5 show balance between both datas
# Values with a high value has a high bias. 

