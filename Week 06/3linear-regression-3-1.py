# -*- coding: utf-8 -*-
"""
Linear regression: extending the scope

@author: Glenn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import rcParams

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
#sns.set_context('paper')   # 'talk' for slightly larger
sns.set_context('talk')   # 'talk' for slightly larger

# change default plot size
rcParams['figure.figsize'] = 8,6

#
# slides 3 and 4
#

# fake data
n = 50
np.random.seed(2)
cache = np.linspace(32, 2048, num=n)
perf = np.log(cache) + np.random.normal(scale=0.5, size=n)

X = cache.reshape((-1,1))
y = perf

reg = LinearRegression()
reg.fit(X, y)

reg.score(X, y)
reg.intercept_
reg.coef_

sns.scatterplot(cache, perf, s=40)
plt.title('CPU performance by cache size')
plt.xlabel('cache size (KB)')
plt.ylabel('performance')
# add the fit
plt.plot(cache, cache*reg.coef_[0] + reg.intercept_, color='black', linestyle='dashed')

# try using polynomial features

pf = PolynomialFeatures(degree=3)
pf.fit(X)
X_poly = pf.transform(X)

reg.fit(X_poly, y)

reg.score(X_poly, y)



#
# slides 5 and 6
#

log_cache = np.log(cache)

X = log_cache.reshape((-1,1))
y = perf

reg = LinearRegression()
reg.fit(X, y)

reg.score(X, y)
reg.intercept_
reg.coef_

sns.scatterplot(log_cache, perf, s=40)
plt.title('CPU performance by cache size')
plt.xlabel('cache size (KB, log scale)')
plt.ylabel('performance')
# add the fit
plt.plot(log_cache, log_cache*reg.coef_[0] + reg.intercept_, color='black', linestyle='dashed')

sns.scatterplot(cache, perf, s=40)
plt.title('CPU performance by cache size')
plt.xlabel('cache size (KB)')
plt.ylabel('performance')
# add the fit
plt.plot(cache, np.log(cache*reg.coef_[0] + reg.intercept_), color='black', linestyle='dashed')




#
# slide 11 - polynomial features
#

#
# CPU data
#

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)    # clock speed in MHz (millions of cycles/sec)

X = df[['cach', 'cs', 'mmax']].values
y = df['prp'].values

pf = PolynomialFeatures(degree=3, include_bias=False)
pf.fit(X)

X_poly = pf.transform(X)
X_poly.shape
pf.get_feature_names()

# fit with new model

reg = LinearRegression()
reg.fit(X_poly, y)

reg.score(X_poly, y)
reg.intercept_
reg.coef_

# were we overfitting?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
reg.fit(X_train, y_train)
predict = reg.predict(X_test)
RMSE = np.sqrt(((predict - y_test)**2).mean())
RMSE

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20, random_state=42)
reg.fit(X_train, y_train)
predict = reg.predict(X_test)
RMSE = np.sqrt(((predict - y_test)**2).mean())
RMSE

# really, to check for overfitting we need to compare training and test performance

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20, random_state=42)
reg.fit(X_train, y_train)

# rmse on training data
predict = reg.predict(X_train)
RMSE = np.sqrt(((predict - y_train)**2).mean())
RMSE

# rmse on test data
predict = reg.predict(X_test)
RMSE = np.sqrt(((predict - y_test)**2).mean())
RMSE


# 
# example for class
#

df = df[df['cach'] < 100]

X = df[['cach']].values
y = df['prp'].values
X_fake = np.linspace(0, 100, 50).reshape(50, 1)

deg = 3
pf = PolynomialFeatures(degree=deg, include_bias=False)
pf.fit(X)
X_poly = pf.transform(X)
fnames = pf.get_feature_names()

reg = LinearRegression()
reg.fit(X_poly, y)
y_predict = reg.predict(pf.transform(X_fake))

plt.plot(X_fake, y_predict, color='darkred')
plt.scatter(X, y, s=30)
plt.title('degree = {}, features = {}'.format(deg, ','.join(fnames)))
plt.xlabel('cache size')
plt.ylabel('performance')










