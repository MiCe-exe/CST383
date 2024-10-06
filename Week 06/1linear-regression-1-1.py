# -*- coding: utf-8 -*-
"""

Code related to linear regression lecture 1

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# reduce the default number of significant digits
np.set_printoptions(precision=3)

# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
# sns.set_context('notebook')   
# sns.set_context('paper')  # smaller
sns.set_context('talk')   # larger

# change default plot size
rcParams['figure.figsize'] = 7,5

# =============================================================================
# 
# mtcars data
# 
# =============================================================================

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/mtcars.csv")

# use just a few cars
df = df.iloc[[17,8,5],:]

plt.figure(figsize=(6,6))

df.plot.scatter(x='hp', y='mpg', c='darkred', s=70)

df.plot.scatter(x='disp', y='mpg', c='darkred', s=70)


# line 1
w0_1 = 38
w1_1 = -0.1
x = np.linspace(75, 225, 10)
y = w0_1 + w1_1*x

# line 2
w0_2 = 44
w1_2 = -0.12
y1 = w0_2 + w1_2*x

df.plot.scatter(x='disp', y='mpg', c='darkred', s=70)
plt.plot(x, y, c='orange', label='{};{}'.format(w0_1,w1_1))
plt.plot(x, y1, c='darkblue', label='{};{}'.format(w0_2,w1_2))
plt.legend(title='intercept; slope')

actual = df['mpg'].values

# errors for line 1
predicted1 = w0_1 + w1_1 * df['disp'].values
errs1 = predicted1 - actual
errs1_sq = errs1**2
sse1 = np.sum(errs1_sq)

# errors for line 2
predicted2 = w0_2 + w1_2 * df['disp'].values
errs2 = predicted2 - actual
errs2_sq = errs2**2
sse2 = np.sum(errs2_sq)

# errors for best line
regr = LinearRegression()
regr.fit(df[['disp']], df['mpg'])
predicted3 = regr.predict(df[['disp']])
w0_3 = regr.intercept_
w1_3 = regr.coef_
errs3 = predicted3 - actual
errs3_sq = errs3**2
sse3 = np.sum(errs3_sq)
y2 = w0_3 + w1_3 * x

df.plot.scatter(x='disp', y='mpg', c='darkred', s=70)
plt.plot(x, y2, c='darkgreen', label='{0:0.1f};{1:0.2f}'.format(w0_3,w1_3[0]))
plt.legend(title='intercept; slope')

regr = LinearRegression()
regr.fit(df[['hp']], df['mpg'])
sns.scatterplot(x='hp', y='mpg', data=df)

df.plot.scatter(x='hp', y='mpg', c='darkred', s=70)
plt.plot(x, y2, c='darkgreen', label='{0:0.1f};{1:0.2f}'.format(w0_3,w1_3[0]))
plt.legend(title='intercept; slope')

x = np.linspace(80, 200, 7)
y = 50 - 0.15 * x
plt.scatter(x,y)
plt.plot(x, y, linestyle=':')
plt.xlabel('Horsepower')
plt.ylabel('Mileage (MPG)')
plt.title('Linear model of car mileage by horsepower')

# college
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)

# college, 1 predictor
predictors = ['Top10perc']
target = 'Outstate'
X = df[predictors].values
y = df[target].values

regr = LinearRegression()
regr.fit(X, y)
b0 = np.round(regr.intercept_, 1)
b1 = np.round(regr.coef_, 1)
print('b0: {:.2f}, b1: {:.2f}'.format(regr.intercept_, reg.coef_[0]))

predicted = regr.predict(X)
MSE = ((predicted - y)**2).mean()

sns.scatterplot(x='Top10perc', y='Outstate', data=df)
plt.plot(X, predicted, color='darkorange')
plt.title('b0 = {}, b1 = {}, MSE = {}'.format(b0, b1[0], np.round(MSE,2)))
plt.xlabel('Top 10 percent of high school class')
plt.ylabel('Out of state tuition')

# college, two predictors

predictors = ['Top10perc', 'F.Undergrad']
target = 'Outstate'
X = df[predictors].values
y = df[target].values

regr = LinearRegression()
regr.fit(X, y)
regr.intercept_
regr.coef_

predicted = regr.predict(X)

print(predictions[:8].astype(int))
print(y[:8])

print('b0: {:.1f}, b1: {:.1f}, b2: {:.2f}'.format(regr.intercept_, regr.coef_[0], regr.coef_[1]))

#
# slide 7
#

sns.scatterplot(x='cs', y='prp', data=df)
plt.title('CPU performance by clock speed')
plt.xlabel('clock speed (MHz)')
plt.ylabel('CPU performance')

# slide 8

# make a tiny version of the data set
i = 48
np.random.seed(i)
df1 = df.iloc[np.random.choice(df.shape[0], 4),:]

sns.scatterplot(x='cs', y='prp', data=df1)
plt.title('CPU performance by clock speed')
plt.xlabel('clock speed (MHz)')
plt.ylabel('CPU performance')

df1 = df1[['cs', 'prp']]

# model 1
b0 = 90
b1 = 5

predicted_prp1 = [b0 + b1*x for x in df1['cs']]
error = predicted_prp1 - df1['prp']
MSE = np.mean(error**2)

print(predicted_prp1)
print(list(df1['prp']))

sns.scatterplot(x='cs', y='prp', data=df1)
plt.plot(df1['cs'], predicted_prp1, color='darkorange')
plt.title('b0 = {}, b1 = {}, MSE = {}'.format(b0, b1, np.round(MSE,2)))
plt.xlabel('clock speed')
plt.ylabel('performance')

# model 2
b0 = 50
b1 = 12

predicted_prp2 = [b0 + b1*x for x in df1['cs']]
error = predicted_prp2 - df1['prp']
MSE = np.mean(error**2)

print(predicted_prp2)
print(list(df1['prp']))

sns.scatterplot(x='cs', y='prp', data=df1)
plt.plot(df1['cs'], predicted_prp2, color='darkgreen')
plt.title('b0 = {}, b1 = {}, MSE = {}'.format(b0, b1, np.round(MSE,2)))
plt.xlabel('clock speed')
plt.ylabel('performance')

# model 3  (found to be best by linear regression)
b0 = 14.64
b1 = 15.52

predicted_prp3 = [b0 + b1*x for x in df1['cs']]
error = predicted_prp3 - df1['prp']
MSE = np.mean(error**2)

sns.scatterplot(x='cs', y='prp', data=df1)
plt.plot(df1['cs'], predicted_prp3, color='darkred')
plt.title('b0 = {}, b1 = {}, MSE = {}'.format(b0, b1, np.round(MSE,2)))
plt.xlabel('clock speed')
plt.ylabel('performance')

# combined plot

sns.scatterplot(x='cs', y='prp', data=df1)
plt.plot(df1['cs'], predicted_prp1, color='darkorange')
plt.plot(df1['cs'], predicted_prp2, color='darkgreen')
plt.title('Models 1 and 2')
plt.xlabel('clock speed')
plt.ylabel('performance')

# =============================================================================
# 
# linear regression with Scikit-Learn
#
# =============================================================================

# fit on the tiny data set 

# convert to NumPy arrays
X = df1[['cs']].values   # 2D
y = df1['prp'].values    # 1D

# fit the model
reg = LinearRegression()
reg.fit(X, y)

# what are the coefficients of the model?
print('b0: {:.2f}, b1: {:.2f}'.format(reg.intercept_, reg.coef_[0]))

# what is the MSE?
print('MSE: {:.2f}'.format(mean_squared_error(reg.predict(X), y)))

# fit on the full data set, using more predictors

# convert to NumPy arrays
X = df[['cs', 'cach']].values   # 2D
y = df['prp'].values            # 1D

# fit the model
reg = LinearRegression()
reg.fit(X, y)

# what are the coefficients of the model?
b0, b1, b2 = reg.intercept_, reg.coef_[0], reg.coef_[1]
print('b0: {:.2f}, b1: {:.2f}, b2: {:.2f}'.format(b0, b1, b2))

# what is the MSE?
print('MSE: {:.2f}'.format(mean_squared_error(reg.predict(X), y)))

# making predictions

df[['cs', 'cach', 'prp']].head(3)

reg.fit(X,y)

predicted_prp = reg.predict(X)

predicted_prp[:3]

reg.predict(pd.DataFrame({'cs': [50,100], 'cach': [256,128]}))

