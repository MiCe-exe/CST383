# -*- coding: utf-8 -*-
"""
Linear regression lecture 2

@author: Glenn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams

# allow output to span multiple output lines in the console
pd.set_option('display.max_columns', 500)

# switch to seaborn default stylistic parameters
# see the very useful https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
#sns.set_context('paper')   # 'talk' for slightly larger
sns.set_context('talk')   # 'talk' for slightly larger

# change default plot size
rcParams['figure.figsize'] = 7,5

#
# slide 5
#

# college data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/College.csv", index_col=0)    
df['Enrollperc'] = 100*(df['Enroll']/df['Accept'])
df['Acceptperc'] = 100*(df['Accept']/df['Apps'])

# explore
sns.distplot(df['Outstate'])
plt.title('Histogram of tuition')

sns.scatterplot(x='Enrollperc', y='Outstate', data=df, s=40)
plt.title('Tuition by Enroll percent')

#
# slide 6
#

y = df['Outstate'].values
predictors = ['Grad.Rate', 'Top10perc']
X = df[predictors].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#
# slide 7
#

# fit the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# what are the coefficients of the model?
print('Intercept: {:.2f}'.format(reg.intercept_))
print('Grad.Rate: {:.3f}'.format(reg.coef_[0]))
print('Top10perc: {:.3f}'.format(reg.coef_[1]))

print('coefficients: \n', reg.coef_)

#
# slide 9
#

predictors = ['Grad.Rate', 'Top10perc', 'Top25perc']
X = df[predictors].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

# what are the coefficients of the model?
print('Intercept: {:.2f}'.format(reg.intercept_))
print('Grad.Rate: {:.3f}'.format(reg.coef_[0]))
print('Top10perc: {:.3f}'.format(reg.coef_[1]))
print('Top25perc: {:.3f}'.format(reg.coef_[2]))

#
# slide 10
#

df1 = pd.read_csv("C:/Users/Glenn/Google Drive/CSUMB/Fall19/DS/lectures-labs/5machine-learning/2linear-regression/resources/steven-meetup/states.csv", index_col=0)
X = df1['dollars'].values.reshape((-1,1))
y = df1['SATM'].values
reg = LinearRegression()
reg.fit(X,y)
print('Intercept: {:.2f}'.format(reg.intercept_))
print('dollars: {:.3f}'.format(reg.coef_[0]))

#
# slide 11
#

X = df1[['dollars', 'percent']].values
y = df1['SATM'].values
reg = LinearRegression()
reg.fit(X,y)
print('Intercept: {:.2f}'.format(reg.intercept_))
print('dollars: {:.3f}'.format(reg.coef_[0]))
print('percent: {:.3f}'.format(reg.coef_[1]))

#
# slide 12
#

# unscaled

predictors = ['Expend', 'perc.alumni', 'Room.Board']
X = df[predictors].values
y = df['Outstate'].values

reg = LinearRegression()
reg.fit(X, y)
print('Intercept: {:.2f}'.format(reg.intercept_))
print('Expend: {:.3f}'.format(reg.coef_[0]))
print('perc.alumni: {:.3f}'.format(reg.coef_[1]))
print('Room.Board: {:.3f}'.format(reg.coef_[2]))

# scaled

X_scaled = df[predictors].apply(zscore).values
reg.fit(X_scaled, y)
print('Intercept: {:.2f}'.format(reg.intercept_))
print('Expend: {:.3f}'.format(reg.coef_[0]))
print('perc.alumni: {:.3f}'.format(reg.coef_[1]))
print('Room.Board: {:.3f}'.format(reg.coef_[2]))

#
# slide 13
#

predictors = ['Expend', 'perc.alumni', 'Room.Board']
X = df[predictors].values
y = df['Outstate'].values

reg = LinearRegression()
reg.fit(X, y)
r2 = reg.score(X,y)
print('R-squared: {:.2f}'.format(r2))

#
# slide 14
#

predicted1 = df['Outstate'].mean()
err1 = ((df['Outstate'] - predicted1)**2).mean()

reg.fit(X, y)
predicted2 = reg.predict(X)
err2 = ((df['Outstate'] - predicted2)**2).mean()

#
# slide 15
#

df['Private'] = (df['Private'] == 'Yes').astype(int)
predictors = ['Private', 'Room.Board', 'perc.alumni', 'Expend']
X = df[predictors].values
y = df['Outstate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
predicted = reg.predict(X_test)

sns.scatterplot(y_test, predicted, s=40)
biggest = np.concatenate([predicted, y_test]).max()
plt.plot([0, biggest], [0, biggest], color='grey', linestyle='dashed')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('Predicted vs. actual values')

# what is the MSE?
RMSE = np.sqrt(((y_test-predicted)**2).mean())

print('MSE: {:.2f}'.format(mean_squared_error(reg.predict(X), y)))