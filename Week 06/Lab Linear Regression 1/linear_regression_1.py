import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)	

#3
sns.pairplot(df)

#4 yes
plt.figure(figsize=(10, 8))
sns.scatterplot(x='cach', y='prp', data=df)
plt.title('Scatter plot of PRP vs Cache Size (cach)')
plt.xlabel('Cache Size (cach)')
plt.ylabel('Processor Performance (prp)')

#5
X = df[['cach']].values
y = df['prp'].values
model = LinearRegression()
fit = model.fit(X, y)
print(f"Intercept: {fit.intercept_}")
print(f"Coefficient: {fit.coef_[0]}")


#6
plt.figure(figsize=(10, 8))
sns.regplot(x='cach', y='prp', data=df, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Seaborn regplot of PRP vs Cache Size (cach)')
plt.xlabel('Cache Size (cach)')
plt.ylabel('Processor Performance (prp)')

#7
df['predicted_prp'] = fit.predict(X)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=df['prp'], y=df['predicted_prp'], label='Predicted vs Actual')
plt.plot([min(df['prp']), max(df['prp'])], [min(df['prp']), max(df['predicted_prp'])], color='red', linestyle='--', label='Ideal Fit')
plt.title('Scatter plot of Actual vs Predicted PRP')
plt.xlabel('Actual PRP')
plt.ylabel('Predicted PRP')
plt.legend()