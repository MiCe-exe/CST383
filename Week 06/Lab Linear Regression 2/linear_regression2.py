import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)	# clock speed in MHz 


#2
X = df[['mmin', 'mmax']]
y = df['prp']
X = X.to_numpy()
y = y.to_numpy()

#3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

#4
print("Coefficients:", model.coef_)

#5 yes

#6 best should be 1
r_squared = model.score(X_train, y_train)
print("R-squared value for the model:", r_squared)

#7
y_pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, color='blue', label='Actual vs. Predicted PRP')
plt.xlabel('Predicted PRP')
plt.ylabel('Actual PRP')
plt.title('Actual vs. Predicted PRP Scatterplot')
plt.legend()
plt.grid(True)