import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)

df.info()
df.head()

#2
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(train_df.shape)
print(test_df.shape)

#3
X_train = train_df.drop(columns=['prp'])
y_train = train_df['prp']
X_test = test_df.drop(columns=['prp'])
y_test = test_df['prp']

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_final = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_final = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Training RMSE: {train_final}")
print(f"Test RMSE: {test_final}")

#4
def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


predicted_fr_erp = calculate_rmse(X_test, y_test)

sns.scatterplot(x='myct', y='prp', data=df)

# try linear fit with myct

X = df[['myct']].values
y = df['prp'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 42)

reg = LinearRegression()
reg.fit(X_train, y_train)

print('Intercept: {:.2f}'.format(reg.intercept_))
print('myct: {:.3f}'.format(reg.coef_[0])) 