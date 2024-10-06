import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)

#2
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

#3
predictors = ['myct', 'mmin']
X_train = train_df[predictors]
y_train = train_df['prp']
X_test = test_df[predictors]
y_test = test_df['prp']

model = LinearRegression()
model.fit(X_train, y_train)

#4
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error test set: {mse}")

#5
def repeat_steps():
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=np.random.randint(1000))

    predictors = ['myct', 'mmin']
    X_train = train_df[predictors]
    y_train = train_df['prp']
    X_test = test_df[predictors]
    y_test = test_df['prp']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    return mse

mse_original = repeat_steps()
print(f"Mean Squared Error (Original): {mse_original}")

mse_new = repeat_steps()
print(f"Mean Squared Error (New): {mse_new}")

rmse_difference = np.sqrt(abs(mse_original - mse_new))
print(f"Difference in RMSE: {rmse_difference}")

labels = ['Original', 'New']
mse_values = [mse_original, mse_new]

plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.xlabel('Run')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error Comparison')
plt.show()