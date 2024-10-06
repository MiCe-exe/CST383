import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# set default figure size
plt.rcParams['figure.figsize'] = [8.0, 6.0]

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/housing.csv")    
df.info()

df.describe()

print(df.isnull().sum())
df[df['total_bedrooms'].isnull()]

df = df.dropna()
# for repeatability
np.random.seed(42)   

# select the predictor variables and target variables to be used with regression
predictors = ['longitude','latitude','housing_median_age','total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
#dropping categortical features, such as ocean_proximity, including spatial ones such as long/lat.
target = 'median_house_value'
X = df[predictors].values
y = df[target].values

# KNN can be slow, so get a random sample of the full data set
indexes = np.random.choice(y.size, size=10000)
X_mini = X[indexes]
y_mini = y[indexes]

# Split the data into training and test sets, and scale
scaler = StandardScaler()

# unscaled version (note that scaling is only used on predictor variables)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_mini, y_mini, test_size=0.30, random_state=42)

# scaled version
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# sanity check
print(X_train.shape)
print(X_train[:3])

#Baseline performance
#For regression problems, our baseline is the "blind" prediction that is just the average value of the target 
#variable. The blind prediction must be calculated using the training data. Calculate and print the test set root mean 
#squared error (test RMSE) using this blind prediction. I have provided a function you can use for RMSE.

def rmse(predicted, actual):
    return np.sqrt(((predicted - actual)**2).mean())

# YOUR CODE HERE
y_train_mean = np.mean(y_train)
y_pred_baseline = np.full_like(y_test, y_train_mean)
baseline_rmse = rmse(y_pred_baseline, y_test)
print(baseline_rmse)

knn_regressor = KNeighborsRegressor(algorithm='brute')
knn_regressor.fit(X_train, y_train)
y_pred_knn = knn_regressor.predict(X_test)
knn_rmse = rmse(y_pred_knn, y_test)
print(knn_rmse)

def get_train_test_rmse(regr, X_train, X_test, y_train, y_test):
    regr.fit(X_train, y_train)
    y_train_pred = regr.predict(X_train)
    y_test_pred = regr.predict(X_test)
    train_rmse = rmse(y_train_pred, y_train)
    test_rmse = rmse(y_test_pred, y_test)  
    return train_rmse, test_rmse

n = 30
test_rmse = []
train_rmse = []
ks = np.arange(1, n+1, 2)
for k in ks:
    print(k, ' ', end='')
    regr = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    rmse_tr, rmse_te = get_train_test_rmse(regr, X_train, X_test, y_train, y_test)
    train_rmse.append(rmse_tr)
    test_rmse.append(rmse_te)
print('done')

# sanity check
print('Test RMSE when k = 3: {:0.1f}'.format(np.array(test_rmse)[ks==3][0]))

def get_best(ks, rmse):
    min_rmse = float('inf')  # Initialize min_rmse to positive infinity
    best_k = None  # Initialize best_k to None
    
    # Iterate through the values of k and their corresponding RMSE values
    for k, rmse_value in zip(ks, rmse):
        if rmse_value < min_rmse:
            min_rmse = rmse_value
            best_k = k
    return best_k, min_rmse

best_k, best_rmse = get_best(ks, test_rmse)
print('best k = {}, best test RMSE: {:0.1f}'.format(best_k, best_rmse))

plt.plot(ks, train_rmse, marker='o', label='Train RMSE')
plt.plot(ks, test_rmse, marker='o', label='Test RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('KNN Regression: RMSE vs. k')
plt.legend()
plt.show()

def add_noise_predictor(X):
    """ add a column of random values to 2D array X """
    noise = np.random.normal(size=(X.shape[0], 1))
    return np.hstack((X, noise))


train_rmse_noisy = []
test_rmse_noisy = []
percent_increase_rmse = []

X_train_noisy = X_train.copy()
X_test_noisy = X_test.copy()

baseline_rmse = test_rmse[0]

for i in range(0, 5):
    print(i, ' ', end='')

    baseline_rmse = test_rmse[i]

    X_train_noisy = add_noise_predictor(X_train_noisy)
    X_test_noisy = add_noise_predictor(X_test_noisy)

    regr = KNeighborsRegressor(n_neighbors=10, algorithm='brute')
    rmse_tr, rmse_te = get_train_test_rmse(regr, X_train_noisy, X_test_noisy, y_train, y_test)
    train_rmse_noisy.append(rmse_tr)
    test_rmse_noisy.append(rmse_te)

    percent_increase_rmse.append(100 * (rmse_te - baseline_rmse) / baseline_rmse)

print('done')

plt.plot(range(5), percent_increase_rmse, marker='o')
plt.xlabel('Number of Noisy Predictors')
plt.ylabel('Percent Increase in Test RMSE (%)')
plt.title('Percent Increase in Test RMSE vs. Number of Noisy Predictors')
plt.xticks(np.arange(0, 5, 0.5))
plt.yticks(np.arange(0, max(percent_increase_rmse) + 2.5, 2.5)) 
plt.grid(True)
plt.show()


#X_train_raw, X_test_raw
n = 30
test_rmse = []
train_rmse = []
ks = np.arange(1, n+1, 2)
for k in ks:
    print(k, ' ', end='')
    regr = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    rmse_tr, rmse_te = get_train_test_rmse(regr, X_train_raw, X_test_raw, y_train, y_test)
    train_rmse.append(rmse_tr)
    test_rmse.append(rmse_te)
print('done')

best_k, best_rmse = get_best(ks, test_rmse)
print('best k = {}, best test RMSE: {:0.1f}'.format(best_k, best_rmse))

plt.plot(ks, test_rmse)
plt.plot(ks, train_rmse)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.legend(['Test unscaled','Train unscaled'])

n = 9
test_rmse_ball_tree= np.array([])
test_rmse_kd= np.array([])
test_rmse_brute= np.array([])
algs = ['ball_tree', 'kd_tree', 'brute']
ks = np.arange(1, n+1, 2)
for alg in algs:
  train_rmse_alg= []
  test_rmse_alg= []
  
  for k in ks:
    print(k, ' ', end='')
    regr = KNeighborsRegressor(n_neighbors=k, algorithm= alg)
    rmse_tr, rmse_te = get_train_test_rmse(regr, X_train, X_test, y_train, y_test)

    if alg == 'ball_tree':
      test_rmse_ball_tree = np.append(test_rmse_ball_tree, rmse_te)
    elif alg == 'kd_tree':
      test_rmse_kd = np.append(test_rmse_kd,rmse_te)
    elif alg == 'brute':
      test_rmse_brute = np.append(test_rmse_brute, rmse_te)
    else:
      print('invalid algorithm')
  print(alg, 'done')

  