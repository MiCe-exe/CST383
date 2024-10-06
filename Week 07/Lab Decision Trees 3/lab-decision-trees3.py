import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import graphviz

sns.set()
sns.set_context('talk')
rcParams['figure.figsize'] = 10, 8

# Read the CPU data
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor'] + ' ' + df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3 / df['myct'], 2)  # Clock speed in MHz

# Get ready for Scikit-Learn
predictors = ['mmin', 'chmax']  # Choose predictors as you like
target = 'prp'
X = df[predictors].values
y = df[target].values

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Example hyperparameters to experiment with
hyperparameters = {
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [None, 5, 10],
    'min_impurity_decrease': [0.0, 0.1, 0.2]
}

# Grid search
grid_search = GridSearchCV(DecisionTreeRegressor(), hyperparameters, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best hyperparameters found
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model 
best_reg = grid_search.best_estimator_
best_reg.fit(X_train, y_train)

# View the tree
dot_data = export_graphviz(best_reg, precision=2,
                           feature_names=predictors,
                           proportion=True,
                           filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  

# Make predictions and compute error
y_predict = best_reg.predict(X_test)
errors = y_test - y_predict
rmse = np.sqrt((errors ** 2).mean())
print('Test RMSE with Best Hyperparameters: {:.2f}'.format(rmse))
