#1 high flexibility of data with properties and structures

#2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)

#3
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})

#4
df.info()
df.describe()

#5
predictors = ['Outstate', 'F.Undergrad']
X = df[predictors].values
y = df['Private'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#6
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

#7
target_names = ['No', 'Yes']
dot_data = export_graphviz(clf, precision=2,
                 	feature_names=predictors,  
                 	proportion=True,
                 	class_names=target_names,  
                 	filled=True, rounded=True,  
                 	special_characters=True)

graph = graphviz.Source(dot_data)  
graph

#8
y_pred = clf.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)