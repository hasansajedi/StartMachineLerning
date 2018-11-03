import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris = sns.load_dataset('iris')
# Setosa is the most separable.
# sns.pairplot(iris, hue='species', palette='Dark2')

# Create a kde plot of sepal_length versus sepal width for setosa species of flower.
setosa = iris[iris['species'] == 'setosa']
# sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap="plasma", shade=True, shade_lowest=False)

# Train Test Split
from sklearn.model_selection import train_test_split

X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a Model
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
## Model Evaluation
predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

## Gridsearch Practice
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_params_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

plt.show()
