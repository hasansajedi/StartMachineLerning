'''
It is your job to predict the sales price for each house.
For each Id in the test set, you must predict the value of the SalePrice variable.
'''
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

################ READ DATA ################
sns.set()
pd.set_option('max_columns', 1000)
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('data/train.csv')

# Get the columns list
print(df.columns)

# Get missing values
num_missing = df.isnull().sum()
percent = num_missing / df.isnull().count()
df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])
df_missing = df_missing.sort_values('Fraction', ascending=False)
print(df_missing[df_missing['MissingValues'] > 0].index)

# Remove columns have null value
variables_to_keep = df_missing[df_missing['MissingValues'] == 0].index
df = df[variables_to_keep]
print(variables_to_keep)

# Variable Analysis
# Here we will do a quick analysis of the variables and the underlying relations. Let’s build a correlation matrix.
# Build the correlation matrix
matrix = df.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(matrix, vmax=0.7, square=True)

interesting_variables = matrix['SalePrice'].sort_values(ascending=False)
# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)
interesting_variables = interesting_variables[abs(interesting_variables) >= 0.6]
interesting_variables = interesting_variables[interesting_variables.index != 'SalePrice']
print(interesting_variables)

values = np.sort(df['OverallQual'].unique())
print('Unique values of "OverallQual":', values)

data = pd.concat([df['SalePrice'], df['OverallQual']], axis=1)
data.plot.scatter(x='OverallQual', y='SalePrice')

cols = interesting_variables.index.values.tolist() + ['SalePrice']
sns.pairplot(df[cols], size=2.5)
plt.show()

# Build the correlation matrix
matrix = df[cols].corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, vmax=1.0, square=True)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[[v for v in interesting_variables.index.values if v != 'SalePrice']]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Build a plot
plt.scatter(y_pred, y_test)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(y_test), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, confusion_matrix, classification_report

print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred))
print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred))
print('confusion_matrix:', confusion_matrix(y_test, y_pred))
print('classification_report:', classification_report(y_test, y_pred))

plt.show()
