import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/train.csv', low_memory=False)

# print(data.head())
# print(data.describe())
# print(data.dtypes)

# To list all different values of the column StateHoliday run the following code:
# print(data.StateHoliday.unique())
# StateHoliday has both 0 as an integer and a string.
data.StateHoliday = data.StateHoliday.astype(str)

# def count_unique(column):
#     return len(column.unique())
# data.apply(count_unique, axis=0).astype(np.int32)

# Check for missing values
data.isnull().any()

store_data = data[data['Store'] == 160].sort_values('Date')
# plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
# plt.plot(store_data.Sales.values[:365])

# plt.figure(figsize=(20, 10))
# plt.scatter(x=store_data[data.Open==1].Promo, y=store_data[data.Open==1].Sales, alpha=0.1)

transformed_data = data.drop(['Store', 'Date', 'Customers'], axis=1)
transformed_data = pd.get_dummies(transformed_data, columns=['DayOfWeek', 'StateHoliday'])

X = transformed_data.drop(['Sales'], axis=1).values
y = transformed_data.Sales.values
print("The training dataset has {} examples and {} features.".format(X.shape[0], X.shape[1]))

from sklearn.linear_model import LinearRegression
from sklearn import model_selection as cv

lr = LinearRegression()
kfolds = cv.KFold(X.shape[0], shuffle=True, random_state=42)
scores = cv.cross_val_score(lr, X, y, cv=kfolds)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

lr = LinearRegression()
X_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values
y_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).Sales.values
lr.fit(X_store, y_store)
y_store_predict = lr.predict(pd.get_dummies(store_data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values)

plt.figure(figsize=(20, 10))  # Set figsize to increase size of figure
plt.plot(store_data.Sales.values[:365], label="ground truth")
plt.plot(y_store_predict[:365], c='r', label="prediction")
plt.legend()


plt.show()
