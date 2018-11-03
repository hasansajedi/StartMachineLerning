import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

################################# 001 #################################
df = pd.read_csv('USA_Housing.csv')
# df.head()
# df.info()
# df.describe()
# print(df.columns)

# sns.pairplot(df)
# sns.distplot(df['Price'])
# sns.heatmap(df.corr(), annot=True)

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

################################# 002 #################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

################################# 003 #################################
# START CALCULATE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
# print(lm.intercept_)
# print(lm.coef_)
# print(X_train.columns)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])

cdf

################################# 003 #################################
predictions = lm.predict(X_test)
# print(predictions)
# plt.scatter(y_test, predictions)
sns.distplot((y_test - predictions))

from sklearn import metrics
metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test, predictions)


# END CALCULATE LINEAR REGRESSION


plt.show()
