# https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/

import warnings  # `do not disturbe` mode

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt

RMSE = pd.DataFrame(columns=['method', 'result'])

# Importing data
df_test = pd.read_csv('data/test.csv')

# Subsetting the dataset
# Index 11856 marks the end of year 2013
df = pd.read_csv('data/train.csv')

# Creating train and test set
# Index 10392 marks the end of October 2013
train = df[0:]
# train = df[0:10392]
# test = df[10392:]

# Aggregating the dataset at daily level
df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.index = df['Timestamp']

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']

df_test['Timestamp'] = pd.to_datetime(df_test['Datetime'], format='%d-%m-%Y %H:%M')
df_test.index = df_test['Timestamp']
test = df_test

y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()
y_hat_avg['Count'] = fit1.predict(start="00:00 26-09-2014", end="00:00 26-04-2015", dynamic=True, full_results=True)
print(y_hat_avg.head())

# RMSE = RMSE.append({"method": 'SARIMA', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.Count))},
#                    ignore_index=True)

# print("------------------------ RESULT ------------------------")
# print(fit1.summary().tables[1])

# print(y_hat_avg.columns)
y_hat_avg = y_hat_avg.drop(['Timestamp','Datetime'], axis=1)
y_hat_avg.to_csv('result.csv', index=None)


# fit1.plot_diagnostics(figsize=(15, 12))
# plt.show()


# pred = fit1.get_prediction(start=pd.to_datetime('09-2014'), dynamic=False)
# pred_ci = pred.conf_int()
# print(pred.summary)


# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color='k', alpha=.2)


# plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(pred.predicted, label='SARIMA')
# plt.legend(loc='best')
# plt.show()
