# https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RMSE = pd.DataFrame(columns=['method', 'result'])
# Importing data
df = pd.read_csv('data/train.csv')

# Subsetting the dataset
# Index 11856 marks the end of year 2013
df = pd.read_csv('data/train.csv', nrows=11856)

# Creating train and test set
# Index 10392 marks the end of October 2013
train = df[0:10392]
test = df[10392:]

# Aggregating the dataset at daily level
df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.index = df['Timestamp']
df = df.resample('D').mean()

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']
train = train.resample('D').mean()

test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
test.index = test['Timestamp']
test = test.resample('D').mean()

'''
Method 1: Start with a Naive Approach
Now we will implement the Naive method to forecast the prices for test data.
'''
dd = np.asarray(train.Count)
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - 1]
# plt.figure(figsize=(12, 8))
# plt.plot(train.index, train['Count'], label='Train')
# plt.plot(test.index, test['Count'], label='Test')
# plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
# plt.legend(loc='best')
# plt.title("Naive Forecast")
# plt.show()

# We will now calculate RMSE to check to accuracy of our model on test data set.
from sklearn.metrics import mean_squared_error
from math import sqrt

RMSE = RMSE.append({"method": 'Naive Approach', "result": sqrt(mean_squared_error(test.Count, y_hat.naive))},
                   ignore_index=True)
# END METHOD#1

'''
Method 2: – Simple Average
Consider the graph given below. Let’s assume that the y-axis depicts the price of a coin and x-axis depicts the time (days).
'''
y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train['Count'].mean()
# plt.figure(figsize=(12, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
# plt.legend(loc='best')
# plt.show()
RMSE = RMSE.append({"method": 'Simple Average', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.avg_forecast))},
                   ignore_index=True)

# END METHOD#2

'''
Method 3 – Moving Average
Consider the graph given below. Let’s assume that the y-axis depicts the price of a coin and x-axis depicts the time(days).
'''
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['Count'].rolling(60).mean().iloc[-1]
# plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
# plt.legend(loc='best')
# plt.show()

RMSE = RMSE.append(
    {"method": 'Moving Average', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.moving_avg_forecast))},
    ignore_index=True)
# END METHOD#3

'''
Method 4 – Simple Exponential Smoothing
'''

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6, optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
# plt.figure(figsize=(16,8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['SES'], label='SES')
# plt.legend(loc='best')
# plt.show()

RMSE = RMSE.append(
    {"method": 'Simple Exponential Smoothing', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.SES))},
    ignore_index=True)
# END METHOD#4

'''
Method 5 – Holt’s Linear Trend method
'''
import statsmodels.api as sm

# sm.tsa.seasonal_decompose(train.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count)
# plt.show()

y_hat_avg = test.copy()

fit1 = Holt(np.asarray(train['Count'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(test))

# plt.figure(figsize=(16,8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
# plt.legend(loc='best')
# plt.show()

RMSE = RMSE.append(
    {"method": 'Holt’s Linear Trend method', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_linear))},
    ignore_index=True)
'''
Method 6 – Holt-Winters Method
'''
y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
# plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
# plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
# plt.legend(loc='best')
# plt.show()

RMSE = RMSE.append(
    {"method": 'Holt-Winters Method', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.Holt_Winter))},
    ignore_index=True)
'''
Method 7 – ARIMA
'''
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
plt.figure(figsize=(16, 8))
# plt.plot(train['Count'], label='Train')
# plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

RMSE = RMSE.append({"method": 'ARIMA', "result": sqrt(mean_squared_error(test.Count, y_hat_avg.SARIMA))},
                   ignore_index=True)

print(RMSE.head(10))
