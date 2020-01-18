# https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

RMSE = pd.DataFrame(columns=['method', 'result'])

# Importing data
df_test = pd.read_csv('data/test.csv')
df = pd.read_csv('data/train.csv')

# Creating train and test set
train = df[0:]

# Aggregating the dataset at daily level
df['Timestamp'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
df.index = df['Timestamp']
df = df.resample('D').mean()

train['Timestamp'] = pd.to_datetime(train['Datetime'], format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']
train = train.resample('D').mean()

df_test['Timestamp'] = pd.to_datetime(df_test['Datetime'], format='%d-%m-%Y %H:%M')
df_test.index = df_test['Timestamp']
test = df_test.resample('D').mean()

y_hat_avg = test.copy()
# fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False,
#                                  enforce_invertibility=False).fit()
# fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()

# y_hat_avg['Count'] = fit1.predict(start="00:00 26-09-2014", end="00:00 27-04-2015", dynamic=True, full_results=True)
# pred = fit1.predict(start="26-09-2014", end="27-04-2015", dynamic=True, full_results=True)

fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="09-2014", end="04-2015", dynamic=True)

# pred = fit1.get_prediction(start=pd.to_datetime('09-2014'), end=pd.to_datetime('04-2015'), dynamic=True, full_results=True)
# pred_ci = pred.conf_int()
# print(pred.summary)

# y_hat_avg = y_hat_avg.drop(['Timestamp', 'Datetime'], axis=1)
# y_hat_avg.to_csv('result.csv', index=None)

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Trained data')
plt.plot(y_hat_avg['SARIMA'], label='Predicted data')
plt.legend(loc='best')

plt.show()
