# Packages
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import plot, show

# Reading the csv files
bank_additional_full_df = pd.read_csv('data/bank-additional/bank-additional-full.csv', sep=';')
bank_additional_df = pd.read_csv('data/bank-additional/bank-additional.csv', sep=';')
bank_full_df = pd.read_csv('data/bank/bank-full.csv', sep=';')
bank_df = pd.read_csv('data/bank/bank.csv', sep=';')

# print('____________________________________ 1 ____________________________________')
# print(bank_additional_full_df.tail(5))
# print('____________________________________ 2 ____________________________________')
# print(bank_additional_df.tail(5))
# print('____________________________________ 3 ____________________________________')
# print(bank_full_df.tail(5))
# print('____________________________________ 4 ____________________________________')
# print(bank_df.tail(5))

# Columns information
# print(bank_additional_full_df.columns)
#
# #size
# print(bank_additional_full_df.shape)
#
# #Info
# print(bank_additional_full_df.info())
#
# #Describe
# print(bank_additional_full_df.describe())

# Checking for null value
# print(bank_additional_full_df.isnull().sum())

# plotting employment variation rate - quarterly indicator emp.var.rate
# plt.rcParams['figure.figsize'] = (8, 6)
# sns.countplot(x='emp.var.rate', hue='emp.var.rate', data=bank_additional_full_df)
# # previous: number of contacts performed before this campaign and for this client (numeric)
# plt.rcParams['figure.figsize'] = (8, 6)
# sns.countplot(x='previous', hue='previous', data=bank_additional_full_df)

# #plot data
# fig, ax = plt.subplots(figsize=(15,7))
# bank_additional_full_df.groupby(['duration']).count()[['education','nr.employed']].plot(ax=ax)
#
# Another way to plot a histogram of duration is shown below
# bank_additional_full_df['duration'].hist(bins=50)

# Describing dummy keys of particular column
y_n_lookup = {'yes': 1, 'no': 0}
bank_additional_full_df['y_dummy'] = bank_additional_full_df['y'].map(lambda x: y_n_lookup[x])
print(bank_additional_full_df['y_dummy'].value_counts())

# getting marital status of groupby people
age_group_names = ['young', 'lower middle', 'middle', 'senior']
bank_additional_full_df['age_binned'] = pd.qcut(bank_additional_full_df['age'], 4, labels=age_group_names)
print(bank_additional_full_df['age_binned'].value_counts())

gb_marital_age = bank_additional_full_df['y_dummy'].groupby(
    [bank_additional_full_df['marital'], bank_additional_full_df['age_binned']])
print(gb_marital_age.mean())

# getting life stage of age group
bank_additional_full_df['life_stage'] = bank_additional_full_df.apply(lambda x: x['age_binned'] + ' & ' + x['marital'],
                                                                      axis=1)
print(bank_additional_full_df['life_stage'].value_counts())

from sklearn import preprocessing
from sklearn import cluster

# getting the pattern of particular age range employee
combined_data = bank_additional_full_df[['age', 'nr.employed']].as_matrix()
combined_data_scaled = preprocessing.scale(combined_data)

# Applying KMeans algorithm
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(combined_data_scaled)
y_pred = kmeans.predict(combined_data_scaled)
# Plotting the graph
plt.scatter(combined_data_scaled[:, 0], combined_data_scaled[:, 1], c=y_pred)
plt.xlabel('Scaled Age')
plt.ylabel('Scaled  number of employees')

plt.show()
