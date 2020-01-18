import datetime

import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
import matplotlib.dates as mdates
import xlwt

# read data from csv
df = pd.read_excel('data/input.xls', encoding="utf-8")
columns = df.columns.values
indexes = df.index.values


def compute_percentage(x):
    pct = float(x / df['count'].sum()) * 100
    return round(pct, 3)

# set date as index
df['device'] = df['device'].apply(str)
df['date'] = df['date'].apply(str)
df = df.sort_values(by=['count'], ascending=False)

df = df.drop(df[df['device'] =='33330000'].index)

devices = {'33330001': 'AP', '33330002': '780', '33330003': 'TOP', '33330004': '3SOT',
           '33330005': 'PAT', '33330007': 'BP', '33330008': 'JIRING'}
# df['device'] = df['device'].replace(devices)

splited = pd.DataFrame(df.groupby(['device'])[["count"]].sum())
splited['percentage'] = splited.apply(compute_percentage, axis=1)

# Convert date from string to date times
df['month'] = df['date'].apply(lambda x: x[:7].replace('-',''))

byMoth = pd.DataFrame(df.groupby(['month', 'device'])[["count"]].sum())

def compute_percentage1(x):
    pct = float(x / byMoth['count'].sum()) * 100
    return round(pct, 2)

byMoth['percentage'] = byMoth.apply(compute_percentage1, axis=1)
# byMoth.to_excel('byMonth1.xls')

byMoth = byMoth.reset_index()
byMoth = pd.get_dummies(byMoth)

# Build the correlation matrix
matrix = byMoth.corr()
# f, ax = plt.subplots(figsize=(16, 12))
# sns.heatmap(matrix, vmax=0.7, square=True)

# interesting_variables = matrix['count'].sort_values(ascending=False)
# Filter out the target variables (SalePrice) and variables with a low correlation score (v such that -0.6 <= v <= 0.6)
# interesting_variables = interesting_variables[abs(interesting_variables) >= 0.6]

# print(df.groupby(['device'])[["count"]].sum())
# sns.scatterplot(x='device',y='percentage', data=splited)
# g = sns.jointplot(x="device", y="count", data=df, kind='kde', color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6)
# sns.barplot(x='device',y='count', data=df)
# sns.countplot(x='device', data=df)
# sns.scatterplot(x='date', y='count', hue='device',data=df)
# sns.pairplot(data=df, kind="reg")
# sns.heatmap(df.corr(), cmap='coolwarm', linecolor='white', linewidths=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = byMoth.drop(['count'], axis=1)
y = byMoth['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(y_test)
print(predictions)

from sklearn.metrics import classification_report, confusion_matrix, f1_score
# print(classification_report(y_test, predictions))

plt.show()
