import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

if not sys.warnoptions:
    warnings.simplefilter("ignore")

##################### START READ_DATA #####################

# Read data from csv file
df = pd.read_csv('data/creditcard.csv')
# print(df.head(2))
# df.info()
# print(df.describe())

byClass = df.groupby('Class')
count = df.groupby(["Class"])["Time"].count()

# sns.countplot(x='Class', data=df)
# sns.barplot(x='Class', y='Amount', data=df)

##################### END READ_DATA #####################

##################### START Model #####################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Only use the 'Amount' and 'V1', ..., 'V28' features
features = ['Amount'] + ['V%d' % number for number in range(1, 29)]
print(features)

# The target variable which we would like to predict, is the 'Class' variable
target = 'Class'

# Now create an X variable (containing the features) and an y variable (containing only the target variable)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Define the model
model = LogisticRegression()

# Fit and predict!
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# And finally: show the results
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

##################### END Model #####################


plt.show()
