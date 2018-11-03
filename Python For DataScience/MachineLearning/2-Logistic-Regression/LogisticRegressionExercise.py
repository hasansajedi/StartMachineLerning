import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Read in the advertising.csv file and set it to a data frame called ad_data.
ad_data = pd.read_csv('advertising.csv')
# ad_data.head()
# ad_data.info()
# ad_data.describe()

# sns.distplot(ad_data['Age'], kde=False, bins=30)
# sns.jointplot(y='Area Income', x='Age', data=ad_data)
# sns.jointplot(y='Daily Time Spent on Site', x='Age', data=ad_data, color='red', kind='kde')
# sns.jointplot(y='Daily Internet Usage', x='Daily Time Spent on Site', data=ad_data, color='green')
# sns.pairplot(ad_data, palette='bwr')

from sklearn.model_selection import train_test_split

ad_data.head(2)

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


plt.show()
