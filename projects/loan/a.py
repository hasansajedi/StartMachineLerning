import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams["patch.force_edgecolor"] = True

loans = pd.read_csv('LoanStats3a1.csv')
print(loans.info())
print(loans.describe())
print(loans.head())
print(help(pd.DataFreame.read_csv))
loans['loan_status'].value_counts()

sns.countplot(x='not.fully.paid',data=loans)

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=35,alpha=0.5,label='credit.policy=1')
loans[loans['credit.policy']==0]['fico'].hist(bins=35,alpha=0.5,label='credit.policy=0')
plt.xlabel('FICO Score')
plt.legend()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,alpha=0.5,label='not.fully.paid=1')
loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,alpha=0.5,label='not.fully.paid=0')
plt.xlabel('FICO Score')
plt.legend()

sns.jointplot(x='fico',y='int.rate',data=loans,color='purple',size=8)

sns.lmplot(x='fico',y='int.rate',data=loans,col='not.fully.paid',hue='credit.policy',palette='Set1')

loans.info()

cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()

from sklearn.model_selection import train_test_split
final_data.columns
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# import Logistic Regression model
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train,y_train)
log_predictions = log_model.predict(X_test)

# import decision tree model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=101)
dtree.fit(X_train,y_train)
dtree_predictions = dtree.predict(X_test)

# import random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000,random_state=101)
rfc.fit(X_train,y_train)
rfc_predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test,log_predictions))
print('\n')
print('Classification Report:')
print(classification_report(y_test,log_predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test,dtree_predictions))
print('\n')
print('Classification Report:')
print(classification_report(y_test,dtree_predictions))
print('Confusion Matrix:')
print(confusion_matrix(y_test,rfc_predictions))
print('\n')
print('Classification Report:')
print(classification_report(y_test,rfc_predictions))












