import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')

# loans.head()
# loans.info()

# plt.figure(figsize=(10, 6))
# loans[loans['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue',
#                                                 bins=30, label='Credit.Policy=1')
# loans[loans['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red',
#                                                 bins=30, label='Credit.Policy=0')
# plt.legend()
# plt.xlabel('FICO')

# plt.figure(figsize=(10, 6))
# loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, color='blue',
#                                                 bins=30, label='not.fully.paid=1')
# loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, color='red',
#                                                 bins=30, label='not.fully.paid=0')
# plt.legend()
# plt.xlabel('FICO')

# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
# sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')

# Let's see the trend between FICO score and interest rate. Recreate the following jointplot.
# sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')

'''
Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. 
Check the documentation for lmplot() if you can't figure out how to separate it into columns.
'''
# sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')

# loans.info()
cat_feats = ['purpose']
'''
Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. 
Set this dataframe as final_data.
'''
final_data = pd.get_dummies(data=loans, columns=cat_feats, drop_first=True)
# loans.info()

'''
## Train Test Split
Now its time to split our data into a training set and a testing set!
Use sklearn to split your data into a training set and a testing set as we've done in the past.
'''
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis=1)
y  = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y)

'''
Training a Decision Tree Model
Let's start by training a single decision tree first!
'''
from sklearn.tree import DecisionTreeClassifier #Import DecisionTreeClassifier
# Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

'''
## Training the Random Forest model
Now its time to train our model!
Create an instance of the RandomForestClassifier class and fit it to our training data from the previous
'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))





plt.show()
