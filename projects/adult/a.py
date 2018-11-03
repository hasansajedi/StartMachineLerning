import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
from numpy.f2py.tests import test_size
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import sklearn.tree as tree
import matplotlib.pyplot as plt
import seaborn as sns

'''
Abstract: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
'''

columns_list = ['raw', 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'result']
original_data = pd.read_csv('adult.data', sep='\t', names=columns_list, na_values="?", engine='python')
original_data = original_data.drop(['raw'], axis=1)

fig = plt.figure(figsize=(20, 15))
cols = 5
rows = np.ceil(float(original_data.shape[1]) / cols)
# for i, column in enumerate(original_data.columns):
#     ax = fig.add_subplot(rows, cols, i+1)
#     ax.set_title(column)
#     if original_data.dtypes[column]==np.object:
#         original_data[column].value_counts().plot(kind="bar", axes=ax)
#     else:
#         original_data[column].hist(axes=ax)
#         plt.xticks(rotation='vertical')
# plt.subplots_adjust(hspace=0.7, wspace=0.2)
print(original_data.shape[0])
print(original_data.shape[1])
print((original_data["native-country"].value_counts() / original_data.shape[0]).head())


# Encode the categorical features as numbers
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


# Calculate the correlation and plot it
encoded_data, _ = number_encode_features(original_data)
# sns.heatmap(encoded_data.corr(), square=True)
# plt.show()

# print(original_data[["education", "education-num"]].head(15))

del original_data["education"]
print(original_data[["sex", "relationship"]].head(15))

X = encoded_data.drop(['result'], axis=1)
y = encoded_data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("float64"))

encoded_data, encoders = number_encode_features(original_data)

cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(12, 12))
# plt.subplot(2, 1, 1)
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["result"].classes_, yticklabels=encoders["result"].classes_)
# plt.ylabel("Real value")
# plt.xlabel("Predicted value")
print("F1 score: %f" % skl.metrics.f1_score(y_test, y_pred))
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort_index(ascending=True)
# plt.subplot(2, 1, 2)
# coefs.plot(kind="bar")
# plt.show()

binary_data = pd.get_dummies(original_data)
# Let's fix the result as it will be converted to dummy vars too
binary_data["result"] = binary_data["result_>50K"]
del binary_data["result_<=50K"]
del binary_data["result_>50K"]
plt.subplots(figsize=(20, 20))
sns.heatmap(binary_data.corr(), square=True)
plt.show()

# def change_result(x):
#     if x == ' <=50K':
#         return 1
#     else:
#         return 0
#
#
# df['result'] = df['result'].apply(lambda x: change_result(x))
#
# from sklearn import preprocessing
#
# x = df.values
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)
# print(df.head())
# print('------------------------------------------- START HEAD')
# print(df.head())
# print('------------------------------------------- START INFO')
# print(df.info())
# print('------------------------------------------- START DESCRIBE')
# print(df.describe())

# sns.distplot(df['age'], kde=False, bins=120)
# res = df['result'].unique()


# sns.countplot(x=df['result'])
# sns.violinplot(x='result', y='age', data=df)
# sns.violinplot(x='result', y='age', hue='sex', data=df)
# sns.violinplot(x='result', y='age', hue='race', data=df)
# sns.violinplot(x='result', y='age', hue='education', data=df)
# sns.pairplot(df, hue='sex', palette='coolwarm')
# sns.jointplot(x='native-country', y='age', data=df)

from sklearn.model_selection import train_test_split

# X = df
# y = df['result']
# X_train, X_test, y_train, y_test = train_test_split()


plt.show()
