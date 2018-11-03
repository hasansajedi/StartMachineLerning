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
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style('whitegrid')
'''
Abstract: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.
'''

columns_list = ['raw', 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'result']
df = pd.read_csv('adult.data', sep='\t', names=columns_list, na_values="?", engine='python')
df = df.drop(['raw'], axis=1)

# print('---------------- df.shape[0] ----------------')
# print(df.shape[0])
# print('---------------- df.shape[1] ----------------')
# print(df.shape[1])
fig = plt.figure(figsize=(20, 15))
cols = 5
rows = np.ceil(float(df.shape[
                         1]) / cols)  # The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as \lceil x \rceil.
for i, column in enumerate(df.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if df.dtypes[column] == np.object:
        df[column].value_counts().plot(kind="bar", axes=ax)
    else:
        df[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)
plt.show()


# print('---------------- (df["native-country"].value_counts() / df.shape[0]).head() ----------------')
# print((df["native-country"].value_counts() / df.shape[0]).head())


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
encoded_data, _ = number_encode_features(df)
# print(encoded_data.head(5))
# print(df[["sex", "relationship"]].head(15))
# Calculate the correlation and plot it
sns.heatmap(encoded_data.corr(), cmap='coolwarm', linecolor='white', linewidths=1)
plt.show()
# We see there is a high correlation between Education and Education-Num. Let’s look at these columns
df[["education", "education-num"]].head(15)  # As you can see these two columns actually represent the same features
df.drop(["education"], axis=1)

# Build a classifier
encoded_data, encoders = number_encode_features(df)

# Split and scale the features
X = encoded_data.drop(['result'], axis=1)
y = encoded_data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("float64"))

binary_data = pd.get_dummies(df)
# print(binary_data.head())
# Let's fix the Target as it will be converted to dummy vars too
binary_data["result"] = binary_data["result_ >50K"]
del binary_data["result_ <=50K"]
del binary_data["result_ >50K"]
plt.subplots(figsize=(20, 20))
sns.heatmap(binary_data.corr(), square=True, cmap='coolwarm')
plt.show()

X = binary_data.drop(['result'], axis=1)
y = binary_data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)

cls = linear_model.LogisticRegression()

cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 1)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoders["result"].classes_, yticklabels=encoders["result"].classes_)
plt.ylabel("Real value")
plt.xlabel("Predicted value")
print("F1 score: %f" % skl.metrics.f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
coefs.sort_values(ascending=True, inplace=True)
ax = plt.subplot(2, 1, 2)
coefs.plot(kind="bar")
plt.show()

plt.show()
