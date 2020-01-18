import pandas as pd
import numpy as np

# replace file path
df = pd.read_csv("train.csv")

# data structure checking

# get the test sample size
print(df.shape)

# check null value in the sample
print(df.apply(lambda x: sum(x.isnull()), axis=0))

# survival counts, 1=survived, 0=died
print(df['survived'].value_counts(ascending=True))

# sex and survived relationship
sex_stats = pd.pivot_table(df, values='survived', index=['sex'], aggfunc=lambda x: x.mean())
print(sex_stats)

# Class and survived relationship
class_stats = pd.pivot_table(df, values='survived', index=['pclass'], aggfunc=lambda x: x.mean())
print(class_stats)

# Class, sex and survived relationship
class_sex_stats = pd.pivot_table(df, values='survived', index=['pclass', 'sex'], aggfunc=lambda x: x.mean())
print(class_sex_stats)

# Class, sex and survived in graph
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x="pclass", y="survived", hue="sex", data=df)

plt.show()

# familySize in graph
df['FamilySize'] = df['sibsp'] + df['parch'] + 1

sns.barplot(x="FamilySize", y="survived", data=df)

plt.show()

# check fare distribution
fare_dist = sns.distplot(df['fare'], label="Skewness : %.2f" % (df["fare"].skew()))
fare_dist.legend(loc="best")

plt.show()

# logarithm the fare
df['fare_log'] = df["fare"].map(lambda i: np.log(i) if i > 0 else 0)

fare_dist_w_log = sns.distplot(df['fare_log'], label="Skewness : %.2f" % (df["fare_log"].skew()))
fare_dist_w_log.legend(loc="best")

plt.show()

# cut into group according to its description
df['fare_log'].describe()

bins = (-1, 2, 2.67, 3.43, 10)
group_names = [1, 2, 3, 4]
categories = pd.cut(df['fare_log'], bins, labels=group_names)
df['fareGroup'] = categories

print(df['fareGroup'].value_counts(ascending=True))

# start prediction

import pandas as pd
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


def setfamilygroup(df):
    # set people into parentchild and spousesib groups
    df['withP'] = 0
    df['withS'] = 0

    df.loc[train_df['sibsp'] > 0, 'withS'] = 1
    df.loc[train_df['parch'] > 0, 'withP'] = 1

    # handle family group
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1

    bins = (-1, 1, 2, 3, 12)
    group_names = [1, 2, 3, 4]
    categories = pd.cut(df['FamilySize'], bins, labels=group_names)
    df['familygroup'] = categories


def setageGroup(df):
    # fill up NaN age according class / sibsp / parch
    index_NaN_age = list(df["age"][df["age"].isnull()].index)

    for i in index_NaN_age:
        age_mean = df["age"].mean()
        age_std = df["age"].std()
        age_pred_w_spc = df["age"][((df['sibsp'] == df.iloc[i]["sibsp"]) & (df['parch'] == df.iloc[i]["parch"]) & (
                    df['pclass'] == df.iloc[i]["pclass"]))].mean()
        age_pred_wo_spc = np.random.randint(age_mean - age_std, age_mean + age_std)

        if not np.isnan(age_pred_w_spc):
            df['age'].iloc[i] = age_pred_w_spc
        else:
            df['age'].iloc[i] = age_pred_wo_spc

            # separate age into 6 groups
    bins = (-1, 15, 23, 33, 43, 53, 100)
    group_names = [1, 2, 3, 4, 5, 6]
    categories = pd.cut(df['age'], bins, labels=group_names)
    df['ageGroup'] = categories


def setfareGroup(df):
    # fill the missing fare with median
    df["fare"] = df["fare"].fillna(df["fare"].median())

    df['fare_log'] = df["fare"].map(lambda i: np.log(i) if i > 0 else 0)

    bins = (-1, 2, 2.68, 3.44, 10)
    group_names = [1, 2, 3, 4]
    categories = pd.cut(df['fare_log'], bins, labels=group_names)
    df['fareGroup'] = categories


def settitle(df):
    df['title'] = df['name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    df["title"] = df["title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')
    df['title'] = df['title'].replace('Mlle', 'Miss')
    df['title'] = df['title'].replace('Ms', 'Miss')
    df['title'] = df['title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['title'] = df['title'].map(title_mapping)


def getmanipulatedDF(train_df, test_df):
    dfs = [train_df, test_df]

    for df in dfs:
        df["sex"] = df["sex"].map({"male": 1, "female": 0})

        setfamilygroup(df)

        setageGroup(df)

        setfareGroup(df)

        # fill up the missing 2 embarked
        df['embarked'] = df['embarked'].fillna('S')
        # map into value
        df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # converse categories into int
        df['familygroup'] = df['familygroup'].astype(int)
        df['ageGroup'] = df['ageGroup'].astype(int)
        df['fareGroup'] = df['fareGroup'].astype(int)

        settitle(df)

    return dfs[0], dfs[1]


train_df, test_df = getmanipulatedDF(train_df, test_df)

# fare_log, survived check
facet = sns.FacetGrid(train_df, hue="survived", aspect=4)
facet.map(sns.kdeplot, 'fare_log', shade=True)
facet.set(xlim=(0, train_df['fare_log'].max()))
facet.add_legend()

plt.show()

X_learning = train_df.drop(['name', 'cabin', 'sibsp', 'parch', 'fare', 'survived', 'ticket'], axis=1)
Y_learning = train_df['survived']

X_test = test_df.drop(['name', 'cabin', 'sibsp', 'parch', 'fare', 'ticket'], axis=1)

# use Kfold validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

random_state = 19

models = []
models.append(("RFC", RandomForestClassifier(random_state=random_state)))
models.append(("ETC", ExtraTreesClassifier(random_state=random_state)))
models.append(("ADA", AdaBoostClassifier(random_state=random_state)))
models.append(("GBC", GradientBoostingClassifier(random_state=random_state)))
models.append(("SVC", SVC(random_state=random_state)))
models.append(("LoR", LogisticRegression(random_state=random_state)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("QDA", QuadraticDiscriminantAnalysis()))
models.append(("DTC", DecisionTreeClassifier(random_state=random_state)))
models.append(("XGB", xgb.XGBClassifier()))

from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10)

k_names = []
k_means = []
k_stds = []

for name, model in models:
    # cross validation among models, score based on accuracy
    cv_results = model_selection.cross_val_score(model, X_learning, Y_learning, scoring='accuracy', cv=kfold)
    print("\n" + name)
    # print("Results: "+str(cv_results))
    print("Mean: " + str(cv_results.mean()))
    print("Standard Deviation: " + str(cv_results.std()))
    k_names.append(name)
    k_means.append(cv_results.mean())
    k_stds.append(cv_results.std())

# display the result
kfc_df = pd.DataFrame({"CrossValMeans": k_means, "CrossValerrors": k_stds, "Algorithm": k_names})

sns.barplot("CrossValMeans", "Algorithm", data=kfc_df, orient="h", **{'xerr': k_stds})

# Using XGBoost
xgbc = xgb.XGBClassifier()
xgbc.fit(X_learning, Y_learning)
predictions = xgbc.predict(X_test)

output = pd.DataFrame({'passengerid': test_df['passengerid'], 'survived': predictions})

# replace file path
output.to_csv("Development/Playground/Titanic/out_xgb.csv", index=False)
