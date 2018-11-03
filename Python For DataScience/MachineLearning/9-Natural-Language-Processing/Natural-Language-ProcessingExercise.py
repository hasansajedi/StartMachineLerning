import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

yelp = pd.read_csv('yelp.csv')
# print(yelp.head())
yelp['text length'] = yelp['text'].apply(len)
# print(yelp['text length'].head(2))

# g = sns.FacetGrid(data=yelp, col='stars')
# g.map(plt.hist, 'text length')

# sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')
# sns.countplot(x='stars', data=yelp, palette='rainbow')

# Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:
stars = yelp.groupby('stars').mean()

# Use the corr() method on that groupby dataframe to produce this dataframe:
# print(stars.corr())

# Then use seaborn to create a heatmap based off that .corr() dataframe:
# sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)

# NLP Classification Task
# Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.
yelp_class = ((yelp[yelp.stars == 1]) | (yelp[yelp.stars == 5]))
print(yelp_class)

# Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)
X = yelp_class['text'] # Features
y = yelp_class['stars'] # Target / Labels

# Import CountVectorizer and create a CountVectorizer object.
from sklearn.feature_extraction.text import CountVectorizer

# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.
cv = CountVectorizer()

# Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.
X = cv.fit_transform(X)

'''
Train Test Split
Let's split our data into training and testing data.
Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

'''
Training a Model
Time to train a model!
Import MultinomialNB and create an instance of the estimator and call is nb
'''
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Now fit nb using the training data.
nb.fit(X_train, y_train)

'''
Predictions and Evaluations
Time to see how our model did!
Use the predict method off of nb to predict labels from X_test.
'''
predictions = nb.predict(X_test)

# Create a confusion matrix and classification report using these predictions and y_test
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))



plt.show()
