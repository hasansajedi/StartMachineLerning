import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs

df = pd.read_csv('College_Data', index_col=0)
df.head()

# sns.scatterplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private')

# Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
# sns.set_style('whitegrid')
# sns.lmplot('Outstate', 'F.Undergrad', data=df, hue='Private', palette='coolwarm', height=6, aspect=1, fit_reg=False)

'''
Create a stacked histogram showing Out of State Tuition based on the Private column. 
Try doing this using sns.FacetGrid. If that is too tricky, see if you can do it just by using two instances of pandas.plot(kind='hist').
'''
# sns.set_style('darkgrid')
# g = sns.FacetGrid(df, hue="Private", palette='coolwarm', height=6, aspect=2)
# g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)

# Create a similar histogram for the Grad.Rate column.
# sns.set_style('darkgrid')
# g = sns.FacetGrid(df, hue="Private", palette='coolwarm', size=6, aspect=2)
# g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

# Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that school?
# print(df[df['Grad.Rate'] > 100])

## K Means Cluster Creation
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
# Fit the model to all the data except for the Private label.
kmeans.fit(df.drop('Private', axis=1))
# What are the cluster center vectors?
print(kmeans.cluster_centers_)

'''
Evaluation
There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, 
we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
'''


def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(converter)
print(df.head())

# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))

plt.show()
