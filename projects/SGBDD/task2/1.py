import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

### Import factura data as data frame ###
# file "DataScienceAufgabe2_FacturaData" is located in folder "/DataScienceAufgabe2/Data/".
# In order to use it in jupyter notebook you have to upload it (See "Upload" Button in http://localhost:8889/tree) )

df = pd.read_csv("data/DataScienceAufgabe2_FacturaData.csv", encoding="iso-8859-1", decimal=",", sep=";")
print('{:,} rows; {:,} columns'.format(df.shape[0], df.shape[1]))

matrix = df.pivot_table(index=['customerNumber'], columns=['articleNumber'], values='purchaseAmount')
matrix = matrix.fillna(0).reset_index()
x_cols = matrix.columns[1:]

from sklearn.cluster import KMeans

cluster = KMeans(n_clusters=5)
# slice matrix so we only include the 0/1 indicator columns in the clustering
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[2:]])
# print(matrix.cluster.value_counts())

# sns.barplot(x="cluster", y=matrix.index, data=matrix)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:, 0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:, 1]
matrix = matrix.reset_index()

customer_clusters = matrix[['customerNumber', 'cluster', 'x', 'y']]

df = pd.merge(df, customer_clusters)
print(df.head())
# ax = sns.scatterplot(x="x", y="y",
#                      hue="cluster",
#                      sizes=(10, 200),
#                      data=df)
#
# plt.show()
