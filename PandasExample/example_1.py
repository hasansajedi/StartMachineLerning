import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)
df = pd.DataFrame(randn(5, 4), ['A','B','C','D','E'], ['W','X','Y','Z'])

# print date frame values
print(df)

# print value of column 'W' in data frame
print(df['W'])

# print the type of data frame
print(type(df))

# print value of data frame Filtered for W and Z column
print(df[['W','Z']])

# add column to data frame with new value
df['new'] = df['W'] + df['Y']
print(df)

df = (df.drop('new', axis=1))
print(df)

# print data frame dimencial
print(df.shape)

# print values for row A
print(df.loc['A'])
print(df.iloc[0])
print(df.loc['B', 'Y'])
print(df.loc[['A', 'B'], ['W', 'Y']])
