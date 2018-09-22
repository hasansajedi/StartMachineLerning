import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101)
df = pd.DataFrame(randn(5, 4), ['A', 'B', 'C', 'D', 'E'], ['W', 'X', 'Y', 'Z'])

# print date frame values
print(df)

# print value of column 'W' in data frame
print(df['W'])

# print the type of data frame
print(type(df))

# print value of data frame Filtered for W and Z column
print(df[['W', 'Z']])

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

# conditional selection
booldf = df > 0
print(booldf)

print(df[booldf])
print(df[df > 0])

# select rows where column W value bigger than zero
print(df[df['W'] > 0])
# select rows where column W value lesser than zero
print(df[df['W'] < 0])

resultdf = df[df['W'] > 0]
print(resultdf['X'])
print(df[df['W'] > 0]['X'])
print(df[df['W'] > 0][['Y', 'X']])

# Use and operator for more than one condition
print(df[(df['W'] > 0) & (df['Y'] > 1)]) # AND
print(df[(df['W'] > 0) | (df['Y'] > 1)]) # OR

newind = 'CA NY WY OR CO'.split()
df['states'] = newind
print(df)
print(df.set_index('states'))



