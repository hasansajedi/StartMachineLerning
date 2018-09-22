import numpy as np
import pandas as pd

from numpy.random import randn

d = {'A': [1, 2, np.nan], 'B': [5, np.nan, np.nan], 'C': [1, 2, 3]}
df = pd.DataFrame(d)
print(df)

print(df.dropna())  # Rows
print(df.dropna(axis=1))  # Cols

df.fillna(value='Fill value')

