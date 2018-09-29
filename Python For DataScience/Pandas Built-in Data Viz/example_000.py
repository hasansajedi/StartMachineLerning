import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import adjusted_figsize

from numpy.random import randn

df1 = pd.read_csv('df1', index_col=0)
print(df1.head())

df2 = pd.read_csv('df2')
print(df2.head())

# df1['A'].hist()
# df1['A'].plot(kind='hist', bins=30)
# df1['A'].plot.hist()

# df2.plot.area(alpha=0.4)
# df2.plot.bar()
# df2.plot.bar(stacked=True)

# df1.plot.line(x=df1.index, y='B', figsize=(12,3), lw=1)
# df1.plot.scatter(x='A', y='B',c='C',cmap='coolwarm')

# df2.plot.box()

df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
# df.plot.hexbin(x='a', y='b')
# df.plot.hexbin(x='a', y='b', gridsize=25)
# df.plot.hexbin(x='a', y='b', gridsize=25,cmap='coolwarm')
df2['a'].plot.kde()
df2.plot.density()
df2['a'].plot.density()


plt.show()
