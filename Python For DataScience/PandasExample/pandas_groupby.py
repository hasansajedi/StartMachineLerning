# Groupby

import pandas as pd

# Create dataframe
data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
        'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
        'Sales': [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)
print(df)

byComp = df.groupby('Company')
print(byComp.mean())
print(byComp.sum())
print(byComp.std())
print(byComp.count())
print(byComp.max())
print(byComp.min())

print(byComp.sum().loc['FB'])

print(byComp.describe())
print(byComp.describe().transpose())
print(byComp.describe().transpose()['FB'])
