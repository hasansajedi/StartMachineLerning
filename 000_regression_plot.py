import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

sns.lmplot(data=tips, x='total_bill', y='tip', hue='sex', markers=['o', 'v'], )
sns.lmplot(data=tips, x='total_bill', y='tip', col='day', row='time')
sns.lmplot(data=tips, x='total_bill', y='tip', col='day', row='time', hue='sex')
sns.lmplot(data=tips, x='total_bill', y='tip', col='day', hue='sex')
sns.lmplot(data=tips, x='total_bill', y='tip', col='day', hue='sex', aspect=0.6, height=8)

plt.show()
