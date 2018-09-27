import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
tips.head()

# sns.set_style('darkgrid')
# sns.set_context('poster') # notebook
# sns.countplot(x='sex', data=tips)

sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='winter')

plt.show()
