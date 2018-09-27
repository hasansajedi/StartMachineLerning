import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
# print(tips.head())
# print(flights.head())

tc = tips.corr()
# sns.heatmap(tc, annot=True, cmap='coolwarm')

fp = flights.pivot_table(index='month',columns='year', values='passengers')

# sns.heatmap(fp)
# sns.heatmap(fp, cmap='coolwarm', linecolor='white', linewidths=1)

sns.clustermap(fp, cmap='coolwarm', standard_scale=1)

plt.tight_layout()
plt.show()
