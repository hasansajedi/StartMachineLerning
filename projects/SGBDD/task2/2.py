from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

import locale
import seaborn as sns

from utils import change_column_to_date

from projects.SGBDD.task2.utils import print_output

locale.setlocale(locale.LC_ALL, "german")
sns.set()

# Import factura data as data frame
df = pd.read_csv("data/DataScienceAufgabe2_FacturaData.csv", encoding="iso-8859-1", decimal=",", sep=";")

# Change purchaseDate values from 2018-01-01 to 2018-01-01 00:00:00 for process it as a Date
df = change_column_to_date(df, "purchaseDate")

# Get the shape of DataFrame
print('{:,} rows; {:,} columns'.format(df.shape[0], df.shape[1]))

# Get the start time and end time for Data Frame
print('Orders from {} to {}'.format(df['purchaseDate'].min(), df['purchaseDate'].max()))

'''
Get grouped data by 3-important column and the sum of purchaseAmount
I create an order dataframe that will aggregate our factura at the order level.
'''
orders = df.groupby(['articleNumber', 'purchaseDate', 'customerNumber']).agg(
    {'purchaseAmount': lambda x: x.sum()}).reset_index()
# print(orders.head())

'''
Finally, I am going to simulate an analysis I am doing in real time by setting the now date at one day after the last purchase. 
This date will be used as a reference to calculate the Recency score.
'''
now = orders['purchaseDate'].max() + timedelta(days=1)
# print(now)

# I am going to study the data over a period of one year. I set a period variable to 365 (days).
period = 365

# Calculate the Recency, Frequency and Monetary Value of each customers
print(print_output("div", "Calculate the Recency, Frequency and Monetary Value of each customers"))
orders['DaysSinceOrder'] = orders['purchaseDate'].apply(lambda x: (now - x).days)

# The scores are calculated for each customer. I need a dataframe with one row per customer. The scores will be stored in columns.
aggr = {
    'DaysSinceOrder': lambda x: x.min(),  # the number of days since last order (Recency)
    'purchaseDate': lambda x: len([d for d in x if d >= now - timedelta(days=period)]),
    # the total number of orders in the last period (Frequency)
}
rfm = orders.groupby('customerNumber').agg(aggr).reset_index()
rfm.rename(columns={'DaysSinceOrder': 'Recency', 'purchaseDate': 'Frequency'}, inplace=True)
rfm.head()

# I have the Recency and Frequency data. I need to add the Monetary value of each customer by adding sales over the last year.
rfm['Monetary'] = rfm['customerNumber'].apply(
    lambda x: orders[(orders['customerNumber'] == x) & (orders['purchaseDate'] >= now - timedelta(days=period))][
        'purchaseAmount'].sum())
# print(rfm.head())

# Calculate the R, F and M scores
print_output("div", "Calculate the R, F and M scores")
quintiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()
# print(quintiles)

# Then I write methods to assign ranks from 1 to 5. A smaller Recency value is better whereas higher Frequency and Monetary values are better. I need to write two separate methods.
def r_score(x):
    if x <= quintiles['Recency'][.2]:
        return 5
    elif x <= quintiles['Recency'][.4]:
        return 4
    elif x <= quintiles['Recency'][.6]:
        return 3
    elif x <= quintiles['Recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5

# Get the R, F and M scores per each customer
rfm['R'] = rfm['Recency'].apply(lambda x: r_score(x))
rfm['F'] = rfm['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
rfm['M'] = rfm['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

# Get customers segments from RFM score
print_output("div", "Get customers segments from RFM score")
rfm['RFM Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
rfm.head()

segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)
print(rfm.head())

# Visualize our customers segments
print_output("div", "Visualize our customers segments")

# plot the distribution of customers over R and F
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

for i, p in enumerate(['R', 'F']):
    parameters = {'R': 'Recency', 'F': 'Frequency'}
    y = rfm[p].value_counts().sort_index()
    x = y.index
    ax = axes[i]
    bars = ax.bar(x, y, color='blue')
    ax.set_frame_on(False)
    ax.tick_params(left=False, labelleft=False, bottom=False)
    ax.set_title('Distribution of {}'.format(parameters[p]),
                 fontsize=14)
    for bar in bars:
        value = bar.get_height()
        if value == y.max():
            bar.set_color('red')
        ax.text(bar.get_x() + bar.get_width() / 2,
                value - 5,
                '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
                ha='center',
                va='top',
                color='w')

plt.show()

# plot the distribution of M for RF score
fig, axes = plt.subplots(nrows=5, ncols=5,
                         sharex=False, sharey=True,
                         figsize=(10, 10))

r_range = range(1, 6)
f_range = range(1, 6)
for r in r_range:
    for f in f_range:
        y = rfm[(rfm['R'] == r) & (rfm['F'] == f)]['M'].value_counts().sort_index()
        x = y.index
        ax = axes[r - 1, f - 1]
        bars = ax.bar(x, y, color='blue')
        if r == 5:
            if f == 3:
                ax.set_xlabel('{}\nF'.format(f), va='top')
            else:
                ax.set_xlabel('{}\n'.format(f), va='top')
        if f == 1:
            if r == 3:
                ax.set_ylabel('R\n{}'.format(r))
            else:
                ax.set_ylabel(r)
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=8)

        for bar in bars:
            value = bar.get_height()
            if value == y.max():
                bar.set_color('red')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value,
                    int(value),
                    ha='center',
                    va='bottom',
                    color='k')
fig.suptitle('Distribution of M for each F and R',
             fontsize=14)
plt.tight_layout()
plt.show()

# count the number of customers in each segment
segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
               segments_counts,
               color='blue')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
    value = bar.get_width()
    if segments_counts.index[i] in ['champions', 'loyal customers']:
        bar.set_color('red')
    ax.text(value,
            bar.get_y() + bar.get_height() / 2,
            '{:,} ({:}%)'.format(int(value),
                                 int(value * 100 / segments_counts.sum())),
            va='center',
            ha='left'
            )

plt.show()
