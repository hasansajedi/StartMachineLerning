from utils import unique_counts

import numpy as np
import copy

from datetime import datetime
import locale

locale.setlocale(locale.LC_ALL, "german")

from scipy import io
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

sns.set()

### Import factura data as data frame ###
# file "DataScienceAufgabe2_FacturaData" is located in folder "/DataScienceAufgabe2/Data/".
# In order to use it in jupyter notebook you have to upload it (See "Upload" Button in http://localhost:8889/tree) )

factura_df = pd.read_csv("data/DataScienceAufgabe2_FacturaData.csv", encoding="iso-8859-1", decimal=",", sep=";")
print(factura_df['purchaseDate'].max())
# get column overview
factura_df.dtypes
# returns purchases in total, unique customers, unique articles
# unique_counts(factura_df)

# print(factura_df.head())

# lets chose one special customer
chosenCustomerNr = 557

# search article with highest std of discount deviation

# first: get factura for chosen customer only
factura_of_chosenCustomer = factura_df[factura_df["customerNumber"] == chosenCustomerNr]
# group by articleNumber using standard deviation of given price discount
groupedByArticle = factura_of_chosenCustomer.groupby("articleNumber")["discountOnRecommendedRetailPrice[%]"].std()
groupedByArticle = groupedByArticle.reset_index()
groupedByArticle = groupedByArticle.sort_values(by="discountOnRecommendedRetailPrice[%]", ascending=False)

articleWithHighestDiscountDeviation = groupedByArticle.iloc[0]["articleNumber"]
# print(groupedByArticle.head(7))

facturaOfArticle = factura_of_chosenCustomer[
    factura_of_chosenCustomer["articleNumber"] == articleWithHighestDiscountDeviation]
ax1 = sns.distplot(facturaOfArticle["discountOnRecommendedRetailPrice[%]"],
                   rug=True, rug_kws={"color": "g"},
                   kde_kws={"lw": 1,
                            "label": "Info:\n customer = " + str(chosenCustomerNr) +
                                     "\n artNr = " + str(articleWithHighestDiscountDeviation) +
                                     "\n turnover = " + str(
                                locale.format("%.2f", facturaOfArticle.purchaseAmount.sum(), grouping=True)) + "€"
                            }
                   )
# plt.show()

# produce monthly overview of purchases of chosenCustomerNr

# convert purchaseDate column from string to datetime format
factura_df["purchaseDate_datatime"] = pd.to_datetime(factura_df["purchaseDate"])
# generate new column purchaseMonth in order to produce monthly overview of purchases
factura_df["purchaseMonth"] = factura_df["purchaseDate_datatime"].apply(lambda x: x.strftime('%m.%Y_%B'))
# third dataset in factura_df


monthlyGrouped = factura_df[factura_df["customerNumber"] == chosenCustomerNr].groupby("purchaseMonth")[
    "purchaseAmount"].sum()
monthlyGrouped.reset_index()
fig = monthlyGrouped.plot(kind="bar")
fig.set_ylabel('Turnover(€) of customer Nr. ' + str(chosenCustomerNr))
# plt.show()

factura_overview_grouped = factura_df.groupby(['customerNumber']).agg({
    'articleNumber': lambda x: x.nunique(),
    'purchaseAmount': ['sum'],
    'branchNumber': ['unique', lambda x: x.nunique()],
    'discountOnRecommendedRetailPrice[%]': ['min', 'max', 'mean']
})
# lets chose one special customer
chosenCustomerNr = 577
factura_overview_grouped.head(7)

# Since recency is calculated for a point in time, and the last invoice date is 2011–12–09, we will use 2011–12–10 to calculate recency.
import datetime as dt

NOW = dt.datetime(2019, 1, 1)
factura_df['purchaseDate'] = pd.to_datetime(factura_df['purchaseDate'])

# Create a RFM table
rfmTable = factura_df.groupby('customerNumber').agg(
    {'purchaseDate': lambda x: (NOW - x.max()).days, 'articleNumber': lambda x: len(x),
     'purchaseAmount': lambda x: x.sum()})
rfmTable['purchaseDate'] = rfmTable['purchaseDate'].astype(int)
rfmTable.rename(columns={'purchaseDate': 'recency',
                         'articleNumber': 'frequency',
                         'purchaseAmount': 'monetary_value'}, inplace=True)
quantiles = rfmTable.quantile(q=[0.25, 0.5, 0.75])
quantiles = quantiles.to_dict()
segmented_rfm = rfmTable


def RScore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4


def FMScore(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1


# Add segment numbers to the newly created segmented RFM table

segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency', quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency', quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value', quantiles,))
# print(segmented_rfm.head())

# Add a new column to combine RFM score: 111 is the highest score as we determined earlier.
segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(
    str) + segmented_rfm.m_quartile.map(str)
# print(segmented_rfm.head())

# Who are the top 10 of our best customers!
print(segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False).head(10))
