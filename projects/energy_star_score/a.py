import features as features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from energy_star_score import helper_class



# Drop the columns
# data = data.drop(columns = list(missing_columns))

# For older versions of pandas (https://github.com/pandas-dev/pandas/issues/19078)
# data = data.drop(list(missing_columns), axis = 1)

'''
Our first plot has already revealed some surprising (and suspicious) information! As the energy_star_score is a 
percentile rank, we would expect to see a completely flat distribution with each score making up 1% of the 
distribution (about 90 buildings). However, this is definitely not the case as we can see that the two most 
common scores, 1 and 100, make up a disproporationate number of the overall scores.
'''
# Histogram of the energy_star_score
# plt.style.use('fivethirtyeight')
# plt.hist(data['ENERGY STAR Score'].dropna(), bins = 100, edgecolor = 'k')
# plt.xlabel('Score')
# plt.ylabel('Number of Buildings')
# plt.title('energy_star_score Distribution')
# plt.show()

'''
To contrast the energy_star_score, we can look at the Energy Use Intensity (EUI), 
which is the total energy use divided by the square footage of the building. 
Here the energy usage is not self-reported, so this could be a more objective measure of the 
energy efficiency of a building. 
Moreover, this is not a percentile rank, so the absolute values are important and we would expect them to be 
approximately normally distributed with perhaps a few outliers on the low or high end.
'''
# Histogram Plot of Site EUI
# figsize(8, 8)
# plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins = 20, edgecolor = 'black');
# plt.xlabel('Site EUI');
# plt.ylabel('Count'); plt.title('Site EUI Distribution');
# # plt.show()

# # Create a list of buildings with more than 100 measurements
types = data.dropna(subset=['ENERGY STAR Score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)
#
# # Plot of distribution of scores for building categories

# # Plot each building
# for b_type in types:
#     # Select the building type
#     subset = data[data['Largest Property Use Type'] == b_type]
#
#     # Density plot of Energy Star scores
#     sns.kdeplot(subset['ENERGY STAR Score'].dropna(),
#                 label=b_type, shade=False, alpha=0.8);
#
# # label the plot
# plt.xlabel('energy_star_score', size=20);
# plt.ylabel('Density', size=20);
# plt.title('Density Plot of Energy Star Scores by Building Type', size=28);
# # plt.show()
# # print(data['Site EUI (kBtu/ft²)'].describe())
#
#
# # Create a list of boroughs with more than 100 observations
# boroughs = data.dropna(subset=['ENERGY STAR Score'])
# boroughs = boroughs['Borough'].value_counts()
# boroughs = list(boroughs[boroughs.values > 100].index)
#
# # Plot each borough distribution of scores
# for borough in boroughs:
#     # Select the building type
#     subset = data[data['Borough'] == borough]
#
#     # Density plot of Energy Star scores
#     sns.kdeplot(subset['ENERGY STAR Score'].dropna(),
#                 label=borough);
#
# # label the plot
# plt.xlabel('energy_star_score', size=20);
# plt.ylabel('Density', size=20);
# plt.title('Density Plot of Energy Star Scores by Borough', size=28);
#
# # Find all correlations and sort
# correlations_data = data.corr()['ENERGY STAR Score'].sort_values()
#
# # Select the numeric columns
# numeric_subset = data.select_dtypes('number')








