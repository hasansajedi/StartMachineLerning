import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from energy_star_score.helper import helper

# Read in data into a dataframe
from energy_star_score.models import regression_preprocessing

filename = "data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv"
data = pd.read_csv(filename)

# Display top of dataframe
data.head()

# See the column data types and non-missing values
helper.write_info_in_file(data)

# Replace all occurrences of Not Available with numpy not a number
data = data.replace({'Not Available': np.nan})

# Iterate through the columns
for col in list(data.columns):
    # Select columns that should be numeric
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in
            col or 'therms' in col or 'gal' in col or 'Score' in col):
        # Convert the data type to float
        data[col] = data[col].astype(float)

# Statistics for each column
data.describe()

# Drop the columns
data = helper.drop_columns(data_frame=data, drop_percent_number=50)

'''
Our first plot has already revealed some surprising (and suspicious) information! As the energy_star_score is a 
percentile rank, we would expect to see a completely flat distribution with each score making up 1% of the 
distribution (about 90 buildings). However, this is definitely not the case as we can see that the two most 
common scores, 1 and 100, make up a disproporationate number of the overall scores.
'''
# Histogram of the energy_star_score
# pl = helper.histogram_of_the_energy_star_score(data_frame=data)
# plt.show()

# Create a list of buildings with more than 100 measurements
# pl = helper.create_a_list_of_buildings_with_more_than_100_measurements(data_frame=data)
# plt.show()

# Plot the site EUI density plot for each building type
# Create a list of buildings with more than 80 measurements
# plt = helper.create_a_list_of_buildings_with_more_than_80_measurements(data_frame=data)
# plt.show()

# Site EUI by Building Type
# helper.plot_the_site_EUI_density_plot_for_each_building_type(data_frame=data)

# Energy Star Score versus Site EUI¶
# plt = helper.energy_star_score_versus_site_EUI(data_frame=data)
# plt.show()

# Linear Correlations with Energy Star Score¶
# Plot the site EUI density plot for each building type
types = data.dropna(subset=['ENERGY STAR Score'])
types = types['Primary Property Type - Self Selected'].value_counts()
types = list(types[types.values > 80].index)

# List of Variables to find correlation coefficients
features = ['Primary Property Type - Self Selected',
            'Weather Normalized Site EUI (kBtu/ft²)',
            'Weather Normalized Site Electricity Intensity (kWh/ft²)',
            'Largest Property Use Type - Gross Floor Area (ft²)',
            'Natural Gas Use (kBtu)',
            'ENERGY STAR Score']

subset = data[features].dropna()
subset = subset[subset['Primary Property Type - Self Selected'].isin(types)]

# Rename the columns
subset.columns = ['Property Type', 'Site EUI',
                  'Electricity Intensity', 'Floor Area',
                  'Natural Gas', 'Energy Star Score']

# Remove outliers
subset = subset[subset['Site EUI'] < 300]
# Group by the building type and calculate correlations
corrs = pd.DataFrame(subset.groupby('Property Type').corr())
corrs = pd.DataFrame(corrs['Energy Star Score'])

# Format the dataframe for display
corrs = corrs.reset_index()
corrs.columns = ['Property Type', 'Variable', 'Correlation with Score']
corrs = corrs[corrs['Variable'] != 'Energy Star Score']
# print(corrs)

new_data = helper.corr_df(data, corr_val=0.5)
print('Old Data Shape:', data.shape)
print('New Data Shape with correlated features removed', new_data.shape)

X_train, X_test, y_train, y_test, missing_scores, feature_names = regression_preprocessing.train_test_reg(new_data)
print('Training Data Shape:', X_train.shape)
print('Testing Data Shape:', X_test.shape)

# Baseline is mean of training label
baseline = np.mean(y_train)
base_error = np.mean(abs(baseline - y_test))
print('Baseline Error: {:0.4f}.'.format(base_error))

# Create and train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions and evaluate
lin_pred = lin_reg.predict(X_test)

print('Linear Regression Error: {:0.4f}.'.format(np.mean(abs(lin_pred - y_test))))


# Create and train random forest
rf_reg = RandomForestRegressor(n_estimators=200, n_jobs=-1)
rf_reg.fit(X_train, y_train)
# Make predicitons and evaluate
rf_reg_pred = rf_reg.predict(X_test)
print('Random Forest Error: {:0.4f}.'.format(np.mean(abs(rf_reg_pred - y_test))))



