import io

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class helper():
    @classmethod
    def write_info_in_file(cls, data_frame):
        buffer = io.StringIO()
        data_frame.info(buf=buffer)
        s = buffer.getvalue()
        with open("data/df_info.txt", "w", encoding="utf-8") as f:
            f.write(s)
        return s

    # Function to calculate missing values by column
    @classmethod
    def missing_values_table(cls, df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        # print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        #                                                           "There are " + str(mis_val_table_ren_columns.shape[0]) +
        #       " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    @classmethod
    def drop_columns(cls, data_frame, drop_percent_number=50):
        # Get the columns with > [drop_percent_number]% missing
        missing_df = helper.missing_values_table(data_frame)

        missing_columns = list(missing_df[missing_df['% of Total Values'] > int(drop_percent_number)].index)
        print('We will remove %d columns.' % len(missing_columns))

        # Drop the columns
        data = data_frame.drop(columns=list(missing_columns))
        return data

    @classmethod
    def histogram_of_the_energy_star_score(cls, data_frame):
        # Histogram of the energy_star_score
        plt.style.use('fivethirtyeight')
        plt.hist(data_frame['ENERGY STAR Score'].dropna(), bins=100, edgecolor='k')
        plt.xlabel('Score')
        plt.ylabel('Number of Buildings')
        plt.title('energy_star_score Distribution')

        return plt

    @classmethod
    def create_a_list_of_buildings_with_more_than_100_measurements(cls, data_frame):
        # Create a list of buildings with more than 100 measurements
        types = data_frame.dropna(subset=['ENERGY STAR Score'])
        types = types['Largest Property Use Type'].value_counts()
        types = list(types[types.values > 100].index)

        # Plot of distribution of scores for building categories

        # Plot each building
        for b_type in types:
            # Select the building type
            subset = data_frame[data_frame['Largest Property Use Type'] == b_type]

            # Density plot of Energy Star scores
            sns.kdeplot(subset['ENERGY STAR Score'].dropna(),
                        label=b_type, shade=False, alpha=0.8)

        # label the plot
        plt.xlabel('Energy Star Score', size=20)
        plt.ylabel('Density', size=20)
        plt.title('Density Plot of Energy Star Scores by Building Type', size=28)

        return plt

    @classmethod
    def create_a_list_of_buildings_with_more_than_80_measurements(cls, data_frame):
        # Create a list of buildings with more than 80 measurements
        types = data_frame.dropna(subset=['ENERGY STAR Score'])
        types = types['Primary Property Type - Self Selected'].value_counts()
        types = list(types[types.values > 80].index)

        for b_type in types:
            # Remove outliers before plotting
            subset = data_frame[(data_frame['Weather Normalized Site EUI (kBtu/ft²)'] < 300) &
                                (data_frame['Primary Property Type - Self Selected'] == b_type)]

            # Plot the site EUI on a density plot
            sns.kdeplot(subset['Weather Normalized Site EUI (kBtu/ft²)'].dropna(),
                        label=b_type)

        plt.xlabel('Site EUI (kBtu/ft^2)')
        plt.ylabel('Density')
        plt.title('Density Plot of Site EUI')

        return plt

    @classmethod
    def plot_the_site_EUI_density_plot_for_each_building_type(cls, data_frame):
        # Plot the site EUI density plot for each building type
        types = data_frame.dropna(subset=['ENERGY STAR Score'])
        types = types['Primary Property Type - Self Selected'].value_counts()
        types = list(types[types.values > 80].index)

        for b_type in types:
            # Remove outliers before plotting
            subset = data_frame[(data_frame['Weather Normalized Site EUI (kBtu/ft²)'] < 300) &
                                (data_frame['Primary Property Type - Self Selected'] == b_type)]

            # Plot the site EUI on a density plot
            sns.kdeplot(subset['Weather Normalized Site EUI (kBtu/ft²)'].dropna(),
                        label=b_type);

        plt.xlabel('Site EUI (kBtu/ft^2)');
        plt.ylabel('Density');
        plt.title('Density Plot of Site EUI');

        return plt

    @classmethod
    def energy_star_score_versus_site_EUI(cls, data_frame):
        # Plot the site EUI density plot for each building type
        types = data_frame.dropna(subset=['ENERGY STAR Score'])
        types = types['Primary Property Type - Self Selected'].value_counts()
        types = list(types[types.values > 80].index)

        # Subset to the buildings with most measurements and remove outliers
        subset = data_frame[(data_frame['Weather Normalized Site EUI (kBtu/ft²)'] < 300) &
                            (data_frame['Primary Property Type - Self Selected'].isin(types))]

        # Drop the buildings without a value
        subset = subset.dropna(subset=['ENERGY STAR Score',
                                       'Weather Normalized Site EUI (kBtu/ft²)'])

        subset = subset.rename(columns={'Primary Property Type - Self Selected': 'Property Type'})

        # Linear Plot of Energy Star Score vs EUI
        sns.lmplot('Weather Normalized Site EUI (kBtu/ft²)', 'ENERGY STAR Score',
                   data=subset, hue='Property Type',
                   scatter_kws={'alpha': 0.8, 's': 32}, fit_reg=False,
                   size=12, aspect=1.2);

        plt.title('Energy Star Score vs Site EUI', size=24);

        return plt

    @classmethod
    # Remove correlations from the dataframe that are above corr_val
    def corr_df(cls, x, corr_val):
        # Dont want to remove correlations between Energy Star Score
        y = x['ENERGY STAR Score']
        x = x.drop(columns=['ENERGY STAR Score'])

        # Matrix of all correlations
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through all correlations
        for i in iters:
            for j in range(i):
                item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
                col = item.columns
                row = item.index
                val = abs(item.values)
                # If correlation is above the threshold, add to list to remove
                if val >= corr_val:
                    drop_cols.append(col.values[0])

        # Remove collinear variables
        drops = set(drop_cols)
        x = x.drop(columns=drops)
        x = x.drop(columns=['Site EUI (kBtu/ft²)'])

        # Make sure to add the label back in to the data
        x['ENERGY STAR Score'] = y

        return x
