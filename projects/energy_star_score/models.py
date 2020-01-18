import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


class regression_preprocessing():
    # Takes in a dataframe with observations for each building
    # Returns all the data needed for training and testing a regression model
    @classmethod
    def train_test_reg(cls, df):
        # Select the numerical columns
        X = df.select_dtypes('number')

        # Add the selected categorical columns
        X.loc[:, 'Largest Property Use Type'] = df['Largest Property Use Type']
        X.loc[:, 'Metered Areas (Energy)'] = df['Metered Areas (Energy)']
        X.loc[:, 'DOF Benchmarking Submission Status'] = df['DOF Benchmarking Submission Status']

        # One-hot encoding of categorical values
        X = pd.get_dummies(X)

        # Extract the buildings with no score
        missing_scores = X[X['ENERGY STAR Score'].isnull()]

        # Drop the missing scores from the data
        X = X.dropna(subset=['ENERGY STAR Score'])

        # Remove the labels from the features
        y = X['ENERGY STAR Score']
        X = X.drop(columns=['Order', 'Property Id', 'ENERGY STAR Score'])

        missing_scores = missing_scores.drop(columns=['Order',
                                                      'Property Id',
                                                      'ENERGY STAR Score'])

        # Feature names will be used for later interpretation
        feature_names = list(X.columns)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.25,
                                                            random_state=42)

        # Impute missing values using a median strategy
        imputer = Imputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        missing_scores = imputer.transform(missing_scores)

        # Return all data
        return X_train, X_test, y_train, y_test, missing_scores, feature_names
