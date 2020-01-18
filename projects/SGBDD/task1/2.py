import numpy as np

np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import seaborn as sns

sns.set()

from analysis.preprocessing import read_spectra
from analysis.plotting import plot_spectra, plot_spectra_by_type, plot_cm, plot_regression
from analysis.regression import regression_experiment, fit_params, transform

'''
molecule: 
    Type of chemotherapeutic agent. Four possible values: A for infliximab, B for bévacizumab, Q for ramucirumab, R for rituximab. Dimension: (n_samples,)
vial: 
    Vial type. Three possible values: 1, 2, 3. Dimension: (1, n_samples)
solute: 
    Solute group. Fourteen possible values: 1, 2, ..., 14. Dimension: (1, n_samples)
concentration: 
    Concentration of the molecule. Dimension: (n_samples, 1)
spectra: 
    Intensity of Raman spectrum. Dimension: (n_samples, 1866)
'''

X_df = pd.read_csv('data/X_train.csv')
y_df = pd.read_csv('data/y_train.csv')
X_test_df = pd.read_csv('data/X_test.csv', sep=';')

spectra = X_df['spectra'].values
spectra_test = X_test_df['spectra'].values

spectra = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
spectra_test = np.array([np.array(dd[1:-1].split(',')).astype(float) for dd in spectra_test])

X_df['spectra'] = spectra.tolist()
X_test_df['spectra'] = spectra_test.tolist()

freqs = pd.read_csv('data/freq.csv')
freqs = freqs['freqs'].values

# print(len(freqs), freqs)
print(np.unique(y_df['molecule'].values))

# Target for classification
molecules_map = {'A': 0, 'B': 1, 'Q': 2, 'R': 3}
y_df['molecule'] = y_df['molecule'].replace(molecules_map)
molecule = y_df['molecule'].values

# Target for regression
concentration = y_df['concentration'].values

# fig, ax = plot_spectra(freqs, spectra, 'All training spectra')
# plt.show()

# fig, ax = plot_spectra_by_type(freqs, spectra, molecule)
# ax.set_title('Mean spectra in function of the molecules')
# plt.show()
#
# fig, ax = plot_spectra_by_type(freqs, spectra, concentration, 'Mean spectra in function of the concentrations')
# plt.show()

from sklearn.model_selection import train_test_split

X_test1 = read_spectra('data/X_test.csv')
# Calculate test size
test_size = ((len(X_test1)) / (len(spectra)))
print('Test size:', test_size)
train_features, test_features, train_labels, test_labels = train_test_split(spectra, molecule, test_size=test_size,
                                                                            random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print('X_test1 Shape:', X_test1.shape)

# Make predictions on test data using the model trained on original data
rf = RandomForestClassifier()
rf.fit(train_features[:len(X_test1) - 1], train_labels[:len(X_test1) - 1])
pred_molecule = rf.predict(X_test1)

molecules_map = {0: 'A', 1: 'B', 2: 'Q', 3: 'R'}
pred_molecule_data = map(lambda x: molecules_map[x], pred_molecule)
X_test_df['molecule'] = pred_molecule_data

print('Predictions Shape:', pred_molecule.shape)
# Performance metrics
errors = abs(np.subtract(pred_molecule, test_labels))
print('Metrics for Random Forest Trained on Original Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')

# Calculate and display accuracy
print('Accuracy:', round(rf.score(test_features, test_labels), 3), '%.')
accuracy_molecule = round(rf.score(test_features, test_labels), 3)

#========================== Start predict concentration ==========================
train_features, test_features, train_labels, test_labels = train_test_split(spectra, concentration, test_size=test_size,
                                                                            random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print('X_test1 Shape:', X_test1.shape)

# Make predictions on test data using the model trained on original data
rfr = RandomForestRegressor()
rfr.fit(train_features[:len(X_test1) - 1], train_labels[:len(X_test1) - 1])
pred_concentration = rfr.predict(X_test1)

X_test_df['concentration'] = pred_concentration.tolist()
X_test_df.to_csv('sample.csv', index=False)

print('Predictions Shape:', pred_concentration.shape)
# Performance metrics
errors = abs(np.subtract(pred_concentration, test_labels))
print('Metrics for Random Forest Trained on Original Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')

# Calculate and display accuracy
print('Accuracy:', round(rfr.score(test_features, test_labels), 3), '%.')
accuracy_concentration = round(rfr.score(test_features, test_labels), 3)

#========================== End predict concentration ==========================

#========================== Calculate score ==========================
score = ((2 / 3) * accuracy_molecule) + ((1 / 3) * accuracy_concentration)
print('Score:', score)
