import os
import glob
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, mean_absolute_error

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

# print(X_df.head())
# print(y_df.head())

freqs = pd.read_csv('data/freq.csv')
freqs = freqs['freqs'].values

# print(len(freqs), freqs)
# print(np.unique(y_df['molecule'].values))

# Target for classification
molecule = y_df['molecule'].values
# Target for regression
concentration = y_df['concentration'].values

# fig, ax = plot_spectra(freqs, spectra, 'All training spectra')
# plt.show()
#
# fig, ax = plot_spectra_by_type(freqs, spectra, molecule)
# ax.set_title('Mean spectra in function of the molecules')
# plt.show()
#
# fig, ax = plot_spectra_by_type(freqs, spectra, concentration, 'Mean spectra in function of the concentrations')
# plt.show()

from sklearn.model_selection import train_test_split

X_test1 = read_spectra('data/X_test.csv')

X = spectra
y = molecule
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
pipeline = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0), rfc)
pred_molecule = pipeline.fit(spectra, molecule).predict(X_test1)
X_test_df['molecule'] = pred_molecule.tolist()

print('Accuracy score: {0:.2f}'.format(pipeline.score(X_test1, pred_molecule)))

pipeline_reg = LinearRegression()
pipeline1 = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0), pipeline_reg)
pred_concentration = pipeline1.fit(spectra, concentration).predict(X_test1)
X_test_df['concentration'] = pred_concentration.tolist()
X_test_df.to_csv('sample.csv', index=False)

accuracy = pipeline1.score(X_test1, pred_molecule)

# fig, ax = plot_cm(
#     confusion_matrix(X_test1, pred_concentration),
#     pipeline.classes_,
#     'Confusion matrix using {}'.format(clf.__class__.__name__))
# print(confusion_matrix(X_test1, pred_concentration))

# print('confusion matrix score: {0:.2f}'.format(confusion_matrix(molecule, y_pred)))

# compute the statistics on the training data
# med, var = fit_params(spectra)
# # transform the training and testing data
# spectra_scaled = transform(spectra, med, var)
# spectra_test_scaled = transform(spectra_test, med, var)
# regression_experiment(spectra_scaled, spectra_test_scaled, concentration, concentration_test)

# score = ((2/3)*accuracy)+((1/3)*1)
# print(score)

