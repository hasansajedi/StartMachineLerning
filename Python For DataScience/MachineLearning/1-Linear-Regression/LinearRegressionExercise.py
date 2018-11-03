import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers')
# customers.head()
# customers.info()
# customers.describe()

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# More time on site, more money spent. Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
# sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

# Do the same but with the Time on App column instead.
# sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
# sns.jointplot(x='Time on App', y='Length of Membership',data=customers, kind='hex',)

# Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.
# sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', lm.coef_)

predictions = lm.predict(X_test)
# sns.scatterplot(y_test, predictions)
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# sns.distplot((y_test - predictions), bins=50)

coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)

plt.show()
