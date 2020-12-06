# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynominal Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualization LR result
plt.scatter(X, y, c = 'red')
plt.plot(X, lin_reg.predict(X), c = 'blue')
plt.title('True or Bluff (Linear Regrassion)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualization Polynominal LR result
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), c = 'blue')
plt.title('True or Bluff (Linear Regrassion)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
