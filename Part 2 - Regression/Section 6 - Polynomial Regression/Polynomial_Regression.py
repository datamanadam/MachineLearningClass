# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#KEEP X AS MATRIX
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

# Fitting linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visulaising the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title("Truth or Bluff(Linear Regressoion")
plt.ylabel("Salary")
plt.xlabel("Position Level")
plt.show()

# Visualising the Polynomial Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression")
plt.ylabel("Salary")
plt.xlabel("Position Level")
plt.show()


