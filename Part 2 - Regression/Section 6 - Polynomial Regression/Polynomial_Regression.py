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

