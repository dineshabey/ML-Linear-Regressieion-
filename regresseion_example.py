# Importing the necessary libraries for data manipulation and visualization
import numpy as np  # NumPy library for numerical computations
import matplotlib.pyplot as plt  # Matplotlib's pyplot module for plotting
import pandas as pd  # Pandas library for data manipulation and analysis

#train_test_split and LinearRegression classes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#import data file
dataset = pd.read_csv('Salary_Data.csv')

# Extracting the independent variable (X) and dependent variable (y) arrays from the dataset
X = dataset.iloc[:,:-1].values  # Independent variable (input features) excluding the last column
y = dataset.iloc[:,1].values    # Dependent variable (target variable) from the second column

# Splitting the dataset into training and testing sets
# X_train: Features for training, X_test: Features for testing
# y_train: Target variable for training, y_test: Target variable for testing
# test_size=1/3: Splitting the data into 1/3 for testing and 2/3 for training
# random_state=0: Setting a seed for reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#fitting data
# Creating a Linear Regression model and fitting it to the training data
# regressor: Linear Regression model
# X_train, y_train: Training features and target variable
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predecting
# Making predictions using the trained model on the test set
y_pred = regressor.predict(X_test)

# Visualizing the training set results
plt.scatter(X_train, y_train, color='red')  # Scatter plot of training data points
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Regression line for training data

# Visualizing the test set results
plt.scatter(X_test, y_test, color='red')  # Scatter plot of test data points
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Regression line for test data (using training data as x-values)

plt.title('Year of experience vs Salary (Test set)')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()