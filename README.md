Sales Prediction using Linear Regression
1. Introduction
This project aims to predict car purchase amounts based on various customer features. We will use machine learning techniques, specifically linear regression, to build a model that can help businesses optimize marketing strategies based on the predicted sales.
2. Steps Involved
2.1 Data Preprocessing
The first step is to preprocess the data. This includes loading the dataset, handling missing values, removing irrelevant columns, and performing feature scaling using the Min-Max scaler.
2.2 Model Training
We train a Linear Regression model using the preprocessed data. The model is fit on the training data, and predictions are made on the test set.
2.3 Model Evaluation
The model is evaluated using metrics such as R² (coefficient of determination), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to measure its performance and accuracy in predicting sales.
2.4 Visualization
A scatter plot is generated to visually compare the actual and predicted car purchase amounts.
3. Code
Here is the Python code implementing the sales prediction model using Linear Regression.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
df = pd.read_csv('car_purchasing.csv')

# Basic Data Inspection
print(df.head())
print(df.info())
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Missing values: {df.isnull().sum()}")

# Drop irrelevant columns
df.drop(columns=['customer name', 'customer e-mail', 'country', 'gender'], inplace=True)

# Define features (X) and target variable (y)
X = df.drop('car purchase amount', axis=1)
y = df['car purchase amount']

# Feature scaling using Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = np.array(y).reshape(-1, 1)
y = scaler.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model using R^2 and MSE
R2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

print(f"R2 Score: {R2}")
print(f"Mean Squared Error: {MSE}")
print(f"Root Mean Squared Error: {RMSE}")

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linestyle='--')
plt.xlabel('Actual Car Purchase Amount')
plt.ylabel('Predicted Car Purchase Amount')
plt.title('Actual vs Predicted Car Purchase Amount')
plt.show()

4. Conclusion
In this project, we applied linear regression to predict car purchase amounts. The model was trained and evaluated, showing good performance based on the R² score, MSE, and RMSE. The model can help businesses optimize their marketing strategies based on predicted sales, improving decision-making processes.
