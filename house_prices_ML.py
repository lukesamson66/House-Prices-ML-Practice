"""
Problem Statement

You are given a dataset containing information about house sizes and their prices.
Your task is to train a Linear Regression model, after applying standard scaling, and compute the Mean Squared Error (MSE) on the test set.

Note

Do not hardcode results
Use StandardScaler from sklearn.preprocessing
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

"""
Input Format

A Pandas DataFrame df with:
size_sqft (float)
num_rooms (int)
price (float)
"""

data = {
    "size_sqft": [500, 750, 1000, 1250, 1500, 1750, 2000],
    "num_rooms": [1, 2, 2, 3, 3, 4, 4],
    "price": [150000, 200000, 250000, 300000, 340000, 380000, 420000]
}

df = pd.DataFrame(data)

X, y = df[["size_sqft", "num_rooms"]], df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = np.mean((y_test - y_pred) ** 2)
mse = mse.round(2)

print("Mean Squared Error:", mse)
