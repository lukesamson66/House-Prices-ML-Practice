"""
Problem Statement

You are given a dataset containing information about house sizes and their prices.
Your task is to train a Linear Regression model, after applying standard scaling, and compute the Mean Squared Error (MSE) on the test set.

Note

Do not hardcode results
Use StandardScaler from sklearn.preprocessing
"""

import pandas as pd

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
