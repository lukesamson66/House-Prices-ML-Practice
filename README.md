This repo contains ML practice problems to complete in order to improve ML skills.



**Problem 1: Feature Scaling and Linear Regression Evaluation**

Problem Statement

You are given a dataset containing information about house sizes and their prices.
Your task is to train a Linear Regression model, after applying standard scaling, and compute the Mean Squared Error (MSE) on the test set.

Note

Do not hardcode results
Use StandardScaler from sklearn.preprocessing

Input Format

A Pandas DataFrame df with:
size_sqft (float)
num_rooms (int)
price (float)

Task

1. Split the data into 80% train / 20% test
2. Apply Standard Scaling to features
3. Train a LinearRegression model
4. Compute Mean Squared Error



**Problem 2: Binary Classification with Logistic Regression**

Problem Statement

You are given customer data for a marketing campaign.
Train a Logistic Regression model and compute the classification accuracy.

Input Format

DataFrame df
Target column: purchased (0 or 1)

Task

1. Separate features and target
2. Perform train-test split (75%-25%)
3. Apply StandardScaler
4. Train LogisticRegression
5. Compute Accuracy Score

