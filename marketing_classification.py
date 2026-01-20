from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60],
    "salary": [25000, 32000, 52000, 58000, 60000, 65000, 68000, 72000],
    "purchased": [0, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X, y = df[["age", "salary"]], df["purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression().fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)

print(accuracy)


