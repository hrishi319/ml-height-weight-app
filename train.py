import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

os.makedirs("models", exist_ok=True)

data = {
    "height": [150, 160, 170, 180, 190],
    "weight": [50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[['height']]
y = df['weight']

model = LinearRegression()
model.fit(X, y)

with open("models/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
