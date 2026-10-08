import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("mining_data.csv")

features = [
    "Depth",
    "Ore_Grade",
    "Equipment_Age",
    "Operating_Hours",
    "Rainfall",
    "Ground_Vibration",
    "Equipment_Vibration"
]

X = data[features]

y = data["Production"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

prediction = model.predict(
    X_test
)

print(
    "Mean Absolute Error:",
    mean_absolute_error(
        y_test,
        prediction
    )
)

print(
    "R2 Score:",
    r2_score(
        y_test,
        prediction
    )
)

# Example prediction

sample = X.iloc[[0]]

result = model.predict(sample)

print(
    "\nPredicted Production:",
    result[0],
    "tons/day"
)