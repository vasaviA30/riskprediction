import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("mining_data.csv")

features = [
    "Depth",
    "Rock_Strength",
    "Rainfall",
    "Temperature",
    "Gas_Level",
    "Ground_Vibration",
    "Equipment_Vibration",
    "Equipment_Temperature",
    "Previous_Accidents",
    "Equipment_Age",
    "Blast_Quantity",
    "Ore_Grade",
    "Operating_Hours"
]

X = data[features]

encoder = LabelEncoder()

y = encoder.fit_transform(
    data["Risk_Level"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# Example new mining condition

new_mining_data = pd.DataFrame({
    "Depth": [450],
    "Rock_Strength": [25],
    "Rainfall": [130],
    "Temperature": [42],
    "Gas_Level": [90],
    "Ground_Vibration": [10],
    "Equipment_Vibration": [11],
    "Equipment_Temperature": [100],
    "Previous_Accidents": [3],
    "Equipment_Age": [8],
    "Blast_Quantity": [700],
    "Ore_Grade": [65],
    "Operating_Hours": [18]
})

prediction = model.predict(
    new_mining_data
)

risk = encoder.inverse_transform(
    prediction
)[0]

probability = model.predict_proba(
    new_mining_data
).max()

print("==============================")
print("     MINING RISK PREDICTION")
print("==============================")

print("Risk Level:", risk)

print(
    "Confidence:",
    round(probability * 100, 2),
    "%"
)

if risk == "High":

    print("\nWARNING: HIGH RISK")

    print("Recommended actions:")
    print("1. Inspect mining zone")
    print("2. Check ground stability")
    print("3. Inspect equipment")
    print("4. Review blasting conditions")
    print("5. Increase monitoring")

elif risk == "Medium":

    print("\nCAUTION: MEDIUM RISK")

    print("Recommended actions:")
    print("1. Monitor mining conditions")
    print("2. Inspect equipment")
    print("3. Check environmental conditions")

else:

    print("\nSTATUS: LOW RISK")

    print("Continue regular monitoring.")