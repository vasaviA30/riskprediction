import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("mining_data.csv")

print("Original shape:", data.shape)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Remove duplicate records
data = data.drop_duplicates()

# Encode categorical columns
label_encoder = LabelEncoder()

data["Risk_Level_Encoded"] = label_encoder.fit_transform(
    data["Risk_Level"]
)

data["Equipment_Failure_Encoded"] = label_encoder.fit_transform(
    data["Equipment_Failure"]
)

print("\nRisk mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(i, label)

# Select numerical features
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

scaler = StandardScaler()

data[features] = scaler.fit_transform(data[features])

data.to_csv("processed_mining_data.csv", index=False)

print("\nPreprocessing completed.")
print("Saved as processed_mining_data.csv")