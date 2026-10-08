import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("mining_data.csv")

print("Dataset loaded successfully!")
print("Number of records:", len(df))


# ==============================
# 2. Select Features
# ==============================

features = [
    "Gas_Level",
    "Ground_Vibration",
    "Equipment_Vibration",
    "Equipment_Temperature",
    "Rainfall"
]

X = df[features]


# ==============================
# 3. Feature Scaling
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==============================
# 4. Isolation Forest
# ==============================

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = model.fit_predict(X_scaled)


# ==============================
# 5. Convert Result
# ==============================

df["Anomaly_Status"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


# ==============================
# 6. Anomaly Distribution
# ==============================

print("\n==============================")
print("ANOMALY DISTRIBUTION")
print("==============================")

print(df["Anomaly_Status"].value_counts())


# ==============================
# 7. Anomaly Percentage
# ==============================

percentage = (
    df["Anomaly_Status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nAnomaly Percentage:")
print(percentage)


# ==============================
# 8. Show Anomalous Records
# ==============================

print("\n==============================")
print("SAMPLE ANOMALOUS RECORDS")
print("==============================")

anomalies = df[
    df["Anomaly_Status"] == "Anomaly"
]

print(
    anomalies[
        [
            "Mine_ID",
            "Gas_Level",
            "Ground_Vibration",
            "Equipment_Vibration",
            "Equipment_Temperature",
            "Rainfall",
            "Risk_Level"
        ]
    ].head(10)
)


# ==============================
# 9. Save Results
# ==============================

df.to_csv(
    "anomaly_mining_data.csv",
    index=False
)

print("\nAnomaly dataset saved successfully!")


# ==============================
# 10. Visualization
# ==============================

plt.figure(figsize=(10, 6))

normal = df[df["Anomaly_Status"] == "Normal"]
anomaly = df[df["Anomaly_Status"] == "Anomaly"]

plt.scatter(
    normal["Gas_Level"],
    normal["Ground_Vibration"],
    s=20,
    label="Normal"
)

plt.scatter(
    anomaly["Gas_Level"],
    anomaly["Ground_Vibration"],
    s=40,
    label="Anomaly"
)

plt.xlabel("Gas Level")
plt.ylabel("Ground Vibration")

plt.title("Mining Condition Anomaly Detection")

plt.legend()

plt.show()