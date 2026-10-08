import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

# ==========================================
# 1. Generate Mining Dataset
# ==========================================

data = pd.DataFrame({

    "Mine_ID": [f"M{i:04d}" for i in range(1, n + 1)],

    "Depth": np.random.randint(50, 600, n),

    "Rock_Strength": np.random.uniform(10, 100, n),

    "Rainfall": np.random.uniform(0, 200, n),

    "Temperature": np.random.uniform(20, 50, n),

    "Gas_Level": np.random.uniform(5, 120, n),

    "Ground_Vibration": np.random.uniform(1, 15, n),

    "Equipment_Vibration": np.random.uniform(1, 15, n),

    "Equipment_Temperature": np.random.uniform(40, 120, n),

    "Previous_Accidents": np.random.randint(0, 6, n),

    "Equipment_Age": np.random.randint(1, 16, n),

    "Blast_Quantity": np.random.uniform(100, 1000, n),

    "Ore_Grade": np.random.uniform(20, 95, n),

    "Operating_Hours": np.random.uniform(1, 24, n)

})


# ==========================================
# 2. Create Risk Score
# ==========================================

risk_score = (

    data["Depth"] * 0.02

    + (100 - data["Rock_Strength"]) * 0.30

    + data["Rainfall"] * 0.08

    + data["Gas_Level"] * 0.30

    + data["Ground_Vibration"] * 2

    + data["Equipment_Vibration"] * 2

    + data["Equipment_Temperature"] * 0.15

    + data["Previous_Accidents"] * 8

    + data["Equipment_Age"] * 2

)


# ==========================================
# 3. Convert Risk Score into Risk Categories
# ==========================================

low_threshold = risk_score.quantile(0.30)
high_threshold = risk_score.quantile(0.70)

data["Risk_Level"] = pd.cut(

    risk_score,

    bins=[
        -np.inf,
        low_threshold,
        high_threshold,
        np.inf
    ],

    labels=[
        "Low",
        "Medium",
        "High"
    ]

)


# ==========================================
# 4. Production
# ==========================================

data["Production"] = (

    1500

    + data["Ore_Grade"] * 8

    - data["Risk_Level"].map({

        "Low": 0,

        "Medium": 150,

        "High": 300

    }).astype(float)

    + np.random.normal(0, 100, n)

)


# ==========================================
# 5. Equipment Failure
# ==========================================

failure_score = (

    data["Equipment_Vibration"] * 2

    + data["Equipment_Temperature"] * 0.3

    + data["Equipment_Age"] * 2

    + data["Operating_Hours"] * 0.5

)

data["Equipment_Failure"] = np.where(

    failure_score > 70,

    "Failure",

    "Normal"

)


# ==========================================
# 6. Save Dataset
# ==========================================

data.to_csv(
    "mining_data.csv",
    index=False
)


# ==========================================
# 7. Display Results
# ==========================================

print("Dataset created successfully!")

print("\nFirst 5 records:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nRisk Level Distribution:")
print(data["Risk_Level"].value_counts())

print("\nRisk Level Percentage:")
print(
    data["Risk_Level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nRisk Score Thresholds:")
print("Low / Medium threshold:", round(low_threshold, 2))
print("Medium / High threshold:", round(high_threshold, 2))