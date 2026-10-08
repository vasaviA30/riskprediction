import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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
    "Depth",
    "Rock_Strength",
    "Rainfall",
    "Gas_Level",
    "Ground_Vibration",
    "Equipment_Vibration",
    "Equipment_Temperature"
]

X = df[features]


# ==============================
# 3. Feature Scaling
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==============================
# 4. Apply K-Means
# ==============================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ==============================
# 5. Cluster Distribution
# ==============================

print("\n==============================")
print("CLUSTER DISTRIBUTION")
print("==============================")

print(df["Cluster"].value_counts().sort_index())


# ==============================
# 6. Cluster Profiles
# ==============================

print("\n==============================")
print("CLUSTER PROFILES")
print("==============================")

cluster_profile = df.groupby("Cluster")[features].mean()

print(cluster_profile.round(2))


# ==============================
# 7. Risk Distribution by Cluster
# ==============================

print("\n==============================")
print("RISK DISTRIBUTION BY CLUSTER")
print("==============================")

risk_distribution = pd.crosstab(
    df["Cluster"],
    df["Risk_Level"],
    normalize="index"
) * 100

print(risk_distribution.round(2))


# ==============================
# 8. Save Clustered Dataset
# ==============================

df.to_csv(
    "clustered_mining_data.csv",
    index=False
)

print("\nClustered dataset saved successfully!")


# ==============================
# 9. Visualization
# ==============================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Depth"],
    df["Gas_Level"],
    c=df["Cluster"],
    s=30
)

plt.xlabel("Depth")
plt.ylabel("Gas Level")

plt.title("Mining Risk Clusters")

plt.colorbar(label="Cluster")

plt.show()