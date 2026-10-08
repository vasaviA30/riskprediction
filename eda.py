import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("mining_data.csv")

# Risk distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=data,
    x="Risk_Level"
)

plt.title("Mining Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Records")
plt.show()


# Risk vs Gas
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Risk_Level",
    y="Gas_Level"
)

plt.title("Gas Level vs Mining Risk")
plt.show()


# Risk vs vibration
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Risk_Level",
    y="Ground_Vibration"
)

plt.title("Ground Vibration vs Mining Risk")
plt.show()


# Correlation
plt.figure(figsize=(12, 8))

numeric_data = data.select_dtypes(include="number")

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Mining Feature Correlation")
plt.show()