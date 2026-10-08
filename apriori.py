import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv("mining_data.csv")

# Convert numerical values into categories

data["High_Gas"] = data["Gas_Level"] > 70

data["High_Rainfall"] = data["Rainfall"] > 100

data["High_Vibration"] = (
    data["Ground_Vibration"] > 8
)

data["Weak_Rock"] = (
    data["Rock_Strength"] < 40
)

data["High_Temperature"] = (
    data["Equipment_Temperature"] > 85
)

data["High_Risk"] = (
    data["Risk_Level"] == "High"
)

transactions = data[
    [
        "High_Gas",
        "High_Rainfall",
        "High_Vibration",
        "Weak_Rock",
        "High_Temperature",
        "High_Risk"
    ]
]

frequent_items = apriori(
    transactions,
    min_support=0.05,
    use_colnames=True
)

rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.5
)

rules = rules.sort_values(
    "lift",
    ascending=False
)

print("\nFrequent Itemsets:")
print(frequent_items)

print("\nAssociation Rules:")
print(
    rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].head(20)
)