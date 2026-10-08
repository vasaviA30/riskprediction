import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

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
y = encoder.fit_transform(data["Risk_Level"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Naive Bayes": GaussianNB(),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy
    })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)