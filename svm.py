import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

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

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(
    "SVM Accuracy:",
    accuracy_score(y_test, prediction)
)

print(
    classification_report(
        y_test,
        prediction,
        target_names=encoder.classes_
    )
)