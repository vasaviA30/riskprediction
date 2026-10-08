import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("mining_data.csv")

features = [
    "Equipment_Vibration",
    "Equipment_Temperature",
    "Equipment_Age",
    "Operating_Hours",
    "Previous_Accidents"
]

X = data[features]

encoder = LabelEncoder()

y = encoder.fit_transform(
    data["Equipment_Failure"]
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

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(
    "Equipment Failure Accuracy:",
    accuracy_score(y_test, prediction)
)

print(
    classification_report(
        y_test,
        prediction,
        target_names=encoder.classes_
    )
)