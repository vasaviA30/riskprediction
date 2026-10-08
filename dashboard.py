import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mining Risk & Decision Support System",
    page_icon="⛏️",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

RISK_FEATURES = [
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

CLUSTER_FEATURES = [
    "Depth",
    "Rock_Strength",
    "Rainfall",
    "Gas_Level",
    "Ground_Vibration",
    "Equipment_Vibration",
    "Equipment_Temperature"
]

EQUIPMENT_FEATURES = [
    "Equipment_Vibration",
    "Equipment_Temperature",
    "Equipment_Age",
    "Operating_Hours",
    "Previous_Accidents"
]

PRODUCTION_FEATURES = [
    "Depth",
    "Ore_Grade",
    "Equipment_Age",
    "Operating_Hours",
    "Rainfall",
    "Ground_Vibration",
    "Equipment_Vibration"
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("mining_data.csv")


df = load_data()


# ============================================================
# TRAIN RISK MODEL - SVM
# ============================================================

@st.cache_resource
def train_risk_model(data):
    X = data[RISK_FEATURES]
    y_text = data["Risk_Level"].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, encoder, accuracy


risk_model, risk_scaler, risk_encoder, risk_accuracy = train_risk_model(df)


# ============================================================
# TRAIN EQUIPMENT FAILURE MODEL
# ============================================================

@st.cache_resource
def train_equipment_model(data):
    X = data[EQUIPMENT_FEATURES]
    y = data["Equipment_Failure"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


equipment_model, equipment_accuracy = train_equipment_model(df)


# ============================================================
# TRAIN PRODUCTION MODEL
# ============================================================

@st.cache_resource
def train_production_model(data):
    X = data[PRODUCTION_FEATURES]
    y = data["Production"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2


production_model, production_mae, production_r2 = train_production_model(df)


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

@st.cache_data
def perform_clustering(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[CLUSTER_FEATURES])

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    result = data.copy()
    result["Cluster"] = clusters

    return result


clustered_df = perform_clustering(df)


# ============================================================
# ISOLATION FOREST ANOMALY DETECTION
# ============================================================

@st.cache_data
def detect_anomalies(data):
    features = [
        "Gas_Level",
        "Ground_Vibration",
        "Equipment_Vibration",
        "Equipment_Temperature",
        "Rainfall"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    prediction = model.fit_predict(X_scaled)

    result = data.copy()
    result["Anomaly_Status"] = np.where(
        prediction == -1,
        "Anomaly",
        "Normal"
    )

    return result


anomaly_df = detect_anomalies(df)


# ============================================================
# APRIORI ASSOCIATION RULES
# ============================================================

@st.cache_data
def generate_apriori_rules(data):
    transactions = pd.DataFrame({
        "High_Gas": data["Gas_Level"] > 70,
        "High_Rainfall": data["Rainfall"] > 100,
        "High_Vibration": data["Ground_Vibration"] > 8,
        "Weak_Rock": data["Rock_Strength"] < 40,
        "High_Temperature": data["Equipment_Temperature"] > 85,
        "High_Risk": data["Risk_Level"].astype(str) == "High"
    })

    frequent_itemsets = apriori(
        transactions,
        min_support=0.05,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.50
    )

    if rules.empty:
        return frequent_itemsets, rules

    # Keep rules whose consequent contains High_Risk
    high_risk_rules = rules[
        rules["consequents"].apply(
            lambda x: "High_Risk" in x
        )
    ].copy()

    high_risk_rules = high_risk_rules.sort_values(
        by="lift",
        ascending=False
    )

    return frequent_itemsets, high_risk_rules


frequent_itemsets, risk_rules = generate_apriori_rules(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_recommendations(values, predicted_risk):
    recommendations = []

    if values["Gas_Level"] > 70:
        recommendations.append("Monitor and inspect ventilation because gas level is high.")

    if values["Rock_Strength"] < 40:
        recommendations.append("Inspect ground stability because rock strength is low.")

    if values["Ground_Vibration"] > 8:
        recommendations.append("Investigate high ground vibration and review blasting conditions.")

    if values["Equipment_Vibration"] > 8:
        recommendations.append("Inspect equipment because equipment vibration is high.")

    if values["Equipment_Temperature"] > 85:
        recommendations.append("Inspect cooling and equipment condition because temperature is high.")

    if values["Rainfall"] > 100:
        recommendations.append("Increase monitoring because rainfall conditions are high.")

    if values["Previous_Accidents"] >= 3:
        recommendations.append("Increase safety monitoring because previous accident count is high.")

    if values["Equipment_Age"] >= 10:
        recommendations.append("Consider detailed maintenance inspection for older equipment.")

    if not recommendations:
        recommendations.append("No major threshold-based warning was detected.")

    if predicted_risk == "High":
        recommendations.insert(
            0,
            "HIGH RISK: Increase monitoring and perform a detailed safety inspection."
        )

    elif predicted_risk == "Medium":
        recommendations.insert(
            0,
            "MEDIUM RISK: Continue monitoring and address identified risk factors."
        )

    else:
        recommendations.insert(
            0,
            "LOW RISK: Continue normal safety monitoring."
        )

    return recommendations


def format_itemset(itemset):
    return " + ".join(sorted(itemset))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⛏️ Mining DWDM System")

page = st.sidebar.selectbox(
    "Select Module",
    [
        "🏠 Dashboard",
        "⚠️ Risk Prediction",
        "📊 Algorithm Comparison",
        "🔍 Risk Analysis",
        "⛏️ Mining Clusters",
        "🔗 Association Rules",
        "🚨 Anomaly Detection",
        "🔧 Equipment Failure",
        "📈 Production Prediction",
        "💡 Recommendations"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("⛏️ Mining Risk & Decision Support System")

    st.markdown(
        """
        **AI/Data Mining based prototype for mining risk analysis,
        equipment failure prediction, anomaly detection and production prediction.**
        """
    )

    st.info(
        "Note: This project uses a synthetically generated dataset. "
        "The displayed model results are experimental and should not be treated "
        "as validated real-world mine-safety predictions."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mining Records", f"{len(df):,}")
    col2.metric("SVM Risk Accuracy", f"{risk_accuracy * 100:.1f}%")
    col3.metric("Equipment Accuracy", f"{equipment_accuracy * 100:.1f}%")
    col4.metric("Anomalies", int((anomaly_df["Anomaly_Status"] == "Anomaly").sum()))

    st.subheader("Project Modules")

    module_data = pd.DataFrame({
        "Module": [
            "Decision Tree",
            "Naive Bayes",
            "KNN",
            "SVM",
            "Random Forest",
            "K-Means",
            "Apriori",
            "Isolation Forest",
            "Equipment Failure",
            "Production Prediction"
        ],
        "Status": ["Completed"] * 10
    })

    st.dataframe(module_data, use_container_width=True, hide_index=True)

    st.subheader("Risk Distribution")

    risk_counts = df["Risk_Level"].value_counts().reindex(
        ["Low", "Medium", "High"]
    ).fillna(0)

    st.bar_chart(risk_counts)


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "⚠️ Risk Prediction":

    st.title("⚠️ Mining Risk Prediction")

    st.write(
        "Enter current mining conditions. The SVM model will classify "
        "the condition as Low, Medium, or High Risk."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        depth = st.number_input("Depth", 50.0, 600.0, 450.0)
        rock_strength = st.number_input("Rock Strength", 10.0, 100.0, 25.0)
        rainfall = st.number_input("Rainfall", 0.0, 200.0, 130.0)
        temperature = st.number_input("Temperature", 20.0, 50.0, 42.0)
        gas_level = st.number_input("Gas Level", 5.0, 120.0, 90.0)

    with col2:
        ground_vibration = st.number_input("Ground Vibration", 1.0, 15.0, 10.0)
        equipment_vibration = st.number_input("Equipment Vibration", 1.0, 15.0, 11.0)
        equipment_temperature = st.number_input(
            "Equipment Temperature", 40.0, 120.0, 100.0
        )
        previous_accidents = st.number_input(
            "Previous Accidents", 0, 5, 3
        )

    with col3:
        equipment_age = st.number_input("Equipment Age", 1, 15, 8)
        blast_quantity = st.number_input("Blast Quantity", 100.0, 1000.0, 700.0)
        ore_grade = st.number_input("Ore Grade", 20.0, 95.0, 65.0)
        operating_hours = st.number_input("Operating Hours", 1.0, 24.0, 18.0)

    if st.button("🔮 Predict Mining Risk", type="primary"):

        input_data = pd.DataFrame([{
            "Depth": depth,
            "Rock_Strength": rock_strength,
            "Rainfall": rainfall,
            "Temperature": temperature,
            "Gas_Level": gas_level,
            "Ground_Vibration": ground_vibration,
            "Equipment_Vibration": equipment_vibration,
            "Equipment_Temperature": equipment_temperature,
            "Previous_Accidents": previous_accidents,
            "Equipment_Age": equipment_age,
            "Blast_Quantity": blast_quantity,
            "Ore_Grade": ore_grade,
            "Operating_Hours": operating_hours
        }])

        scaled_input = risk_scaler.transform(input_data[RISK_FEATURES])

        prediction_encoded = risk_model.predict(scaled_input)[0]
        prediction = risk_encoder.inverse_transform([prediction_encoded])[0]

        probabilities = risk_model.predict_proba(scaled_input)[0]
        confidence = probabilities.max() * 100

        st.divider()

        if prediction == "High":
            st.error(f"🚨 HIGH RISK — Confidence: {confidence:.2f}%")
        elif prediction == "Medium":
            st.warning(f"⚠️ MEDIUM RISK — Confidence: {confidence:.2f}%")
        else:
            st.success(f"✅ LOW RISK — Confidence: {confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Risk Level": risk_encoder.classes_,
            "Probability": probabilities * 100
        }).set_index("Risk Level")

        st.bar_chart(probability_df)

        st.subheader("Recommended Actions")

        values = input_data.iloc[0].to_dict()

        for recommendation in risk_recommendations(values, prediction):
            st.write("•", recommendation)


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

elif page == "📊 Algorithm Comparison":

    st.title("📊 Algorithm Comparison")

    try:
        comparison = pd.read_csv("model_comparison.csv")

        comparison["Accuracy_Percentage"] = comparison["Accuracy"] * 100

        st.dataframe(
            comparison[
                ["Algorithm", "Accuracy_Percentage"]
            ].rename(
                columns={"Accuracy_Percentage": "Accuracy (%)"}
            ),
            use_container_width=True,
            hide_index=True
        )

        chart_data = comparison.set_index("Algorithm")[
            "Accuracy_Percentage"
        ]

        st.bar_chart(chart_data)

        best_row = comparison.loc[
            comparison["Accuracy"].idxmax()
        ]

        st.success(
            f"🏆 Best algorithm by accuracy: "
            f"{best_row['Algorithm']} "
            f"({best_row['Accuracy'] * 100:.1f}%)"
        )

    except FileNotFoundError:
        st.error(
            "model_comparison.csv was not found. "
            "Run model_comparison.py first."
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "🔍 Risk Analysis":

    st.title("🔍 Mining Risk Analysis")

    st.subheader("Risk Distribution")

    risk_counts = df["Risk_Level"].value_counts().reindex(
        ["Low", "Medium", "High"]
    ).fillna(0)

    st.bar_chart(risk_counts)

    st.subheader("Risk vs Gas Level")

    gas_risk = df.groupby("Risk_Level")["Gas_Level"].mean().reindex(
        ["Low", "Medium", "High"]
    )

    st.bar_chart(gas_risk)

    st.subheader("Risk vs Ground Vibration")

    vibration_risk = df.groupby("Risk_Level")[
        "Ground_Vibration"
    ].mean().reindex(["Low", "Medium", "High"])

    st.bar_chart(vibration_risk)

    st.subheader("Risk vs Equipment Temperature")

    temperature_risk = df.groupby("Risk_Level")[
        "Equipment_Temperature"
    ].mean().reindex(["Low", "Medium", "High"])

    st.bar_chart(temperature_risk)


# ============================================================
# K-MEANS CLUSTERS
# ============================================================

elif page == "⛏️ Mining Clusters":

    st.title("⛏️ Mining Zone Clustering")

    cluster_counts = clustered_df["Cluster"].value_counts().sort_index()

    st.subheader("Cluster Distribution")

    st.bar_chart(cluster_counts)

    st.subheader("Risk Distribution by Cluster")

    cluster_risk = pd.crosstab(
        clustered_df["Cluster"],
        clustered_df["Risk_Level"],
        normalize="index"
    ) * 100

    cluster_risk = cluster_risk.reindex(
        columns=["Low", "Medium", "High"],
        fill_value=0
    )

    st.dataframe(
        cluster_risk.round(2),
        use_container_width=True
    )

    st.bar_chart(cluster_risk)

    st.subheader("Cluster Interpretation")

    for cluster_id in sorted(clustered_df["Cluster"].unique()):

        row = cluster_risk.loc[cluster_id]

        high = row.get("High", 0)
        low = row.get("Low", 0)

        if high >= 40:
            interpretation = "Higher-risk mining conditions"
        elif low >= 35:
            interpretation = "Relatively lower-risk mining conditions"
        else:
            interpretation = "Moderate-risk mining conditions"

        st.write(
            f"**Cluster {cluster_id}:** {interpretation} "
            f"(High Risk: {high:.2f}%, Low Risk: {low:.2f}%)"
        )

    st.subheader("Cluster Profile")

    profile = clustered_df.groupby("Cluster")[CLUSTER_FEATURES].mean()

    st.dataframe(
        profile.round(2),
        use_container_width=True
    )


# ============================================================
# APRIORI
# ============================================================

elif page == "🔗 Association Rules":

    st.title("🔗 Apriori Association Rules")

    st.write(
        "These rules identify combinations of mining conditions "
        "associated with High Risk."
    )

    if risk_rules.empty:
        st.warning("No High-Risk association rules were found.")
    else:

        display_rules = risk_rules.copy()

        display_rules["Antecedent"] = display_rules[
            "antecedents"
        ].apply(format_itemset)

        display_rules["Consequent"] = display_rules[
            "consequents"
        ].apply(format_itemset)

        display_rules = display_rules[
            [
                "Antecedent",
                "Consequent",
                "support",
                "confidence",
                "lift"
            ]
        ].head(15)

        display_rules.columns = [
            "Condition",
            "Result",
            "Support",
            "Confidence",
            "Lift"
        ]

        display_rules["Support"] = (
            display_rules["Support"] * 100
        ).round(2)

        display_rules["Confidence"] = (
            display_rules["Confidence"] * 100
        ).round(2)

        display_rules["Lift"] = display_rules["Lift"].round(3)

        st.dataframe(
            display_rules,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Strongest Association")

        strongest = risk_rules.iloc[0]

        antecedent = format_itemset(strongest["antecedents"])
        consequent = format_itemset(strongest["consequents"])

        st.success(
            f"{antecedent} → {consequent} | "
            f"Lift = {strongest['lift']:.3f} | "
            f"Confidence = {strongest['confidence'] * 100:.2f}%"
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.title("🚨 Mining Condition Anomaly Detection")

    normal_count = (
        anomaly_df["Anomaly_Status"] == "Normal"
    ).sum()

    anomaly_count = (
        anomaly_df["Anomaly_Status"] == "Anomaly"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Normal", f"{normal_count:,}")
    col2.metric("Anomalies", f"{anomaly_count:,}")
    col3.metric(
        "Anomaly %",
        f"{anomaly_count / len(anomaly_df) * 100:.1f}%"
    )

    st.subheader("Anomaly Distribution")

    distribution = anomaly_df[
        "Anomaly_Status"
    ].value_counts()

    st.bar_chart(distribution)

    st.subheader("Sample Anomalous Records")

    anomalies = anomaly_df[
        anomaly_df["Anomaly_Status"] == "Anomaly"
    ]

    columns = [
        "Mine_ID",
        "Gas_Level",
        "Ground_Vibration",
        "Equipment_Vibration",
        "Equipment_Temperature",
        "Rainfall",
        "Risk_Level"
    ]

    st.dataframe(
        anomalies[columns].head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EQUIPMENT FAILURE
# ============================================================

elif page == "🔧 Equipment Failure":

    st.title("🔧 Equipment Failure Prediction")

    col1, col2 = st.columns(2)

    with col1:
        equipment_vibration = st.number_input(
            "Equipment Vibration",
            1.0,
            15.0,
            11.0
        )

        equipment_temperature = st.number_input(
            "Equipment Temperature",
            40.0,
            120.0,
            100.0
        )

        equipment_age = st.number_input(
            "Equipment Age",
            1,
            15,
            8
        )

    with col2:
        operating_hours = st.number_input(
            "Operating Hours",
            1.0,
            24.0,
            18.0
        )

        previous_accidents = st.number_input(
            "Previous Accidents",
            0,
            5,
            3
        )

    if st.button("🔧 Predict Equipment Status", type="primary"):

        input_data = pd.DataFrame([{
            "Equipment_Vibration": equipment_vibration,
            "Equipment_Temperature": equipment_temperature,
            "Equipment_Age": equipment_age,
            "Operating_Hours": operating_hours,
            "Previous_Accidents": previous_accidents
        }])

        prediction = equipment_model.predict(
            input_data[EQUIPMENT_FEATURES]
        )[0]

        probabilities = equipment_model.predict_proba(
            input_data[EQUIPMENT_FEATURES]
        )[0]

        confidence = probabilities.max() * 100

        if prediction == "Failure":
            st.error(
                f"🚨 EQUIPMENT FAILURE RISK — "
                f"Confidence: {confidence:.2f}%"
            )
        else:
            st.success(
                f"✅ EQUIPMENT NORMAL — "
                f"Confidence: {confidence:.2f}%"
            )

        st.subheader("Recommended Action")

        if prediction == "Failure":
            st.warning(
                "Perform equipment inspection and preventive maintenance."
            )
        else:
            st.info(
                "Continue normal monitoring and scheduled maintenance."
            )


# ============================================================
# PRODUCTION PREDICTION
# ============================================================

elif page == "📈 Production Prediction":

    st.title("📈 Mining Production Prediction")

    col1, col2 = st.columns(2)

    with col1:
        depth = st.number_input("Depth", 50.0, 600.0, 450.0)
        ore_grade = st.number_input("Ore Grade", 20.0, 95.0, 65.0)
        equipment_age = st.number_input("Equipment Age", 1, 15, 8)
        operating_hours = st.number_input(
            "Operating Hours",
            1.0,
            24.0,
            18.0
        )

    with col2:
        rainfall = st.number_input("Rainfall", 0.0, 200.0, 130.0)
        ground_vibration = st.number_input(
            "Ground Vibration",
            1.0,
            15.0,
            10.0
        )
        equipment_vibration = st.number_input(
            "Equipment Vibration",
            1.0,
            15.0,
            11.0
        )

    if st.button("📈 Predict Production", type="primary"):

        input_data = pd.DataFrame([{
            "Depth": depth,
            "Ore_Grade": ore_grade,
            "Equipment_Age": equipment_age,
            "Operating_Hours": operating_hours,
            "Rainfall": rainfall,
            "Ground_Vibration": ground_vibration,
            "Equipment_Vibration": equipment_vibration
        }])

        prediction = production_model.predict(
            input_data[PRODUCTION_FEATURES]
        )[0]

        st.success(
            f"📈 Predicted Production: "
            f"**{prediction:,.2f} tons/day**"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Mean Absolute Error",
            f"{production_mae:.2f} tons/day"
        )

        col2.metric(
            "R² Score",
            f"{production_r2:.3f}"
        )

        st.info(
            "The R² score indicates how much variation in production "
            "is explained by the current model and selected features."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.title("💡 Mining Safety Recommendations")

    st.write(
        "This module provides rule-based recommendations based on "
        "the mining conditions entered by the user."
    )

    st.subheader("Enter Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        gas_level = st.number_input("Gas Level", 5.0, 120.0, 90.0)
        rock_strength = st.number_input("Rock Strength", 10.0, 100.0, 25.0)
        ground_vibration = st.number_input(
            "Ground Vibration",
            1.0,
            15.0,
            10.0
        )

    with col2:
        equipment_vibration = st.number_input(
            "Equipment Vibration",
            1.0,
            15.0,
            11.0
        )
        equipment_temperature = st.number_input(
            "Equipment Temperature",
            40.0,
            120.0,
            100.0
        )
        rainfall = st.number_input("Rainfall", 0.0, 200.0, 130.0)

    with col3:
        previous_accidents = st.number_input(
            "Previous Accidents",
            0,
            5,
            3
        )
        equipment_age = st.number_input(
            "Equipment Age",
            1,
            15,
            8
        )

    if st.button("💡 Generate Recommendations", type="primary"):

        values = {
            "Gas_Level": gas_level,
            "Rock_Strength": rock_strength,
            "Ground_Vibration": ground_vibration,
            "Equipment_Vibration": equipment_vibration,
            "Equipment_Temperature": equipment_temperature,
            "Rainfall": rainfall,
            "Previous_Accidents": previous_accidents,
            "Equipment_Age": equipment_age
        }

        input_data = pd.DataFrame([{
            **values,
            "Depth": 450,
            "Temperature": 42,
            "Blast_Quantity": 700,
            "Ore_Grade": 65,
            "Operating_Hours": 18
        }])

        scaled_input = risk_scaler.transform(input_data[RISK_FEATURES])

        prediction_encoded = risk_model.predict(scaled_input)[0]
        prediction = risk_encoder.inverse_transform([prediction_encoded])[0]

        if prediction == "High":
            st.error(f"Predicted Risk: HIGH")
        elif prediction == "Medium":
            st.warning(f"Predicted Risk: MEDIUM")
        else:
            st.success(f"Predicted Risk: LOW")

        st.subheader("Recommended Actions")

        recommendations = risk_recommendations(
            values,
            prediction
        )

        for item in recommendations:
            st.write("•", item)
