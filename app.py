import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import xgboost as xgb

# ---------------- Neural Network ----------------
class HeartNet(nn.Module):
    def __init__(self, input_dim=13):
        super(HeartNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Page Config ----------------
st.set_page_config(page_title="IBM Certified Heart AI", layout="wide")
st.title("💓 Heart Disease Diagnostic System")

# ---------------- Load Scaler ----------------
scaler = joblib.load("models/scaler.pkl")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Patient Vitals")

def get_input():
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 120, 500, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.sidebar.slider("Resting ECG", 0, 2, 0)
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope", 0, 2, 1)
    ca = st.sidebar.slider("Major Vessels", 0, 3, 0)
    thal = st.sidebar.slider("Thal", 0, 3, 2)

    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

user_data = get_input()

# ---------------- Model Selection ----------------
st.subheader("🤖 Diagnostic Core")

model_map = {
    "random_forest_model.pkl": "Random Forest Model",
    "xgboost_model.json": "XGBoost Model",
    "pytorch_model.pth": "Neural Network (PyTorch)"
}

model_display = list(model_map.values())
selected_display = st.selectbox("Select Model Architecture", model_display)
selected_model = [k for k, v in model_map.items() if v == selected_display][0]

# ---------------- Layout ----------------
feature_names = [
    "Age", "Sex", "ChestPain", "RestBP", "Chol", "FBS",
    "RestECG", "MaxHR", "ExAng", "OldPeak", "Slope", "CA", "Thal"
]
summary_df = pd.DataFrame(user_data, columns=feature_names)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Patient Input Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("📈 Vital Distribution")
    chart_df = summary_df.T
    chart_df.columns = ["Value"]
    st.bar_chart(chart_df)

# ---------------- Single Model Prediction ----------------
with col2:
    st.subheader("🩺 Diagnosis Result")

    if st.button("Run Prediction"):
        scaled_input = scaler.transform(user_data)

        if selected_model.endswith(".pth"):
            net = HeartNet()
            net.load_state_dict(torch.load(f"models/{selected_model}", map_location="cpu"))
            net.eval()
            with torch.no_grad():
                res = 1 if net(torch.FloatTensor(scaled_input)) > 0.5 else 0

        elif selected_model.endswith(".json"):
            bst = xgb.XGBClassifier()
            bst.load_model(f"models/{selected_model}")
            res = bst.predict(scaled_input)[0]

        else:
            clf = joblib.load(f"models/{selected_model}")
            res = clf.predict(scaled_input)[0]

        if res == 1:
            st.error("🚨 HIGH RISK: Cardiac abnormalities detected.")
            risk_text = "High Risk"
        else:
            st.success("✅ LOW RISK: No significant heart disease markers found.")
            risk_text = "Low Risk"

        st.metric("Heart Disease Risk", risk_text)

        # Download report
        report_df = summary_df.copy()
        report_df["Prediction"] = risk_text
        csv = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Patient Report",
            data=csv,
            file_name="heart_disease_report.csv",
            mime="text/csv"
        )

# ---------------- Run All Models ----------------
st.subheader("🤖 AI Model Comparison")

if st.button("Run All AI Models"):
    scaled = scaler.transform(user_data)

    rf = joblib.load("models/random_forest_model.pkl")
    rf_res = rf.predict(scaled)[0]

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/xgboost_model.json")
    xgb_res = xgb_model.predict(scaled)[0]

    net = HeartNet()
    net.load_state_dict(torch.load("models/pytorch_model.pth", map_location="cpu"))
    net.eval()

    with torch.no_grad():
        nn_res = 1 if net(torch.FloatTensor(scaled)) > 0.5 else 0

    results = {
        "Model": ["Random Forest", "XGBoost", "Neural Network"],
        "Prediction": [rf_res, xgb_res, nn_res]
    }

    res_df = pd.DataFrame(results)
    res_df["Prediction"] = res_df["Prediction"].map({1: "High Risk", 0: "Low Risk"})

    st.dataframe(res_df, use_container_width=True)