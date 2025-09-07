import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load pre-trained models
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("💓 Heart Disease Prediction App")
df = pd.read_csv("heart.csv")


# Load trained models
models = {
    "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "K-Nearest Neighbors": joblib.load("models/knn_model.pkl"),
}

st.sidebar.header("Enter Patient Details")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 94, 200, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dL)", 126, 564, 245)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.slider("Slope (0-2)", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.slider("Thal (0=Normal, 1=Fixed, 2=Reversible)", 0, 2, 1)

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Show dataset overview
st.subheader("📊 Dataset Preview")
st.write(df.head())


# Target distribution
# Shows how many people in the dataset had heart disease (target = 1) vs no disease (target = 0).
st.subheader("❤️ Target Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(data=df, x="target", palette="Set2", ax=ax2)
st.pyplot(fig2)


# Age vs Max Heart Rate
# hue="target" separates healthy vs. diseased points by color.
st.subheader("📈 Age vs Max Heart Rate by Target")
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x="age", y="thalach", hue="target", palette="cool", ax=ax3)
st.pyplot(fig3)

# Model selection
st.subheader("🤖 Predict Heart Disease")
model_choice = st.selectbox("Choose ML Model", list(models.keys()))
model = models[model_choice]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.write("## ✅ Prediction:")
    st.success("💔 Heart Disease Detected" if prediction == 1 else "💚 No Heart Disease")
    st.write("### 🔎 Input Summary:")
    st.write(input_df)