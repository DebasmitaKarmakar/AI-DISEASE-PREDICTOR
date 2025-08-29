import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Load model, scaler, and template
# -----------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("heart_rf_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    template = pd.read_csv("Heart_user_template.csv")  # 30 columns
    template = template.drop(columns=["target"], errors="ignore")
    return model, scaler, template

model, scaler, template = load_resources()

# -----------------------------
# Helper: preprocess input
# -----------------------------
def preprocess_input(data_dict):
    # Convert dict → DataFrame
    df = pd.DataFrame([data_dict])

    # One-hot encode (must match training)
    df_encoded = pd.get_dummies(df)

    # Align with training columns (30 features)
    df_encoded = df_encoded.reindex(columns=template.columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Predictor ❤️", page_icon="🫀", layout="centered")
st.title("🫀 Heart Disease Predictor")

st.markdown(
    """
    Welcome! Enter your health details below and let's check  
    the possibility of **Heart Disease** with AI assistance.  
    """
)

# Collect inputs (raw 13 features)
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    # Build dictionary for input
    input_dict = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    processed = preprocess_input(input_dict)

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    # Show results
    if prediction == 1:
        st.error(
            f"⚠️ Heart Disease Detected with probability {probability:.2f}\n\n"
            "💡 Please consult a cardiologist immediately."
        )
    else:
        st.success(
            f"✅ No Heart Disease Detected (probability {1 - probability:.2f})\n\n"
            "✨ *'Take care of your heart, it's the rhythm of life.'* 💖"
        )

    st.markdown(
        """
        ---
        🌱 *"Health is the real wealth. Care for your heart today,  
        and it will care for your tomorrow."* 💙
        """,
        unsafe_allow_html=True
    )
