# app/app.py
import streamlit as st
import pandas as pd
import os
import joblib

from preprocessing import preprocess_features

# -----------------
# 🔹 Load Artifacts
# -----------------
MODEL_DIR = "saved_models"
TARGET_COLUMNS = ["Depression", "Anxiety", "PersonallyDisorder", "PTSD"]

@st.cache_resource
def load_artifacts():
    models = {}
    for target in TARGET_COLUMNS:
        model_path = os.path.join(MODEL_DIR, f"{target}.joblib")
        if os.path.exists(model_path):
            models[target] = joblib.load(model_path)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    pca_path = os.path.join(MODEL_DIR, "pca.joblib")
    pca = joblib.load(pca_path) if os.path.exists(pca_path) else None

    return models, scaler, pca


# -----------------
# Sidebar Navigation
# -----------------
st.sidebar.title("🧠 Mental Health Prediction")
page = st.sidebar.radio("Navigate", ["Data Upload", "Predict"])

# -----------------
# 1️⃣ Data Upload Page
# -----------------
if page == "Data Upload":
    st.title("📂 Upload Your Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.write("✅ Data Loaded:", df.shape)
        st.dataframe(df.head())

# -----------------
# 2️⃣ Predict Page
# -----------------
elif page == "Predict":
    st.title("🔮 Make Predictions")

    # Load trained models and preprocessing artifacts
    models, scaler, pca = load_artifacts()
    if not models:
        st.error("❌ No trained models found in saved_models/. Please run train.py first.")
        st.stop()

    # Example user inputs
    age = st.number_input("Age", min_value=18, max_value=35, value=25)
    income_before = st.selectbox("Income before war", ["very low", "low", "medium", "high"])
    income_after = st.selectbox("Income after war", ["very low", "low", "medium", "high"])
    fear_life = st.radio("Felt afraid of losing life?", ["yes", "no"])

    # Build single-row dataframe
    input_dict = {
        "Age": [age],
        "income level before war": [income_before],
        "income level after war": [income_after],
        "felt afraid for lossing life": [fear_life],
    }
    df_input = pd.DataFrame(input_dict)

    # Apply same preprocessing
    df_proc = preprocess_features(df_input, target_columns=TARGET_COLUMNS)

    # Drop targets if they exist (user won’t provide them)
    X_input = df_proc.drop(columns=TARGET_COLUMNS, errors="ignore")

    # Apply scaler + PCA (same as training)
    X_input = scaler.transform(X_input)
    if pca is not None:
        X_input = pca.transform(X_input)

    # -----------------
    # 🔮 Predictions
    # -----------------
    st.subheader("Predictions")
    for target, model in models.items():
        try:
            pred = model.predict(X_input)[0]
            st.write(f"**{target}:** {pred}")
        except Exception as e:
            st.warning(f"⚠️ Could not predict {target}: {e}")
