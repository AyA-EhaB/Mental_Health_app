# app/app.py
import streamlit as st
import pandas as pd
import os

from preprocessing import load_data, preprocess_features, prepare_ml_data
from modeling import train_models, save_best_model, load_model

# -----------------
# Sidebar Navigation
# -----------------
st.sidebar.title("🧠 Mental Health Prediction")
page = st.sidebar.radio("Navigate", ["Data Upload", "Train Models", "Predict"])

# -----------------
# 1️⃣ Data Upload Page
# -----------------
if page == "Data Upload":
    st.title("📂 Upload Your Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state["df"] = df
        st.write("✅ Data Loaded:", df.shape)
        st.dataframe(df.head())

# -----------------
# 2️⃣ Train Models Page
# -----------------
elif page == "Train Models":
    st.title("⚙️ Train & Evaluate Models")

    if "df" not in st.session_state:
        st.warning("Please upload data first!")
    else:
        df = st.session_state["df"]
        target_columns = ['Depression', 'Anxiety', 'Personally Disorder', 'PTSD']

        # preprocess
        df_processed = preprocess_features(df, target_columns)
        X, y, scaler, pca = prepare_ml_data(df_processed, target_columns)

        results_summary = {}

        for target in target_columns:
            if target in y.columns:
                st.subheader(f"📊 Training for {target}")
                results, trained = train_models(X, y[target], target)

                # save best model
                filepath, best_name, best_metrics = save_best_model(trained, results, target)
                st.success(f"Best {target} model: {best_name} (F1={best_metrics['f1_score']:.3f})")
                results_summary[target] = results

        st.session_state["scaler"] = scaler
        st.session_state["pca"] = pca

# -----------------
# 3️⃣ Predict Page
# -----------------
elif page == "Predict":
    st.title("🔮 Make Predictions")

    if "scaler" not in st.session_state:
        st.warning("Please train models first!")
    else:
        # Example: Ask user for inputs
        age = st.number_input("Age", min_value=18, max_value=35, value=25)
        income_before = st.selectbox("Income before war", ["very low", "low", "medium", "high"])
        income_after = st.selectbox("Income after war", ["very low", "low", "medium", "high"])
        fear_life = st.radio("Felt afraid of losing life?", ["yes", "no"])

        # Build a single-row dataframe
        input_dict = {
            "Age": [age],
            "income level before war": [income_before],
            "income level after war": [income_after],
            "felt afraid for lossing life": [fear_life],
            # add more fields as needed
        }
        df_input = pd.DataFrame(input_dict)

        # preprocess same way as training
        df_proc = preprocess_features(df_input, target_columns=['Depression','Anxiety','Personally Disorder','PTSD'])
        X_input, _, _, _ = prepare_ml_data(df_proc, target_columns=[], use_pca=("pca" in st.session_state and st.session_state["pca"]))

        # Apply scaler + pca
        if st.session_state["scaler"]:
            X_input = st.session_state["scaler"].transform(df_proc.drop(columns=['Depression','Anxiety','Personally Disorder','PTSD'], errors="ignore"))
        if st.session_state["pca"]:
            X_input = st.session_state["pca"].transform(X_input)

        st.subheader("Predictions")
        for target in ['Depression', 'Anxiety', 'Personally Disorder', 'PTSD']:
            model_files = [f for f in os.listdir("saved_models") if target in f]
            if model_files:
                model = load_model(os.path.join("saved_models", model_files[0]))
                pred = model.predict(X_input)[0]
                st.write(f"**{target}:** {pred}")
