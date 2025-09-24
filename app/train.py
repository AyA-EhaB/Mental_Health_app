# app/train.py
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from preprocessing import load_data, preprocess_features, prepare_ml_data

# -------------------
# ⚙️ Load and preprocess
# -------------------
DATA_PATH = "data/your_dataset.csv"  # 👈 update with your CSV path
target_columns = ['Depression', 'Anxiety', 'Personally Disorder', 'PTSD']

print("📥 Loading dataset...")
df = load_data(DATA_PATH)
df_processed = preprocess_features(df, target_columns)

print("⚙️ Preparing ML data...")
X, y, scaler, pca = prepare_ml_data(df_processed, target_columns, use_pca=True)

# -------------------
# 🤖 Define models
# -------------------
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    ),
    'SVM': SVC(
        random_state=42, probability=True, class_weight='balanced'
    )
}

trained_models = {}

# -------------------
# 🎯 Train per target
# -------------------
for target in target_columns:
    if target not in y.columns:
        continue

    print(f"\n{'='*70}")
    print(f"🎯 Training models for: {target}")
    print(f"{'='*70}")

    y_target = y[target]

    if len(y_target.unique()) < 2:
        print(f"⚠️ Skipping {target} - insufficient class variety")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_target, test_size=0.2, random_state=42, stratify=y_target
    )

    target_models = {}

    for model_name, model in models.items():
        print(f"🔄 Training {model_name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ {model_name} Accuracy: {acc:.3f}")
            target_models[model_name] = model
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")

    trained_models[target] = target_models

# -------------------
# 💾 Save artifacts
# -------------------
os.makedirs("saved_models", exist_ok=True)

best_models = {
    "Depression": trained_models["Depression"]["Logistic Regression"],
    "Anxiety": trained_models["Anxiety"]["Random Forest"],
    "PersonallyDisorder": trained_models["Personally Disorder"]["Random Forest"],
    "PTSD": trained_models["PTSD"]["Random Forest"]
}

for target, model in best_models.items():
    path = f"saved_models/{target}.joblib"
    joblib.dump(model, path)
    print(f"💾 Saved best model for {target} at {path}")

# Save scaler and PCA
joblib.dump(scaler, "saved_models/scaler.joblib")
if pca is not None:
    joblib.dump(pca, "saved_models/pca.joblib")

print("✅ All models, scaler, and PCA saved!")
