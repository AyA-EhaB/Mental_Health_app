# app/modeling.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =========================================
# Step 1: Define candidate models
# =========================================
def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        ),
        "SVM": SVC(
            random_state=42, probability=True, class_weight="balanced"
        )
    }

# =========================================
# Step 2: Train + evaluate models
# =========================================
def train_models(X, y, target_name):
    """
    Train all models for a single target (Depression, Anxiety, etc.)
    Returns metrics and trained models.
    """
    models = get_models()
    results, trained = {}, {}

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

            # metrics
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )

            # AUC
            auc = np.nan
            if y_proba is not None:
                try:
                    if len(np.unique(y)) == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=np.unique(y))
                        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
                except Exception:
                    pass

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")

            results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "auc": auc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }

            trained[name] = model

        except Exception as e:
            print(f"⚠️ {name} failed for {target_name}: {e}")

    return results, trained

# =========================================
# Step 3: Save best model
# =========================================
def save_best_model(trained_models, results, target, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    # pick best F1-score
    best_model_name, best_metrics = max(
        results.items(), key=lambda kv: kv[1]["f1_score"]
    )
    best_model = trained_models[best_model_name]

    filename = os.path.join(save_dir, f"{target}_{best_model_name.replace(' ', '')}.joblib")
    joblib.dump(best_model, filename)

    return filename, best_model_name, best_metrics

# =========================================
# Step 4: Load model
# =========================================
def load_model(filepath):
    return joblib.load(filepath)
