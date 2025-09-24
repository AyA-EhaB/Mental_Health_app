# app/preprocessing.py

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# -----------------
# 🔹 Load & Clean
# -----------------
def load_data(path: str) -> pd.DataFrame:
    """Load dataset and clean column names"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[(df['Age'] >= 18) & (df['Age'] <= 35)]  # Keep target age group
    return df

# -----------------
# 🔹 Feature Engineering
# -----------------
def preprocess_features(df: pd.DataFrame, target_columns=None):
    """Apply feature engineering, encoding, TF-IDF, and derived features"""
    
    if target_columns is None:
        target_columns = ['Depression', 'Anxiety', 'Personally Disorder', 'PTSD']

    df_processed = df.copy()

    # --- Mapping dictionaries ---
    severity_mapping = {
        'never': 0, 'few times': 1, 'sometimes': 2, 'many times': 3, 'very much': 4,
        'Always': 5, 'Very much': 5,
        'Not difficult': 1, 'Difficult': 3, 'Very Difficult': 4, 'Extremely difficult': 5,
        'None-Minimal': 0, 'None-Menimal': 0,
        'Mild Depression': 1, 'Moderate Depression': 2, 'Moderately severe Depression': 3, 'Severe Depression': 4,
        'Mild Anxiety': 1, 'Moderate Anxiety': 2, 'Severe Anxiety': 3,
        'Not affected': 0, 'Affected': 1, 'PTSD': 1, 'Subthreshold PTSD': 0
    }
    income_mapping = {'very low': 1, 'low': 2, 'medium': 3, 'high': 4}
    yes_no_mapping = {'yes': 1, 'no': 0}

    # --- Binary yes/no columns ---
    yes_no_columns = ['leave home for war', 'witnessed direct violence',
                      'hurt or relatives hard', 'lost persons', 'you arrested',
                      'Lack of food or medicine', 'Your property destroyed',
                      'felt afraid for lossing life']
    for col in yes_no_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(yes_no_mapping)

    # --- Income columns ---
    for col in ['income level before war', 'income level after war']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(income_mapping)

    # --- Severity / frequency mappings ---
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            for key, value in severity_mapping.items():
                df_processed[col] = df_processed[col].replace(key, value)

    # --- Target columns mapping ---
    for target in target_columns:
        if target in df_processed.columns:
            if target == 'Depression':
                df_processed[target] = df_processed[target].map({
                    'None-Minimal': 0, 'Mild Depression': 1, 'Moderate Depression': 2,
                    'Moderately severe Depression': 3, 'Severe Depression': 4
                })
            elif target == 'Anxiety':
                df_processed[target] = df_processed[target].map({
                    'None-Minimal': 0, 'Mild Anxiety': 1, 'Moderate Anxiety': 2, 'Severe Anxiety': 3
                })
            elif target == 'Personally Disorder':
                df_processed[target] = df_processed[target].map({'Not affected': 0, 'Affected': 1})
            elif target == 'PTSD':
                df_processed[target] = df_processed[target].map({'Not affected': 0, 'Subthreshold PTSD': 1, 'PTSD': 2})
            
            df_processed[target] = df_processed[target].fillna(0).astype(int)

    # --- NLP Processing (Needs type column) ---
    def clean_text(text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    if 'Needs type' in df_processed.columns:
        df_processed['Needs type_cleaned'] = df_processed['Needs type'].apply(clean_text)

        tfidf = TfidfVectorizer(max_features=20, min_df=2, max_df=0.8,
                                stop_words='english', ngram_range=(1, 2))
        tfidf = TfidfVectorizer(max_features=20, min_df=2, max_df=0.8,
                        stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(df_processed['Needs type_cleaned'])

        tfidf_features = pd.DataFrame(
             tfidf_matrix.toarray(),  # ✅ use the matrix, not the vectorizer
             columns=[f"needs_tfidf_{f}" for f in tfidf.get_feature_names_out()],
             index=df_processed.index
            )

        df_processed = pd.concat([df_processed, tfidf_features], axis=1)
        df_processed.drop(['Needs type', 'Needs type_cleaned', 'why no', 'good workings'],
                  axis=1, errors='ignore', inplace=True)

        df_processed = pd.concat([df_processed, tfidf_features], axis=1)
        df_processed.drop(['Needs type', 'Needs type_cleaned', 'why no', 'good workings'],
                          axis=1, errors='ignore', inplace=True)

    # --- Label Encoding for remaining categoricals ---
    le = LabelEncoder()
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))

    # --- Derived features ---
    symptom_cols = [c for c in df_processed.columns if any(w in c.lower() for w in 
                    ['feel', 'sleeping', 'eating', 'focus', 'stressed', 'tired'])]
    if symptom_cols:
        df_processed['total_symptoms_score'] = df_processed[symptom_cols].sum(axis=1)

    trauma_cols = [c for c in ['witnessed direct violence', 'hurt or relatives hard',
                               'lost persons', 'you arrested', 'Your property destroyed']
                   if c in df_processed.columns]
    if trauma_cols:
        df_processed['trauma_exposure_score'] = df_processed[trauma_cols].sum(axis=1)

    social_cols = [c for c in ['feel alone', 'feel free to talk', 'close friends']
                   if c in df_processed.columns]
    if social_cols:
        social_score = 0
        for col in social_cols:
            if col == 'feel alone':
                social_score += (5 - df_processed[col])
            else:
                social_score += df_processed[col]
        df_processed['social_support_score'] = social_score

    # Fill missing
    df_processed = df_processed.fillna(0)

    return df_processed

# -----------------
# 🔹 Data Balancing (SMOTE) + Scaling + PCA
# -----------------
def prepare_ml_data(df_processed, target_columns, use_pca=True):
    """Apply SMOTE, scaling, and PCA, return X, y"""
    smote_datasets = []
    for target in target_columns:
        if target in df_processed.columns:
            X = df_processed.drop(columns=target_columns)
            y = df_processed[target]
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_res, columns=X.columns)
            df_resampled[target] = y_res
            smote_datasets.append(df_resampled)

    df_aug = pd.concat(smote_datasets, ignore_index=True).drop_duplicates()

    X_final = df_aug.drop(columns=target_columns)
    y_final = df_aug[target_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    if use_pca:
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, y_final, scaler, pca
    else:
        return X_scaled, y_final, scaler, None
