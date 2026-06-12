import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import json
import os

print("--- Phase 1: Model Training & Serialization ---")

dataset_path = "DataSet.csv"
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found!")
    exit(1)

# Load data
print("Loading dataset...")
df = pd.read_csv(dataset_path)

# Drop leakages
leaks = ['Unnamed: 0', 'F3912', 'F2230']
df_clean = df.drop(columns=[c for c in leaks if c in df.columns])

# Load hybrid features list
hybrid_features_path = "hybrid_features.json"
if os.path.exists(hybrid_features_path):
    with open(hybrid_features_path, "r") as f:
        hybrid_info = json.load(f)
        hybrid_features = hybrid_info["hybrid_features"]
    print(f"Loaded {len(hybrid_features)} hybrid features from {hybrid_features_path}.")
else:
    print("hybrid_features.json not found. Generating hybrid features...")
    # Preprocess categorical features temporarily to find importances
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    df_temp = df_clean.copy()
    for col in cat_cols:
        df_temp[col] = LabelEncoder().fit_transform(df_temp[col].fillna('Missing').astype(str))
    
    # Train full XGBoost to get features
    y_temp = df_temp['F3924']
    X_temp = df_temp.drop(columns=['F3924'])
    
    neg_count = (y_temp == 0).sum()
    pos_count = (y_temp == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    print("Training temporary XGBoost model for feature selection...")
    model_temp = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
    )
    model_temp.fit(X_temp, y_temp)
    
    importances = model_temp.feature_importances_
    feat_imp = pd.Series(importances, index=X_temp.columns).sort_values(ascending=False)
    top_150 = feat_imp.head(150).index.tolist()
    
    bank_features = ["F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122", 
                     "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887", "F3889", "F3891", "F3894"]
    hybrid_features = list(set(top_150 + bank_features))
    
    # Save hybrid features
    with open(hybrid_features_path, "w") as f:
        json.dump({
            "hybrid_features": hybrid_features,
            "top_150_features": top_150,
            "bank_recommended_features": bank_features
        }, f, indent=4)
    print(f"Saved {len(hybrid_features)} features to {hybrid_features_path}.")

# Filter dataset to hybrid features + target F3924
df_hybrid = df_clean[hybrid_features + ['F3924']].copy()

# Compute Medians for Imputation
print("Computing column medians...")
medians = {}
for col in hybrid_features:
    if pd.api.types.is_numeric_dtype(df_hybrid[col]):
        median_val = float(df_hybrid[col].median())
        # Replace NaN with median for model training
        df_hybrid[col] = df_hybrid[col].fillna(median_val)
        medians[col] = median_val

# Process Categorical Encoders
print("Encoding categorical columns...")
categorical_mappings = {}
cat_cols = df_hybrid.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    # Replace NaN with 'Missing'
    df_hybrid[col] = df_hybrid[col].fillna('Missing')
    df_hybrid[col] = le.fit_transform(df_hybrid[col].astype(str))
    # Save the encoder classes as a list to map later
    categorical_mappings[col] = le.classes_.tolist()

# Train-test split (stratified, shuffled)
y = df_hybrid['F3924']
X = df_hybrid.drop(columns=['F3924'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"Train samples: {len(X_train)} (Mules: {y_train.sum()})")
print(f"Test samples: {len(X_test)} (Mules: {y_test.sum()})")

# 1. Train XGBoost
print("\nTraining final XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# XGBoost Evaluation
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_cm = confusion_matrix(y_test, xgb_preds)
xgb_ap = average_precision_score(y_test, xgb_probs)
xgb_rep = classification_report(y_test, xgb_preds, output_dict=True)

print("XGBoost Confusion Matrix:")
print(xgb_cm)
print(f"XGBoost PR-AUC: {xgb_ap:.4f}")

# 2. Train LightGBM
print("\nTraining final LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.03,
    min_child_samples=8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train)

# LightGBM Evaluation
lgb_preds = lgb_model.predict(X_test)
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
lgb_cm = confusion_matrix(y_test, lgb_preds)
lgb_ap = average_precision_score(y_test, lgb_probs)
lgb_rep = classification_report(y_test, lgb_preds, output_dict=True)

print("LightGBM Confusion Matrix:")
print(lgb_cm)
print(f"LightGBM PR-AUC: {lgb_ap:.4f}")

# Serialize Models
print("\nSaving model files...")
xgb_model.save_model("xgb_hybrid_model.json")
lgb_model.booster_.save_model("lgb_hybrid_model.txt")

# Serialize Preprocessing Metadata & Metrics
metadata = {
    "hybrid_features": hybrid_features,
    "medians": medians,
    "categorical_mappings": categorical_mappings,
    "metrics": {
        "xgboost": {
            "recall": float(xgb_rep['1']['recall']),
            "precision": float(xgb_rep['1']['precision']),
            "f1_score": float(xgb_rep['1']['f1-score']),
            "pr_auc": float(xgb_ap),
            "confusion_matrix": {
                "tp": int(xgb_cm[1, 1]),
                "fp": int(xgb_cm[0, 1]),
                "fn": int(xgb_cm[1, 0]),
                "tn": int(xgb_cm[0, 0])
            }
        },
        "lightgbm": {
            "recall": float(lgb_rep['1']['recall']),
            "precision": float(lgb_rep['1']['precision']),
            "f1_score": float(lgb_rep['1']['f1-score']),
            "pr_auc": float(lgb_ap),
            "confusion_matrix": {
                "tp": int(lgb_cm[1, 1]),
                "fp": int(lgb_cm[0, 1]),
                "fn": int(lgb_cm[1, 0]),
                "tn": int(lgb_cm[0, 0])
            }
        }
    }
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("model_metadata.json saved successfully.")
print("Model training and serialization completed!")
