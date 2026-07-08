"""
train_model.py
----------------
CreditSense AI - Model Training Pipeline

This script performs the complete Machine Learning workflow for the
Credit Risk Assessment project:

    1. Dataset generation / loading
    2. Data cleaning & missing value handling
    3. Label encoding of categorical features
    4. Feature scaling
    5. Train / test split
    6. Training of multiple classification models
    7. Model evaluation & comparison
    8. Automatic selection of the best performing model
    9. Persisting the trained model, scaler/encoders, and evaluation report

NOTE ON DATASET:
This environment has no internet access to download a dataset directly
from a public repository (e.g. Kaggle / UCI). To keep the project fully
self-contained and reproducible, this script programmatically generates
a realistic credit-risk dataset whose feature set mirrors well known
public credit-risk datasets (German Credit Data, Kaggle "Credit Risk
Dataset", etc.) and whose target label is derived from an underlying
credit-risk scoring rule plus random noise (to simulate real-world
label imperfection). If you have access to a real dataset with the
same columns, simply replace `dataset/credit_risk.csv` with it and
re-run this script -- no other code changes are required.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Constants / Paths
# --------------------------------------------------------------------------
RANDOM_STATE = 42
DATASET_DIR = "dataset"
MODEL_DIR = "model"
DATASET_PATH = os.path.join(DATASET_DIR, "credit_risk.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

CATEGORICAL_COLUMNS = [
    "Gender",
    "Employment_Status",
    "Credit_History",
    "Loan_Purpose",
]
NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income",
    "Loan_Amount",
    "Loan_Duration",
    "Existing_Loans",
    "Debt_to_Income_Ratio",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "Risk"


# --------------------------------------------------------------------------
# 1. Dataset generation
# --------------------------------------------------------------------------
def generate_dataset(n_samples: int = 4000) -> pd.DataFrame:
    """Generates a realistic synthetic credit-risk dataset.

    The label (Low Risk / High Risk) is derived from a weighted credit
    scoring formula based on real-world risk factors, with added noise
    to simulate natural class overlap.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    genders = rng.choice(["Male", "Female"], size=n_samples)
    employment = rng.choice(
        ["Salaried", "Self-Employed", "Unemployed", "Student"],
        size=n_samples,
        p=[0.55, 0.25, 0.10, 0.10],
    )
    credit_history = rng.choice(
        ["Good", "Average", "Poor"], size=n_samples, p=[0.5, 0.32, 0.18]
    )
    loan_purpose = rng.choice(
        ["Home", "Car", "Education", "Business", "Personal", "Medical"],
        size=n_samples,
    )

    age = rng.integers(21, 65, size=n_samples)
    annual_income = np.round(rng.normal(55000, 22000, size=n_samples)).clip(9000, 250000)
    loan_amount = np.round(rng.normal(18000, 12000, size=n_samples)).clip(1000, 150000)
    loan_duration = rng.choice([6, 12, 24, 36, 48, 60, 72, 84], size=n_samples)
    existing_loans = rng.integers(0, 5, size=n_samples)
    debt_to_income = np.round(rng.normal(35, 15, size=n_samples), 1).clip(1, 95)

    names = [f"Applicant_{i+1:05d}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "Applicant_Name": names,
            "Age": age,
            "Gender": genders,
            "Employment_Status": employment,
            "Annual_Income": annual_income,
            "Loan_Amount": loan_amount,
            "Loan_Duration": loan_duration,
            "Credit_History": credit_history,
            "Existing_Loans": existing_loans,
            "Debt_to_Income_Ratio": debt_to_income,
            "Loan_Purpose": loan_purpose,
        }
    )

    # ---- Underlying credit risk scoring rule (used only to build labels) ----
    credit_history_score = df["Credit_History"].map({"Good": 0, "Average": 1, "Poor": 2})
    employment_score = df["Employment_Status"].map(
        {"Salaried": 0, "Self-Employed": 1, "Student": 1.5, "Unemployed": 3}
    )
    loan_to_income = df["Loan_Amount"] / df["Annual_Income"]

    risk_score = (
        credit_history_score * 2.2
        + employment_score * 1.6
        + (df["Debt_to_Income_Ratio"] / 100) * 3.0
        + df["Existing_Loans"] * 0.55
        + loan_to_income * 2.5
        + np.where((df["Age"] < 23) | (df["Age"] > 58), 0.6, 0.0)
    )

    # Add random noise to avoid a perfectly separable / unrealistic dataset
    noise = rng.normal(0, 0.9, size=n_samples)
    risk_score = risk_score + noise

    threshold = np.quantile(risk_score, 0.62)  # ~62% Low Risk, ~38% High Risk
    df[TARGET_COLUMN] = np.where(risk_score <= threshold, "Low Risk", "High Risk")

    # ---- Inject a small amount of missing values to simulate real data ----
    for col in ["Annual_Income", "Debt_to_Income_Ratio", "Employment_Status"]:
        missing_idx = rng.choice(df.index, size=int(0.015 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def load_or_create_dataset() -> pd.DataFrame:
    """Loads the dataset from disk if present, otherwise generates and saves it."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH):
        print(f"[INFO] Found existing dataset at '{DATASET_PATH}'. Loading it.")
        return pd.read_csv(DATASET_PATH)

    print("[INFO] No dataset found. Generating a synthetic credit-risk dataset...")
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"[INFO] Dataset saved to '{DATASET_PATH}' ({len(df)} rows).")
    return df


# --------------------------------------------------------------------------
# 2 & 3. Cleaning, missing values, encoding
# --------------------------------------------------------------------------
def clean_and_encode(df: pd.DataFrame):
    """Handles missing values, encodes categorical columns, and returns
    the cleaned feature matrix, target vector, and fitted encoders."""
    df = df.copy()

    # ---- Missing value handling ----
    for col in NUMERIC_COLUMNS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Drop rows with a missing target (shouldn't normally happen)
    df = df.dropna(subset=[TARGET_COLUMN])

    # ---- Label Encoding for categorical features ----
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    # Ensure "High Risk" = 1 and "Low Risk" = 0 for intuitive probability meaning
    target_encoder.classes_ = np.array(["Low Risk", "High Risk"])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Low Risk": 0, "High Risk": 1})

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y, encoders, target_encoder


# --------------------------------------------------------------------------
# 4-8. Scaling, training, evaluation, model selection
# --------------------------------------------------------------------------
def train_and_select_best_model(X, y):
    """Splits data, scales features, trains multiple models, evaluates them,
    and returns the best model along with all evaluation metrics."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in candidate_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
        }
        fitted_models[name] = model

        print(f"\n[MODEL] {name}")
        print(f"        Accuracy : {results[name]['accuracy']}%")
        print(f"        Precision: {results[name]['precision']}%")
        print(f"        Recall   : {results[name]['recall']}%")
        print(f"        F1 Score : {results[name]['f1_score']}%")
        print(f"        Confusion Matrix: {cm}")

    # ---- Select the best model based on accuracy ----
    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    best_model = fitted_models[best_model_name]

    print(f"\n[BEST MODEL] {best_model_name} selected "
          f"(Accuracy: {results[best_model_name]['accuracy']}%)")

    return best_model, best_model_name, scaler, results


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_or_create_dataset()
    print(f"[INFO] Dataset shape: {df.shape}")

    X, y, encoders, target_encoder = clean_and_encode(df)
    best_model, best_model_name, scaler, results = train_and_select_best_model(X, y)

    # ---- Persist model ----
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[INFO] Best model saved to '{MODEL_PATH}'")

    # ---- Persist scaler + encoders + feature order bundled together ----
    preprocessing_bundle = {
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
    }
    joblib.dump(preprocessing_bundle, SCALER_PATH)
    print(f"[INFO] Scaler & encoders saved to '{SCALER_PATH}'")

    # ---- Persist evaluation metrics for all models ----
    metrics_report = {
        "best_model": best_model_name,
        "results": results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_report, f, indent=4)
    print(f"[INFO] Evaluation metrics saved to '{METRICS_PATH}'")

    print("\n[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
