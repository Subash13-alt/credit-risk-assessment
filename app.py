"""
app.py
------
CreditSense AI - Flask Backend

Serves the single-page frontend and exposes a /predict API endpoint
that loads the trained model + preprocessing bundle and returns a
credit risk prediction for a given applicant.
"""

import os
import json
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
app = Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# --------------------------------------------------------------------------
# Load model & preprocessing bundle at startup
# --------------------------------------------------------------------------
model = None
preprocessing_bundle = None
evaluation_metrics = None
MODEL_LOAD_ERROR = None

try:
    model = joblib.load(MODEL_PATH)
    preprocessing_bundle = joblib.load(SCALER_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            evaluation_metrics = json.load(f)
except Exception as exc:  # noqa: BLE001
    # We don't crash the app on import -- we surface a clear error at
    # request time instead, so the frontend can display a helpful message.
    MODEL_LOAD_ERROR = str(exc)


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "applicant_name": str,
    "age": (int, float),
    "gender": str,
    "employment_status": str,
    "annual_income": (int, float),
    "loan_amount": (int, float),
    "loan_duration": (int, float),
    "credit_history": str,
    "existing_loans": (int, float),
    "debt_to_income_ratio": (int, float),
    "loan_purpose": str,
}

VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "employment_status": {"Salaried", "Self-Employed", "Unemployed", "Student"},
    "credit_history": {"Good", "Average", "Poor"},
    "loan_purpose": {"Home", "Car", "Education", "Business", "Personal", "Medical"},
}


def validate_payload(data: dict):
    """Validates the incoming JSON payload. Returns (is_valid, error_message)."""
    if not isinstance(data, dict):
        return False, "Invalid request format. Expected a JSON object."

    for field in REQUIRED_FIELDS:
        if field not in data or data[field] in (None, ""):
            return False, f"Missing required field: '{field}'."

    # Type / range checks
    try:
        age = float(data["age"])
        if not (18 <= age <= 100):
            return False, "Age must be between 18 and 100."

        annual_income = float(data["annual_income"])
        if annual_income <= 0:
            return False, "Annual income must be greater than 0."

        loan_amount = float(data["loan_amount"])
        if loan_amount <= 0:
            return False, "Loan amount must be greater than 0."

        loan_duration = float(data["loan_duration"])
        if loan_duration <= 0:
            return False, "Loan duration must be greater than 0."

        existing_loans = float(data["existing_loans"])
        if existing_loans < 0:
            return False, "Existing loans cannot be negative."

        dti = float(data["debt_to_income_ratio"])
        if not (0 <= dti <= 100):
            return False, "Debt-to-Income ratio must be between 0 and 100."
    except (ValueError, TypeError):
        return False, "One or more numeric fields contain invalid values."

    for field, valid_set in VALID_CATEGORIES.items():
        if str(data[field]) not in valid_set:
            return False, f"Invalid value for '{field}'."

    if not str(data["applicant_name"]).strip():
        return False, "Applicant name cannot be empty."

    return True, None


def build_feature_row(data: dict) -> pd.DataFrame:
    """Converts validated raw input into a properly ordered, encoded,
    and scaled feature row ready for model prediction."""
    numeric_columns = preprocessing_bundle["numeric_columns"]
    categorical_columns = preprocessing_bundle["categorical_columns"]
    feature_columns = preprocessing_bundle["feature_columns"]
    encoders = preprocessing_bundle["encoders"]
    scaler = preprocessing_bundle["scaler"]

    raw = {
        "Age": float(data["age"]),
        "Annual_Income": float(data["annual_income"]),
        "Loan_Amount": float(data["loan_amount"]),
        "Loan_Duration": float(data["loan_duration"]),
        "Existing_Loans": float(data["existing_loans"]),
        "Debt_to_Income_Ratio": float(data["debt_to_income_ratio"]),
        "Gender": str(data["gender"]),
        "Employment_Status": str(data["employment_status"]),
        "Credit_History": str(data["credit_history"]),
        "Loan_Purpose": str(data["loan_purpose"]),
    }

    row = pd.DataFrame([raw])

    # Apply saved label encoders to categorical columns
    for col in categorical_columns:
        le = encoders[col]
        row[col] = le.transform(row[col].astype(str))

    row = row[feature_columns]
    scaled = scaler.transform(row)
    return scaled


def get_recommendation(risk_label: str, probability: float) -> str:
    """Produces a human-readable recommendation based on the prediction."""
    if risk_label == "Low Risk":
        if probability >= 85:
            return "Eligible for Loan Approval"
        return "Eligible for Loan Approval (Minor Review Suggested)"
    else:
        if probability >= 80:
            return "High Default Risk"
        return "Manual Verification Required"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the single-page frontend."""
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    """Returns the saved model evaluation metrics (used by the frontend
    to optionally display model performance)."""
    if evaluation_metrics is None:
        return jsonify({"error": "Evaluation metrics not available."}), 404
    return jsonify(evaluation_metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts applicant data as JSON and returns a credit risk prediction."""
    if MODEL_LOAD_ERROR is not None or model is None or preprocessing_bundle is None:
        return jsonify({
            "error": "Model is not available on the server. "
                     "Please run 'python train_model.py' first."
        }), 500

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        is_valid, error_message = validate_payload(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        features = build_feature_row(data)

        prediction = model.predict(features)[0]  # 0 = Low Risk, 1 = High Risk
        probabilities = model.predict_proba(features)[0]

        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        confidence = round(float(probabilities[prediction]) * 100, 2)

        recommendation = get_recommendation(risk_label, confidence)

        response = {
            "risk": risk_label,
            "probability": confidence,
            "message": recommendation,
            "applicant_name": str(data["applicant_name"]).strip(),
        }
        return jsonify(response), 200

    except Exception as exc:  # noqa: BLE001
        # Log full traceback server-side for debugging, but return a
        # clean, generic error message to the client.
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected error occurred while processing your request.",
            "details": str(exc),
        }), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
