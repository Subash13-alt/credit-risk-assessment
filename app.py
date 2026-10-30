"""
app.py
------
CreditSense AI - Flask Backend (multi-page app)

Pages:
    GET  /                        Assessment page (single applicant)
    GET  /batch                    Batch Scoring page (CSV upload UI)
    GET  /dashboard                  Analytics dashboard (usage over time)
    GET  /dashboard/models             Model Performance dashboard (supervised)
    GET  /dashboard/segments             Customer Segmentation dashboard (unsupervised: K-Means + PCA)
    GET  /dashboard/anomalies              Anomaly Detection dashboard (unsupervised: Isolation Forest)
    GET  /api-docs                           API documentation page

API:
    POST /predict                  Single applicant prediction (+ explanation)
    POST /batch-predict              CSV upload -> risk-score many applicants at once
    GET  /batch-predict/sample        Downloadable sample CSV template
    POST /export-pdf                    Downloadable PDF report for one prediction
    GET  /metrics                         Saved supervised model evaluation metrics
    GET  /dashboard-data                    JSON stats for the Analytics dashboard
    GET  /segments-data                       JSON data for the Customer Segmentation dashboard
    GET  /anomalies-data                        JSON data for the Anomaly Detection dashboard
    GET  /postman-collection.json                 Downloadable Postman collection
"""

import io
import os
import json
import csv
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file, Response

import database
from explainability import explain_prediction
from pdf_report import build_prediction_report
from loan_offer import build_loan_offer

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------
app = Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
SEGMENTS_PATH = os.path.join(MODEL_DIR, "segments.json")
ANOMALIES_PATH = os.path.join(MODEL_DIR, "anomalies.json")
DATASET_PATH = os.path.join("dataset", "credit_risk.csv")

MAX_BATCH_ROWS = 500  # sanity cap so a huge CSV can't hang the request

database.init_db()

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
    "cibil_score": (int, float),
    "pan_verified": str,
    "collateral_provided": str,
    "co_applicant_income": (int, float),
    "bank_relationship_years": (int, float),
    "residential_status": str,
}

VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "employment_status": {"Salaried", "Self-Employed", "Unemployed", "Student"},
    "credit_history": {"Good", "Average", "Poor"},
    "loan_purpose": {"Home", "Car", "Education", "Business", "Personal", "Medical"},
    "pan_verified": {"Yes", "No"},
    "collateral_provided": {"Yes", "No"},
    "residential_status": {"Owned", "Rented", "Mortgaged"},
}

BATCH_CSV_COLUMNS = [
    "applicant_name", "age", "gender", "employment_status", "annual_income",
    "loan_amount", "loan_duration", "credit_history", "existing_loans",
    "debt_to_income_ratio", "loan_purpose", "cibil_score", "pan_verified",
    "collateral_provided", "co_applicant_income", "bank_relationship_years",
    "residential_status",
]


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

        cibil_score = float(data["cibil_score"])
        if not (300 <= cibil_score <= 900):
            return False, "CIBIL Score must be between 300 and 900."

        co_applicant_income = float(data["co_applicant_income"])
        if co_applicant_income < 0:
            return False, "Co-applicant income cannot be negative."

        bank_relationship_years = float(data["bank_relationship_years"])
        if not (0 <= bank_relationship_years <= 60):
            return False, "Bank relationship (years) must be between 0 and 60."
    except (ValueError, TypeError):
        return False, "One or more numeric fields contain invalid values."

    for field, valid_set in VALID_CATEGORIES.items():
        if str(data[field]) not in valid_set:
            return False, f"Invalid value for '{field}'."

    if not str(data["applicant_name"]).strip():
        return False, "Applicant name cannot be empty."

    return True, None


def build_feature_row(data: dict) -> np.ndarray:
    """Converts validated raw input into a properly ordered, encoded,
    and scaled feature row ready for model prediction."""
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
        "Cibil_Score": float(data["cibil_score"]),
        "Pan_Verified": str(data["pan_verified"]),
        "Collateral_Provided": str(data["collateral_provided"]),
        "Co_Applicant_Income": float(data["co_applicant_income"]),
        "Bank_Relationship_Years": float(data["bank_relationship_years"]),
        "Residential_Status": str(data["residential_status"]),
    }

    row = pd.DataFrame([raw])

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


def run_single_prediction(data: dict, include_explanation: bool = True):
    """Shared prediction logic used by both /predict and /batch-predict.
    Returns a response dict (never raises for validation errors -- returns
    an 'error' key instead so batch processing can continue row by row)."""
    is_valid, error_message = validate_payload(data)
    if not is_valid:
        return {"error": error_message}

    features = build_feature_row(data)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    confidence = round(float(probabilities[prediction]) * 100, 2)
    recommendation = get_recommendation(risk_label, confidence)

    response = {
        "risk": risk_label,
        "probability": confidence,
        "message": recommendation,
        "applicant_name": str(data["applicant_name"]).strip(),
        "cibil_score": int(float(data["cibil_score"])),
    }

    response["loan_offer"] = build_loan_offer(
        risk_label=risk_label,
        cibil_score=float(data["cibil_score"]),
        collateral_provided=str(data["collateral_provided"]),
        loan_amount=float(data["loan_amount"]),
        loan_duration=float(data["loan_duration"]),
    )

    if include_explanation:
        feature_columns = preprocessing_bundle["feature_columns"]
        response["explanation"] = explain_prediction(
            model, feature_columns, features, int(prediction), top_n=5
        )

    return response


# --------------------------------------------------------------------------
# Page Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    """Assessment page (home) -- single applicant form."""
    return render_template("index.html")


@app.route("/batch")
def batch_page():
    """Batch Scoring page (CSV upload UI)."""
    return render_template("batch.html")


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard showing aggregate stats across all predictions
    made through this server (single + batch)."""
    return render_template("dashboard.html")


@app.route("/dashboard/models")
def model_dashboard():
    """Model Performance dashboard: compares all 3 supervised models."""
    return render_template("model_dashboard.html")


@app.route("/dashboard/segments")
def segments_dashboard():
    """Unsupervised Learning dashboard: K-Means customer segmentation + PCA scatter plot."""
    return render_template("segments_dashboard.html")


@app.route("/dashboard/anomalies")
def anomalies_dashboard():
    """Unsupervised Learning dashboard: Isolation Forest anomaly detection."""
    return render_template("anomalies_dashboard.html")


@app.route("/api-docs")
def api_docs():
    """Human-readable API documentation page."""
    return render_template("api_docs.html")


# --------------------------------------------------------------------------
# API Routes
# --------------------------------------------------------------------------
@app.route("/metrics")
def metrics():
    """Returns the saved model evaluation metrics (used by the Model
    Performance dashboard)."""
    if evaluation_metrics is None:
        return jsonify({"error": "Evaluation metrics not available."}), 404
    return jsonify(evaluation_metrics)


@app.route("/sample-applicant")
def sample_applicant():
    """Returns one random real applicant row from the training dataset,
    formatted to match the assessment form's field names. Powers the
    'Load Sample Applicant' button so the demo pulls from actual data
    rather than a hard-coded fixture."""
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not available. Run train_model.py first."}), 404

    try:
        df = pd.read_csv(DATASET_PATH)
        df = df.dropna()  # sample only complete rows so every field is fillable
        if len(df) == 0:
            return jsonify({"error": "No complete rows available in the dataset."}), 404

        row = df.sample(n=1).iloc[0]

        sample = {
            "applicant_name": str(row["Applicant_Name"]),
            "age": int(row["Age"]),
            "gender": str(row["Gender"]),
            "employment_status": str(row["Employment_Status"]),
            "annual_income": round(float(row["Annual_Income"]), 0),
            "co_applicant_income": round(float(row["Co_Applicant_Income"]), 0),
            "loan_amount": round(float(row["Loan_Amount"]), 0),
            "loan_duration": int(row["Loan_Duration"]),
            "credit_history": str(row["Credit_History"]),
            "existing_loans": int(row["Existing_Loans"]),
            "debt_to_income_ratio": round(float(row["Debt_to_Income_Ratio"]), 1),
            "loan_purpose": str(row["Loan_Purpose"]),
            "cibil_score": int(round(float(row["Cibil_Score"]))),
            "pan_verified": str(row["Pan_Verified"]),
            "collateral_provided": str(row["Collateral_Provided"]),
            "bank_relationship_years": round(float(row["Bank_Relationship_Years"]), 1),
            "residential_status": str(row["Residential_Status"]),
            "actual_risk_in_dataset": str(row["Risk"]),
        }
        return jsonify(sample), 200

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Could not load a sample applicant: {exc}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts applicant data as JSON and returns a credit risk prediction
    with a short, ranked explanation of the top contributing factors."""
    if MODEL_LOAD_ERROR is not None or model is None or preprocessing_bundle is None:
        return jsonify({
            "error": "Model is not available on the server. "
                     "Please run 'python train_model.py' first."
        }), 500

    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        result = run_single_prediction(data, include_explanation=True)
        if "error" in result:
            return jsonify(result), 400

        database.log_prediction(
            applicant_name=result["applicant_name"],
            risk=result["risk"],
            probability=result["probability"],
            message=result["message"],
            source="single",
            annual_income=data.get("annual_income"),
            loan_amount=data.get("loan_amount"),
            credit_history=data.get("credit_history"),
            debt_to_income_ratio=data.get("debt_to_income_ratio"),
            cibil_score=data.get("cibil_score"),
        )

        return jsonify(result), 200

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected error occurred while processing your request.",
            "details": str(exc),
        }), 500


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Accepts a CSV file of applicants (multipart/form-data, field name
    'file') and returns a risk prediction for every row. Rows that fail
    validation are reported individually rather than failing the whole batch."""
    if MODEL_LOAD_ERROR is not None or model is None or preprocessing_bundle is None:
        return jsonify({"error": "Model is not available on the server."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please attach a CSV file under field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported."}), 400

    try:
        raw_bytes = file.read()
        text = raw_bytes.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not parse CSV file: {exc}"}), 400

    missing_cols = [c for c in BATCH_CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        return jsonify({
            "error": f"CSV is missing required column(s): {', '.join(missing_cols)}. "
                     f"Download the sample template from /batch-predict/sample for the expected format."
        }), 400

    if len(df) == 0:
        return jsonify({"error": "The uploaded CSV has no data rows."}), 400
    if len(df) > MAX_BATCH_ROWS:
        return jsonify({"error": f"Too many rows ({len(df)}). Maximum is {MAX_BATCH_ROWS} per batch."}), 400

    results = []
    success_count = 0
    for idx, row in df.iterrows():
        row_data = row.to_dict()
        try:
            result = run_single_prediction(row_data, include_explanation=False)
        except Exception as exc:  # noqa: BLE001
            result = {"error": f"Unexpected error: {exc}"}

        result["row"] = int(idx) + 1
        result["applicant_name"] = str(row_data.get("applicant_name", f"Row {idx + 1}"))

        if "error" not in result:
            success_count += 1
            database.log_prediction(
                applicant_name=result["applicant_name"],
                risk=result["risk"],
                probability=result["probability"],
                message=result["message"],
                source="batch",
                annual_income=row_data.get("annual_income"),
                loan_amount=row_data.get("loan_amount"),
                credit_history=row_data.get("credit_history"),
                debt_to_income_ratio=row_data.get("debt_to_income_ratio"),
                cibil_score=row_data.get("cibil_score"),
            )
        results.append(result)

    return jsonify({
        "total_rows": len(df),
        "successful": success_count,
        "failed": len(df) - success_count,
        "results": results,
    }), 200


@app.route("/batch-predict/sample")
def batch_predict_sample():
    """Downloadable sample CSV template for batch scoring."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(BATCH_CSV_COLUMNS)
    writer.writerow(["Priya Sharma", 32, "Female", "Salaried", 650000, 150000, 36,
                      "Good", 1, 22, "Home", 780, "Yes", "Yes", 0, 4, "Owned"])
    writer.writerow(["Rahul Verma", 45, "Male", "Unemployed", 30000, 120000, 36,
                      "Poor", 4, 80, "Personal", 490, "No", "No", 0, 0, "Rented"])
    csv_bytes = output.getvalue().encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=creditsense_batch_template.csv"},
    )


@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    """Generates a downloadable PDF report for a single prediction result.
    Expects the same shape returned by /predict, plus the original applicant inputs."""
    try:
        payload = request.get_json(silent=True)
        if not payload or "result" not in payload or "inputs" not in payload:
            return jsonify({"error": "Request must include 'result' and 'inputs'."}), 400

        result = payload["result"]
        inputs = payload["inputs"]
        explanation = payload.get("explanation", [])

        pdf_bytes = build_prediction_report(result, inputs, explanation)
        buffer = io.BytesIO(pdf_bytes)
        buffer.seek(0)

        applicant = str(result.get("applicant_name", "applicant")).strip().replace(" ", "_") or "applicant"
        filename = f"CreditSense_Report_{applicant}.pdf"

        return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=filename)

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Could not generate PDF: {exc}"}), 500


@app.route("/dashboard-data")
def dashboard_data():
    """JSON data powering the Analytics dashboard's charts."""
    try:
        stats = database.get_dashboard_stats()
        return jsonify(stats), 200
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Could not load dashboard data: {exc}"}), 500


@app.route("/segments-data")
def segments_data():
    """JSON data powering the Customer Segmentation dashboard (from
    train_model.py's unsupervised K-Means + PCA step)."""
    if not os.path.exists(SEGMENTS_PATH):
        return jsonify({"error": "Segmentation data not available. Run train_model.py first."}), 404
    with open(SEGMENTS_PATH, "r") as f:
        return jsonify(json.load(f)), 200


@app.route("/anomalies-data")
def anomalies_data():
    """JSON data powering the Anomaly Detection dashboard (from
    train_model.py's unsupervised Isolation Forest step)."""
    if not os.path.exists(ANOMALIES_PATH):
        return jsonify({"error": "Anomaly detection data not available. Run train_model.py first."}), 404
    with open(ANOMALIES_PATH, "r") as f:
        return jsonify(json.load(f)), 200


@app.route("/postman-collection.json")
def postman_collection():
    """Downloadable Postman collection covering all API endpoints."""
    base_url = request.host_url.rstrip("/")
    collection = {
        "info": {
            "name": "CreditSense AI API",
            "description": "Endpoints for the CreditSense AI credit risk assessment service.",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "item": [
            {
                "name": "Predict (single applicant)",
                "request": {
                    "method": "POST",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": {
                        "mode": "raw",
                        "raw": json.dumps({
                            "applicant_name": "Priya Sharma", "age": 32, "gender": "Female",
                            "employment_status": "Salaried", "annual_income": 650000,
                            "loan_amount": 150000, "loan_duration": 36, "credit_history": "Good",
                            "existing_loans": 1, "debt_to_income_ratio": 22, "loan_purpose": "Home",
                            "cibil_score": 780, "pan_verified": "Yes", "collateral_provided": "Yes",
                            "co_applicant_income": 0, "bank_relationship_years": 4,
                            "residential_status": "Owned",
                        }, indent=2),
                    },
                    "url": {"raw": f"{base_url}/predict", "host": [base_url], "path": ["predict"]},
                },
            },
            {
                "name": "Batch Predict (CSV upload)",
                "request": {
                    "method": "POST",
                    "body": {"mode": "formdata", "formdata": [{"key": "file", "type": "file", "src": ""}]},
                    "url": {"raw": f"{base_url}/batch-predict", "host": [base_url], "path": ["batch-predict"]},
                },
            },
            {
                "name": "Download Batch CSV Template",
                "request": {
                    "method": "GET",
                    "url": {"raw": f"{base_url}/batch-predict/sample", "host": [base_url],
                             "path": ["batch-predict", "sample"]},
                },
            },
            {
                "name": "Model Evaluation Metrics",
                "request": {"method": "GET", "url": {"raw": f"{base_url}/metrics", "host": [base_url], "path": ["metrics"]}},
            },
            {
                "name": "Dashboard Data (Analytics)",
                "request": {"method": "GET", "url": {"raw": f"{base_url}/dashboard-data", "host": [base_url], "path": ["dashboard-data"]}},
            },
            {
                "name": "Customer Segmentation Data (Unsupervised)",
                "request": {"method": "GET", "url": {"raw": f"{base_url}/segments-data", "host": [base_url], "path": ["segments-data"]}},
            },
            {
                "name": "Anomaly Detection Data (Unsupervised)",
                "request": {"method": "GET", "url": {"raw": f"{base_url}/anomalies-data", "host": [base_url], "path": ["anomalies-data"]}},
            },
        ],
    }
    return Response(
        json.dumps(collection, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=CreditSenseAI.postman_collection.json"},
    )


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
