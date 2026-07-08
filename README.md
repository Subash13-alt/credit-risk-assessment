# CreditSense AI

**Intelligent Credit Risk Assessment System Using Machine Learning**

CreditSense AI is a single-page web application that predicts whether a loan
applicant is a **Low Risk** or **High Risk** borrower. It combines a
scikit-learn classification pipeline with a Flask backend and a modern,
responsive vanilla HTML/CSS/JS frontend — no frontend framework required.

---

## 1. Project Overview

| | |
|---|---|
| **Goal** | Predict loan applicant credit risk (Low Risk / High Risk) with a confidence score |
| **ML Models compared** | Logistic Regression, Decision Tree, Random Forest |
| **Model selection** | Best model auto-selected by test-set accuracy |
| **Backend** | Python + Flask (`GET /`, `POST /predict`, `GET /metrics`) |
| **Frontend** | Single `index.html` file — HTML, CSS, and JS embedded together |
| **Persistence** | Trained model + preprocessing pipeline saved with `joblib` |

### How a prediction works

```
User fills form → JS validates & sends POST /predict → Flask loads model.pkl
→ input is label-encoded & scaled → model predicts risk + probability
→ JSON returned → frontend updates the result card, no page reload
```

### Current model performance (on the bundled dataset)

The training script automatically evaluates all three models on a held-out
20% test split and picks the most accurate one. The values below are from
the last run (see `model/metrics.json` for the live numbers, and the
**Model Performance** panel in the app for a live view):

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 78.12% | 75.90% | 62.17% | 68.35% |
| Decision Tree | 85.25% | 81.00% | 79.93% | 80.46% |
| **Random Forest (selected)** | **89.25%** | **87.59%** | **83.55%** | **85.52%** |

### About the dataset

This sandbox environment has no internet access to pull a dataset directly
from a public repository (Kaggle, UCI, etc.). To keep the project fully
self-contained, reproducible, and free of missing files, `train_model.py`
**programmatically generates** a realistic credit-risk dataset
(`dataset/credit_risk.csv`, 4,000 rows) whose columns mirror well-known
public credit-risk datasets (e.g. German Credit Data / Kaggle "Credit Risk
Dataset"). The target label is derived from a weighted credit-scoring
formula (credit history, debt-to-income ratio, employment status, existing
loans, loan-to-income ratio, age) plus random noise, so the classes are
realistically overlapping rather than perfectly separable.

**To use a real public dataset instead:** replace `dataset/credit_risk.csv`
with your own file that has the same 11 feature columns + a `Risk` column
(values `Low Risk` / `High Risk`), then re-run `python train_model.py`. No
other code changes are required.

---

## 2. Folder Structure

```
credit-risk-assessment/
│
├── app.py                 # Flask backend (routes, validation, prediction)
├── train_model.py         # Full ML pipeline: data → clean → train → evaluate → save
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── dataset/
│   └── credit_risk.csv    # Generated training dataset (4,000 rows)
│
├── model/
│   ├── model.pkl           # Best trained model (auto-selected)
│   ├── scaler.pkl           # StandardScaler + LabelEncoders + feature order bundle
│   └── metrics.json         # Evaluation metrics for all trained models
│
└── templates/
    └── index.html          # Single-page frontend (HTML + CSS + JS in one file)
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone / unzip the project, then move into it
cd credit-risk-assessment

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (creates dataset/, model/model.pkl, model/scaler.pkl, model/metrics.json)
python train_model.py

# 5. Run the Flask app
python app.py
```

The app will be available at **http://127.0.0.1:5000**.

> `model.pkl` and `scaler.pkl` are already included in this repo, so you can
> skip straight to step 5 if you don't need to retrain. Re-run
> `train_model.py` any time you change the dataset or want fresh metrics.

---

## 4. How to Train the Model

```bash
python train_model.py
```

This will:
1. Load `dataset/credit_risk.csv` if present, otherwise generate it.
2. Handle missing values (median for numeric columns, mode for categorical).
3. Label-encode categorical columns (`Gender`, `Employment_Status`,
   `Credit_History`, `Loan_Purpose`).
4. Split the data 80/20 (stratified) into train/test sets.
5. Scale numeric + encoded features with `StandardScaler`.
6. Train **Logistic Regression**, **Decision Tree**, and **Random Forest**.
7. Evaluate each on Accuracy, Precision, Recall, F1 Score, and a Confusion
   Matrix.
8. Automatically select the model with the highest accuracy.
9. Save `model/model.pkl`, `model/scaler.pkl` (bundled with encoders and
   feature order), and `model/metrics.json`.

## 5. How to Run the Flask App

```bash
python app.py
```

Routes exposed:

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page frontend (`templates/index.html`) |
| `POST` | `/predict` | Accepts applicant JSON, returns a risk prediction |
| `GET` | `/metrics` | Returns saved evaluation metrics for the Model Performance panel |

### Example `POST /predict` request

```json
{
  "applicant_name": "Priya Sharma",
  "age": 32,
  "gender": "Female",
  "employment_status": "Salaried",
  "annual_income": 650000,
  "loan_amount": 150000,
  "loan_duration": 36,
  "credit_history": "Good",
  "existing_loans": 1,
  "debt_to_income_ratio": 22,
  "loan_purpose": "Home"
}
```

### Example response

```json
{
  "risk": "Low Risk",
  "probability": 93.47,
  "message": "Eligible for Loan Approval",
  "applicant_name": "Priya Sharma"
}
```

Invalid or missing fields return `HTTP 400` with a descriptive `error`
message; unexpected server-side issues return `HTTP 500`.

---

## 6. Frontend Features

- Single responsive page (`templates/index.html`) — no external CSS/JS files
- White / blue / dark navy color palette with cards, shadows, and rounded corners
- Animated SVG risk gauge that fills based on prediction confidence
- Tooltips on every input field explaining what it means
- Full client-side input validation (type, range, required fields) before submit
- Loading spinner while the prediction request is in flight
- Reset button to clear the form and result
- Prediction history stored in the browser's `localStorage` (persists across visits)
- Dark mode toggle (persisted via `localStorage`, respects system preference on first visit)
- Live "Model Performance" panel (accuracy / precision / recall / F1 / confusion matrix)
- Professional footer with disclaimer

---

## 7. Model Evaluation Details

All four core metrics — **Accuracy, Precision, Recall, F1 Score** — plus a
**Confusion Matrix** are computed for every candidate model during training
and saved to `model/metrics.json`. The frontend fetches this file (via the
`/metrics` route) and renders it live in the **Model Performance** panel,
including a 2×2 confusion matrix grid for the winning model.

---

## 8. Deployment (Render / Railway)

Both platforms can run this app directly with **gunicorn**, already listed
in `requirements.txt`.

**Build command:**
```bash
pip install -r requirements.txt && python train_model.py
```

**Start command:**
```bash
gunicorn app:app
```

Make sure the platform's port is honored — `app.py` already reads the
`PORT` environment variable (`os.environ.get("PORT", 5000)`), which both
Render and Railway set automatically.

---

## 9. Screenshots

> _Add screenshots of the running application here before submission, e.g.:_
> - `screenshots/home-light.png` — Home page, light mode
> - `screenshots/home-dark.png` — Home page, dark mode
> - `screenshots/result-low-risk.png` — Low Risk result card
> - `screenshots/result-high-risk.png` — High Risk result card
> - `screenshots/model-performance.png` — Model performance panel

---

## 10. Future Enhancements

- Swap the synthetic dataset for a large real-world credit bureau dataset
- Add SHAP/feature-importance explanations alongside each prediction
- Add user authentication and per-user prediction history (server-side)
- Add a batch-upload (CSV) mode for assessing multiple applicants at once
- Add model retraining/versioning endpoints and a simple admin dashboard
- Add automated tests (pytest) and CI for the Flask routes and ML pipeline
- Containerize with Docker for more consistent deployment

---

## 11. Tech Stack Summary

**Frontend:** HTML5, CSS3 (custom properties, grid/flexbox, animations), vanilla JavaScript (Fetch API, localStorage)
**Backend:** Python, Flask
**Machine Learning:** pandas, NumPy, scikit-learn, joblib
**Deployment:** Render / Railway (gunicorn)

---

## Disclaimer

CreditSense AI is an academic / educational project. Predictions are
generated by a machine learning model trained on **synthetic** data and
must **not** be used for real lending or credit decisions.
