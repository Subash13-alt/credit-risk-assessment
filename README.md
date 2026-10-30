# CreditSense AI

**Intelligent Credit Risk Assessment using Machine Learning, CIBIL-Aware Banking Logic, and Unsupervised Learning**

CreditSense AI is a **multi-page** Flask web application that scores loan
applicants as **Low Risk** or **High Risk** using a full banking-style
application form (including CIBIL score, KYC/PAN verification, collateral,
and co-applicant income), then goes further than a bare classification: it
attaches a real loan offer (interest rate, EMI, totals), explains *why* the
model decided what it decided, and runs genuine **unsupervised learning**
(K-Means customer segmentation + Isolation Forest anomaly detection) on the
same applicant pool — all without ever using the Risk label during fitting.

---

## 1. Project Overview

| | |
|---|---|
| **Goal** | Score loan applicants as Low/High Risk, and separately discover natural borrower segments and unusual applications without labels |
| **Supervised task** | Binary classification (3 candidate models, best auto-selected) |
| **Unsupervised task** | K-Means customer segmentation + PCA visualization, and Isolation Forest anomaly detection |
| **Architecture** | Multi-page Flask app — 7 distinct pages, shared layout, shared stylesheet/JS |
| **Backend** | Python + Flask, SQLite logging, PDF export, EMI/interest-rate calculator, REST API |
| **Frontend** | Server-rendered Jinja2 templates + vanilla JS (no frontend framework) |

### Why CIBIL score and other banking fields?

The original version scored applicants on 11 generic attributes. Real
Indian lending applications are built around the **CIBIL score** (a
300–900 bureau score, the single biggest driver of a real approval/interest
rate decision), plus **KYC/PAN verification**, **collateral**,
**co-applicant income**, **existing bank relationship tenure**, and
**residential status**. This version adds all of them as genuine model
features (not decorative fields) — the synthetic dataset's underlying risk
formula weights CIBIL score as the dominant factor, exactly as a real
underwriting model would.

### Why both supervised *and* unsupervised learning?

"Is this applicant Low or High Risk?" is supervised — we have labeled
historical outcomes. But two other genuinely useful banking questions are
unsupervised:

1. **"What natural groups of borrowers exist in our pool?"** — answered by
   K-Means customer segmentation on the **Segments** dashboard, without the
   algorithm ever seeing the Risk label.
2. **"Which applications look unusual and deserve manual review?"** —
   answered by Isolation Forest anomaly detection on the **Anomalies**
   dashboard, again fit without labels.

In both cases, the Risk label is used **only afterwards**, to check how
well the unsupervised output happens to align with real outcomes (segment
risk purity; High Risk rate among flagged anomalies vs. the overall
population) — a standard way to sanity-check unsupervised results when
ground truth happens to be available.

---

## 2. Pages (Multi-Page Architecture)

| Page | Route | Purpose |
|---|---|---|
| **Assessment** | `/` | Full banking application form → risk verdict, CIBIL band, loan offer (EMI/interest), explanation, what-if sliders |
| **Batch Scoring** | `/batch` | Upload a CSV, score up to 500 applicants at once |
| **Analytics** | `/dashboard` | Usage stats: risk mix, CIBIL averages, volume over time, recent activity (from SQLite log) |
| **Model Performance** | `/dashboard/models` | Compare all 3 supervised models: accuracy/precision/recall/F1 + confusion matrix |
| **Segments** | `/dashboard/segments` | K-Means customer segmentation: PCA scatter plot, per-segment persona and financial profile |
| **Anomalies** | `/dashboard/anomalies` | Isolation Forest anomaly detection: flagged applicants, scatter plot, manual-review queue |
| **API Docs** | `/api-docs` | Full REST API reference with request/response examples and cURL |

Every page shares one layout (`templates/base.html`) with a persistent top
navigation bar, one stylesheet (`static/style.css`), and one small shared
script (`static/app.js` — theme toggle, toast notifications, CIBIL band
color helper, active-link highlighting).

---

## 3. Real-World Banking Features (Beyond Bare Classification)

These are the features that make this feel like an actual banking tool
rather than a label-only demo:

### CIBIL Score, front and center
Every prediction shows a CIBIL score strip — a colored band (Poor / Fair /
Good / Excellent, matching real CIBIL conventions) with a marker showing
exactly where the applicant lands, on the result card, in history, and in
the PDF report.

### Loan Offer Calculator (interest rate + EMI)
`loan_offer.py` implements real lending-style pricing: the interest rate is
tiered by CIBIL score band and risk classification (with a discount for
collateral, mirroring how secured loans price lower), and the standard EMI
formula computes the monthly installment, total interest, and total
payable. Applicants below a hard CIBIL floor are marked ineligible for
standard financing with a clear reason, instead of being given a nonsense
rate. This is deterministic banking arithmetic layered on top of the ML
prediction — not itself a model — but it's what turns "Low Risk, 93%" into
something a loan officer or applicant can actually act on.

### Load Sample Applicant (from real data, not a fixture)
The Assessment page's **Load Sample Applicant** button calls `GET
/sample-applicant`, which pulls one random row directly from
`dataset/credit_risk.csv` and autofills the entire form — including the
applicant's *actual* label from the dataset, shown in a toast, so you can
compare the model's live prediction against the ground truth it was trained
on. This is real data flowing through the UI, not a hard-coded string.

### Explainability that survives confident predictions
The ablation-based explanation (`explainability.py`) initially broke on
confidently-classified applicants: once Logistic Regression became the
winning model, predicted probabilities for clear-cut cases saturate to
~100%/~0%, and naively measuring "probability drop when a feature is
removed" collapsed to zero for every feature — precisely on the cases most
worth explaining. The fix: when the model exposes `decision_function()`
(true for Logistic Regression), impact is measured in **log-odds space**
instead, which is unbounded and never saturates, so the explanation stays
meaningful even for a 100%-confidence verdict. Models without
`decision_function` (Random Forest, Decision Tree) fall back to the
probability-based method, which doesn't have this problem since their
outputs are averaged/voted rather than a single saturating sigmoid.

### What-If Analysis
Live sliders for CIBIL score, income, loan amount, and DTI re-call
`/predict` (debounced) and update the verdict, gauge, and loan offer in
place — answering "how much would this applicant's CIBIL score need to
improve to flip the verdict?" without resubmitting the form.

---

## 4. Unsupervised Learning (K-Means Segmentation + Isolation Forest)

Implemented in `clustering.py`, run automatically at the end of
`train_model.py`, reusing the same fitted `StandardScaler`/encoders as the
supervised model but applied to the **full** applicant pool.

### 4.1 K-Means Customer Segmentation
K-Means (`k=5`) is fit on the scaled feature matrix — income, CIBIL score,
DTI, loan amount, employment, collateral, and every other feature — with no
access to the Risk label. For each discovered segment we report size,
average financial profile, dominant employment status, collateral rate,
and (evaluation-only) the dominant true Risk outcome and its purity. A
small heuristic assigns each segment a human-readable persona (e.g. *"Prime
Creditworthy Applicants"*, *"Unemployed Applicants, Weak Credit History"*,
*"Secured Applicants (Collateral-Backed)"*) based on its centroid — purely
descriptive, never fed back into the algorithm.

### 4.2 PCA (Principal Component Analysis)
The scaled feature matrix is dense (not sparse, unlike text data), so PCA —
rather than Truncated SVD, which is for sparse matrices — reduces it to 2
dimensions for the **Segments** dashboard's scatter plot. Toggle the plot
between "color by segment" (what K-Means discovered) and "color by true
risk" (the real label) to visually compare them.

### 4.3 Isolation Forest Anomaly Detection
Isolation Forest (`contamination=0.05`) flags the ~5% of applicants whose
financial profile is most unusual relative to the rest of the pool — a
real banking use case for fraud triage and data-quality review, entirely
unsupervised. On the bundled dataset, flagged anomalies have a **70% High
Risk rate vs. 38% overall** — a strong (if imperfect, exactly as expected
from an unsupervised method) signal that "unusual" and "risky" correlate,
without the algorithm ever being told what "risky" means.

### 4.4 Evaluation
| Metric | Type | What it measures |
|---|---|---|
| **Silhouette Score** | Intrinsic (no labels needed) | How well-separated the K-Means segments are |
| **Segment Risk Purity** | Extrinsic (evaluation-only) | Share of each segment's applicants belonging to its dominant true Risk outcome |
| **Anomaly High-Risk Rate** | Extrinsic (evaluation-only) | High Risk rate among flagged anomalies vs. the overall population |

---

## 5. Folder Structure

```
credit-risk-assessment/
│
├── app.py                   # Flask backend — all page + API routes
├── train_model.py           # Supervised pipeline: data → clean → train → evaluate → save
├── clustering.py              # Unsupervised pipeline: K-Means segmentation + PCA, Isolation Forest
├── loan_offer.py                # Interest rate tiering + EMI calculator (real banking arithmetic)
├── database.py                    # SQLite logging that powers the Analytics dashboard
├── explainability.py                # Ablation-based "why this prediction" (log-odds aware)
├── pdf_report.py                      # PDF report generator (fpdf2)
├── requirements.txt
├── README.md
│
├── dataset/
│   └── credit_risk.csv            # Generated training dataset (4,000 rows, incl. CIBIL score etc.)
│
├── model/
│   ├── model.pkl                     # Best trained supervised model (auto-selected)
│   ├── scaler.pkl                      # StandardScaler + LabelEncoders + feature order bundle
│   ├── metrics.json                      # Accuracy/Precision/Recall/F1/Confusion Matrix, all 3 models
│   ├── segments.json                       # K-Means segmentation report + PCA 2D coordinates
│   ├── anomalies.json                        # Isolation Forest anomaly report
│   └── predictions.db                          # SQLite log of predictions (created on first run)
│
├── static/
│   ├── style.css                    # Shared design system
│   └── app.js                         # Shared JS: theme toggle, toast, CIBIL band helper, nav highlighting
│
└── templates/
    ├── base.html                 # Shared layout: header/nav/footer, Jinja blocks for page content
    ├── index.html                  # Assessment page (the form + result + loan offer + what-if)
    ├── batch.html                    # Batch Scoring page
    ├── dashboard.html                  # Analytics dashboard
    ├── model_dashboard.html              # Model Performance dashboard
    ├── segments_dashboard.html             # Customer Segmentation dashboard (unsupervised)
    ├── anomalies_dashboard.html              # Anomaly Detection dashboard (unsupervised)
    └── api_docs.html                           # API reference page
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
cd credit-risk-assessment

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Trains the classifier AND runs K-Means segmentation + Isolation Forest anomaly detection
python train_model.py

python app.py
```

The app will be available at **http://127.0.0.1:5000**. Click **Load
Sample Applicant** on the Assessment page to try it instantly with a real
row from the training data, or jump straight to `/dashboard/segments` or
`/dashboard/anomalies` to see the unsupervised learning results.

> `model.pkl`, `scaler.pkl`, `segments.json`, and `anomalies.json` are
> already included in this repo, so you can skip straight to `python
> app.py` if you don't need to retrain.

---

## 7. Dataset Information

- **Format:** CSV with 17 columns: the original 11 applicant attributes
  plus 6 new banking fields — `Cibil_Score`, `Pan_Verified`,
  `Collateral_Provided`, `Co_Applicant_Income`, `Bank_Relationship_Years`,
  `Residential_Status` — and a `Risk` target.
- **Size:** 4,000 rows (`n_samples=4000` in `train_model.py`, adjustable)
- **Source:** Synthetically generated (no internet access to public
  dataset repositories in this sandbox). CIBIL score is generated
  correlated with `Credit_History` (mirroring how a bureau score and
  self-reported history track the same underlying behavior in reality),
  and the risk-scoring formula weights CIBIL as the dominant driver, with
  employment status, DTI, existing loans, loan-to-income ratio, KYC
  verification, collateral, and bank relationship tenure as secondary
  factors, plus noise for realistic class overlap.
- **Swapping in a real dataset:** replace `dataset/credit_risk.csv` with
  your own file using the same 17 columns and re-run `python
  train_model.py` — every downstream step (supervised training,
  segmentation, anomaly detection) re-runs automatically.

---

## 8. Model Training Steps

```bash
python train_model.py
```

This runs, in order:
1. **Supervised pipeline** — load/generate data → clean (median/mode
   imputation) → label-encode categoricals → 80/20 stratified split →
   `StandardScaler` fit on training data only → train **Logistic
   Regression**, **Decision Tree**, **Random Forest** → evaluate all 3 on
   Accuracy/Precision/Recall/F1/Confusion Matrix → auto-select the best by
   accuracy → save `model.pkl`, `scaler.pkl`, `metrics.json`.
2. **Unsupervised pipeline** (`clustering.py`, reusing the same fitted
   scaler/encoders but running on the *full* applicant pool) — fit K-Means
   (`k=5`) → evaluate with Silhouette Score and segment risk purity → fit
   PCA for 2D visualization coordinates → fit Isolation Forest for anomaly
   detection → save `segments.json`, `anomalies.json`.

Current results on the bundled dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **86.00%** | **82.43%** | **80.26%** | **81.33%** |
| Random Forest | 82.88% | 82.49% | 69.74% | 75.58% |
| Decision Tree | 77.62% | 70.63% | 70.39% | 70.51% |

(Full precision/recall/F1 breakdown for all 3 models is in
`model/metrics.json` and viewable live on the **Model Performance**
dashboard — they land in this range after adding CIBIL score as a strongly
linear-separable driver, which is exactly why Logistic Regression now edges
out the tree ensembles.)

| Unsupervised result | Value |
|---|---|
| K-Means Silhouette Score (k=5) | ≈ 0.077 |
| Isolation Forest anomaly rate | 5.0% (200 of 4,000 applicants) |
| High Risk rate — overall population | 38.0% |
| High Risk rate — among flagged anomalies | 70.0% |

---

## 9. Running the Application

```bash
python app.py
```

### Page routes
See the table in **Section 2** above.

### API routes

| Method | Route | Description |
|---|---|---|
| `POST` | `/predict` | Score one applicant; returns risk, CIBIL band, loan offer, explanation |
| `GET` | `/sample-applicant` | Returns one random real applicant from the training dataset |
| `POST` | `/batch-predict` | Upload a CSV (`file` field) → score every row |
| `GET` | `/batch-predict/sample` | Downloads a sample CSV template |
| `POST` | `/export-pdf` | Returns a downloadable PDF report for one prediction |
| `GET` | `/metrics` | Supervised model evaluation metrics (all 3 models) |
| `GET` | `/dashboard-data` | Usage analytics (from SQLite log) |
| `GET` | `/segments-data` | K-Means segmentation report + PCA scatter coordinates |
| `GET` | `/anomalies-data` | Isolation Forest anomaly report |
| `GET` | `/postman-collection.json` | Downloadable Postman collection |

Full documentation with request/response examples for every endpoint is
available at `/api-docs` when the app is running.

---

## 10. Deployment (Render / Railway)

**Build command:**
```bash
pip install -r requirements.txt && python train_model.py
```

**Start command:**
```bash
gunicorn app:app
```

`app.py` reads the `PORT` environment variable automatically.

---

## 11. Future Enhancements

- Swap the synthetic dataset for a large real-world credit bureau dataset
- Add SHAP for formally rigorous per-prediction explanations (current
  method is a documented, simpler ablation approximation)
- Automatically choose `k` for K-Means (elbow method / silhouette sweep)
  instead of hard-coding it
- Add fairness/bias auditing across demographic groups — a real
  regulatory requirement before any real lending deployment
- Add user authentication and per-branch/per-officer dashboards
- Add automated tests (pytest) and CI for the Flask routes and ML pipeline
- Containerize with Docker for more consistent deployment

---

## Disclaimer

CreditSense AI is an academic / educational project. Predictions,
CIBIL-based scoring, and loan offers (interest rate, EMI) are generated by
a machine learning model and rule-based calculations trained/configured on
**synthetic** data, and must **not** be used for real lending or credit
decisions.
