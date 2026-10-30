"""
explainability.py
-------------------
Lightweight, model-agnostic "why did the model say this?" explanations
for CreditSense AI.

We use an ablation (occlusion) method: for each feature, we replace its
scaled value with 0 (the training-set mean, since features are
standardized) and measure how much that changes the model's confidence
in the *predicted class*. A feature whose removal causes a big drop was
contributing a lot of evidence toward that prediction.

Measuring "confidence" naively via predict_proba() has a real failure
mode: once a model is very confident (probability rounds to ~0% or
~100%), the probability is numerically saturated and ablating any single
feature barely moves it -- so every feature looks equally unimportant,
which is wrong and, worse, happens most often on exactly the confident
cases where an explanation is most wanted. To avoid this, whenever the
model exposes decision_function() (e.g. LogisticRegression) we measure
impact in log-odds space instead, which is unbounded and never saturates.
Models without decision_function (e.g. RandomForest, DecisionTree) fall
back to the predict_proba method, which works fine for them since their
probabilities are averaged/voted and rarely fully saturate.

This is NOT SHAP or LIME -- it's a simpler, dependency-free approximation
that is still a legitimate and widely used local explanation technique
(sometimes called "leave-one-out" or "occlusion" importance). It's
documented here plainly so nobody mistakes it for something more
sophisticated than it is.
"""

import numpy as np

FEATURE_LABELS = {
    "Age": "Age",
    "Annual_Income": "Annual Income",
    "Loan_Amount": "Loan Amount",
    "Loan_Duration": "Loan Duration",
    "Existing_Loans": "Existing Loans",
    "Debt_to_Income_Ratio": "Debt-to-Income Ratio",
    "Gender": "Gender",
    "Employment_Status": "Employment Status",
    "Credit_History": "Credit History",
    "Loan_Purpose": "Loan Purpose",
    "Cibil_Score": "CIBIL Score",
    "Pan_Verified": "PAN / KYC Verification",
    "Collateral_Provided": "Collateral Provided",
    "Co_Applicant_Income": "Co-Applicant Income",
    "Bank_Relationship_Years": "Bank Relationship (Years)",
    "Residential_Status": "Residential Status",
}


def _confidence_score(model, row, predicted_class_index):
    """Returns a scalar 'how confident is the model in predicted_class_index'
    score for one row, using log-odds (unbounded, never saturates) when
    available, otherwise predicted-class probability."""
    if hasattr(model, "decision_function"):
        raw = model.decision_function(row)[0]
        # Binary decision_function returns one score per sample: positive
        # favors class index 1, negative favors class index 0. Orient it
        # so a higher value always means "more confident in the class we
        # actually predicted", regardless of which class that is.
        return float(raw) if predicted_class_index == 1 else float(-raw)
    return float(model.predict_proba(row)[0][predicted_class_index])


def explain_prediction(model, feature_columns, scaled_row, predicted_class_index, top_n=5):
    """Returns the top_n features ranked by their local contribution to
    the model's confidence in the predicted class.

    Args:
        model: fitted classifier (predict_proba, and ideally decision_function)
        feature_columns: list of column names in the same order as scaled_row's columns
        scaled_row: 2D numpy array, shape (1, n_features), already scaled
        predicted_class_index: index into model.classes_ of the predicted class
        top_n: how many top contributing features to return

    Returns:
        list of dicts: [{"feature": ..., "impact": float, "direction": "supports"|"opposes"}, ...]
        sorted by absolute impact, descending. "impact" is in log-odds
        units when decision_function is available, otherwise percentage
        points of predicted-class probability -- in both cases, larger
        magnitude means a bigger contribution to this specific prediction.
    """
    baseline_score = _confidence_score(model, scaled_row, predicted_class_index)
    using_log_odds = hasattr(model, "decision_function")

    contributions = []
    for i, col in enumerate(feature_columns):
        modified = scaled_row.copy()
        modified[0][i] = 0.0  # neutralize this feature (its standardized mean)
        new_score = _confidence_score(model, modified, predicted_class_index)
        impact = baseline_score - new_score
        display_impact = round(impact, 3) if using_log_odds else round(impact * 100, 2)
        contributions.append({
            "feature": FEATURE_LABELS.get(col, col),
            "impact": display_impact,
            "direction": "supports" if impact > 0 else "opposes",
        })

    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions[:top_n]
