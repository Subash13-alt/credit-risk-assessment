"""
clustering.py
--------------
Unsupervised Learning pipeline for CreditSense AI.

The core app is a *supervised* classifier (train_model.py). This module
runs two classic unsupervised techniques on the same scaled feature
matrix, purely to discover structure in the applicant pool without using
the Risk label during fitting:

    1. K-Means customer segmentation  -- groups applicants into K natural
                                           segments based on their financial
                                           profile (income, CIBIL score, DTI,
                                           loan amount, etc).
    2. PCA (2 components)              -- reduces the feature space to 2
                                           dimensions so the segments can be
                                           plotted on a scatter chart. PCA
                                           (not Truncated SVD) is used here
                                           because this feature matrix is
                                           dense and already standardized,
                                           unlike sparse TF-IDF text data.
    3. Isolation Forest anomaly detection -- flags applications whose
                                           financial profile looks unusual
                                           relative to the rest of the pool,
                                           a common real-world banking use
                                           case (fraud triage / data-quality
                                           review), entirely unsupervised.

The known Risk labels are used ONLY afterwards, to *evaluate* how well the
unsupervised output lines up with real risk outcomes (segment risk purity,
and the High Risk rate among flagged anomalies vs. the overall population)
-- the algorithms themselves never see the labels while fitting.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
MAX_SCATTER_POINTS = 800


def _assign_persona(avg_cibil, avg_income, avg_dti, dominant_employment, collateral_rate, dominant_risk):
    """Heuristic, human-readable label for a discovered segment, based on
    its centroid characteristics. Purely descriptive -- not used by the
    clustering algorithm itself."""
    if dominant_employment == "Unemployed":
        return "Unemployed Applicants, Strong Credit History" if avg_cibil >= 650 else "Unemployed Applicants, Weak Credit History"
    if avg_cibil < 550:
        return "High-Risk / Thin-File Applicants"
    if collateral_rate >= 80:
        return "Secured Applicants (Collateral-Backed)"
    if collateral_rate <= 10:
        return "Unsecured Applicants (No Collateral)"
    if avg_cibil >= 750 and avg_dti < 30:
        return "Prime Creditworthy Applicants"
    if avg_income >= 85000:
        return "High-Income Established Professionals"
    if avg_dti >= 55:
        return "Debt-Heavy Applicants"
    return "Moderate-Risk Mixed Profile"


def run_segmentation(X_scaled, feature_columns, df_display, n_clusters=5):
    """Runs K-Means customer segmentation and a 2D PCA projection.

    Args:
        X_scaled: 2D numpy array, the full scaled+encoded feature matrix
        feature_columns: list of column names in X_scaled's column order
        df_display: DataFrame (same row order as X_scaled) with
            human-readable raw values (Age, Annual_Income, Cibil_Score,
            Debt_to_Income_Ratio, Employment_Status, Risk, etc.) used only
            for descriptive per-segment statistics -- never fed to K-Means.
        n_clusters: number of segments to discover

    Returns:
        dict, JSON-serializable segmentation report.
    """
    print(f"[clustering] Running K-Means with k={n_clusters} on {X_scaled.shape[0]} applicants...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sample_size = min(1500, X_scaled.shape[0])
    sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=sample_size, random_state=RANDOM_STATE)
    print(f"[clustering] Silhouette Score: {sil_score:.4f}")

    print("[clustering] Projecting to 2D with PCA...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    risk_values = df_display["Risk"].values

    clusters_report = []
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        size = int(mask.sum())
        if size == 0:
            continue

        sub = df_display.loc[mask]
        avg_income = float(sub["Annual_Income"].mean())
        avg_loan = float(sub["Loan_Amount"].mean())
        avg_cibil = float(sub["Cibil_Score"].mean())
        avg_dti = float(sub["Debt_to_Income_Ratio"].mean())
        avg_age = float(sub["Age"].mean())

        risk_counts = sub["Risk"].value_counts()
        dominant_risk = str(risk_counts.idxmax())
        risk_purity = round(float(risk_counts.max()) / size, 3)

        dominant_employment = str(sub["Employment_Status"].mode()[0])
        collateral_rate = round(float((sub["Collateral_Provided"] == "Yes").mean()) * 100, 1)

        persona = _assign_persona(avg_cibil, avg_income, avg_dti, dominant_employment, collateral_rate, dominant_risk)

        clusters_report.append({
            "cluster_id": cluster_id,
            "persona": persona,
            "size": size,
            "avg_annual_income": round(avg_income, 0),
            "avg_loan_amount": round(avg_loan, 0),
            "avg_cibil_score": round(avg_cibil, 0),
            "avg_debt_to_income_ratio": round(avg_dti, 1),
            "avg_age": round(avg_age, 1),
            "dominant_employment_status": dominant_employment,
            "collateral_rate_pct": collateral_rate,
            "dominant_risk": dominant_risk,
            "risk_purity": risk_purity,
        })

    n_points = X_scaled.shape[0]
    if n_points > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(n_points, size=MAX_SCATTER_POINTS, replace=False)
    else:
        sample_idx = np.arange(n_points)

    scatter = []
    names = df_display["Applicant_Name"].values if "Applicant_Name" in df_display.columns else None
    for i in sample_idx:
        scatter.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(cluster_labels[i]),
            "risk": str(risk_values[i]),
            "label": str(names[i]) if names is not None else f"Applicant {i}",
        })

    return {
        "k": n_clusters,
        "n_applicants": int(n_points),
        "silhouette_score": round(float(sil_score), 4),
        "explained_variance_ratio_2d": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "clusters": clusters_report,
        "scatter": scatter,
    }


def run_anomaly_detection(X_scaled, risk_labels, applicant_names, X_full_raw, contamination=0.05):
    """Runs Isolation Forest to flag applicants whose financial profile is
    unusual relative to the rest of the pool -- entirely unsupervised.

    Args:
        X_scaled: 2D numpy array, the full scaled+encoded feature matrix
        risk_labels: array-like of true Risk labels (same row order), used
            only afterwards to evaluate the flagged anomalies
        applicant_names: array-like of applicant names (same row order)
        X_full_raw: DataFrame (same row order) with human-readable raw
            values, used only to describe the flagged anomalies
        contamination: expected proportion of anomalies (Isolation Forest
            hyperparameter)

    Returns:
        dict, JSON-serializable anomaly detection report.
    """
    print(f"[clustering] Running Isolation Forest (contamination={contamination}) "
          f"on {X_scaled.shape[0]} applicants...")
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
    )
    iso.fit(X_scaled)
    predictions = iso.predict(X_scaled)          # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_scaled)      # lower = more anomalous

    is_anomaly = predictions == -1
    n_anomalies = int(is_anomaly.sum())
    n_total = len(X_scaled)
    anomaly_rate = round(n_anomalies / n_total * 100, 2)

    risk_labels = np.asarray(risk_labels)
    overall_high_risk_rate = round(float((risk_labels == "High Risk").mean()) * 100, 2)
    if n_anomalies > 0:
        anomaly_high_risk_rate = round(float((risk_labels[is_anomaly] == "High Risk").mean()) * 100, 2)
    else:
        anomaly_high_risk_rate = None

    print(f"[clustering] Flagged {n_anomalies} anomalies ({anomaly_rate}% of applicants)")
    print(f"[clustering] High Risk rate: overall={overall_high_risk_rate}%, "
          f"among anomalies={anomaly_high_risk_rate}%")

    # ---- 2D projection for visualization (fresh small PCA on this matrix) ----
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    n_points = X_scaled.shape[0]
    if n_points > MAX_SCATTER_POINTS:
        rng = np.random.default_rng(RANDOM_STATE)
        # Always include all anomalies in the sample, fill the rest randomly
        anomaly_idx = np.where(is_anomaly)[0]
        normal_idx = np.where(~is_anomaly)[0]
        remaining = max(MAX_SCATTER_POINTS - len(anomaly_idx), 0)
        sampled_normal = rng.choice(normal_idx, size=min(remaining, len(normal_idx)), replace=False)
        sample_idx = np.concatenate([anomaly_idx, sampled_normal])
    else:
        sample_idx = np.arange(n_points)

    scatter = []
    for i in sample_idx:
        scatter.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "is_anomaly": bool(is_anomaly[i]),
            "risk": str(risk_labels[i]),
            "label": str(applicant_names[i]),
        })

    # ---- Top most-anomalous applicants, described in human-readable terms ----
    order = np.argsort(scores)  # ascending: most negative (most anomalous) first
    top_indices = order[: min(25, n_anomalies) if n_anomalies > 0 else 0]

    top_anomalies = []
    for idx in top_indices:
        row = X_full_raw.iloc[idx]
        top_anomalies.append({
            "applicant_name": str(applicant_names[idx]),
            "anomaly_score": round(float(scores[idx]), 4),
            "risk": str(risk_labels[idx]),
            "age": int(row["Age"]),
            "annual_income": float(row["Annual_Income"]),
            "loan_amount": float(row["Loan_Amount"]),
            "cibil_score": float(row["Cibil_Score"]),
            "debt_to_income_ratio": float(row["Debt_to_Income_Ratio"]),
            "employment_status": str(row["Employment_Status"]),
        })

    return {
        "n_applicants": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_rate_pct": anomaly_rate,
        "contamination": contamination,
        "overall_high_risk_rate_pct": overall_high_risk_rate,
        "anomaly_high_risk_rate_pct": anomaly_high_risk_rate,
        "top_anomalies": top_anomalies,
        "scatter": scatter,
    }
