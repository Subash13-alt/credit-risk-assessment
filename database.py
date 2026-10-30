"""
database.py
------------
Lightweight SQLite persistence layer for CreditSense AI.

Logs every prediction (single or batch) so the /dashboard route can show
aggregate analytics (risk distribution, volume over time, average
confidence, etc.). Uses Python's built-in sqlite3 -- no extra database
server or dependency required.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join("model", "predictions.db")


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_connection():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates the predictions table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                applicant_name TEXT,
                risk TEXT NOT NULL,
                probability REAL NOT NULL,
                message TEXT,
                source TEXT DEFAULT 'single',
                annual_income REAL,
                loan_amount REAL,
                credit_history TEXT,
                debt_to_income_ratio REAL,
                cibil_score REAL
            )
        """)


def log_prediction(applicant_name, risk, probability, message, source="single",
                    annual_income=None, loan_amount=None, credit_history=None,
                    debt_to_income_ratio=None, cibil_score=None):
    """Inserts one prediction record. Never raises -- logging failures
    should never break the actual prediction response."""
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (created_at, applicant_name, risk, probability, message, source,
                    annual_income, loan_amount, credit_history, debt_to_income_ratio, cibil_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    applicant_name, risk, probability, message, source,
                    annual_income, loan_amount, credit_history, debt_to_income_ratio, cibil_score,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[database] Failed to log prediction: {exc}")


def get_dashboard_stats(limit=500):
    """Returns aggregate stats for the analytics dashboard."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]

        risk_counts = conn.execute(
            "SELECT risk, COUNT(*) AS c FROM predictions GROUP BY risk"
        ).fetchall()

        avg_conf = conn.execute(
            "SELECT AVG(probability) AS avg_p FROM predictions"
        ).fetchone()["avg_p"]

        avg_cibil = conn.execute(
            "SELECT AVG(cibil_score) AS avg_c FROM predictions WHERE cibil_score IS NOT NULL"
        ).fetchone()["avg_c"]

        recent = conn.execute(
            """SELECT created_at, applicant_name, risk, probability, source
               FROM predictions ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        by_day = conn.execute(
            """SELECT substr(created_at, 1, 10) AS day,
                      SUM(CASE WHEN risk='Low Risk' THEN 1 ELSE 0 END) AS low,
                      SUM(CASE WHEN risk='High Risk' THEN 1 ELSE 0 END) AS high
               FROM predictions GROUP BY day ORDER BY day DESC LIMIT 14"""
        ).fetchall()

        credit_hist = conn.execute(
            """SELECT credit_history, COUNT(*) AS c FROM predictions
               WHERE credit_history IS NOT NULL GROUP BY credit_history"""
        ).fetchall()

    return {
        "total_predictions": total,
        "risk_distribution": [dict(r) for r in risk_counts],
        "average_confidence": round(avg_conf, 2) if avg_conf is not None else None,
        "average_cibil_score": round(avg_cibil, 0) if avg_cibil is not None else None,
        "recent": [dict(r) for r in recent],
        "by_day": [dict(r) for r in reversed(by_day)],
        "credit_history_distribution": [dict(r) for r in credit_hist],
    }
