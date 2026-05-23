"""
scoring_model.py — Train model PD và tính FlexiScore.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

RANDOM_SEED = 42
MODEL_PATH = Path("models/pd_model.pkl")

FEATURE_COLS = [
    "income_cv", "active_days_per_week", "avg_monthly_income", "cashflow_drop_30d",
    "expense_to_income_ratio", "monthly_surplus_ratio", "saving_buffer",
    "bill_on_time_ratio", "utility_payment_delay", "late_payment_count_6m",
    "avg_payment_delay_days", "wallet_activity_consistency",
    "seller_revenue_growth", "refund_rate", "rating_avg", "cancel_rate",
    "completion_rate", "platform_tenure_months",
    "shared_device_count", "shared_ip_count", "bad_neighbor_count",
    "fraud_ring_flag", "circular_transaction_ratio", "trust_score", "graph_risk_score",
    "data_confidence", "missing_data_ratio", "source_reliability_score",
]


# ── FlexiScore formula ────────────────────────────────────────────────────────

def _norm(val, lo, hi, invert=False):
    """Normalize value to 0–100."""
    score = float(np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)) * 100
    return 100 - score if invert else score


def compute_flexiscore(row: dict) -> dict:
    """Return FlexiScore (0–1000) and sub-scores for a single customer dict."""

    # 1. Income Stability (35%)
    s_cv    = _norm(row.get("income_cv", 0.5),           0, 1,   invert=True)
    s_days  = _norm(row.get("active_days_per_week", 4),  0, 7)
    s_inc   = _norm(row.get("avg_monthly_income", 0),    0, 50_000_000)
    s_drop  = _norm(row.get("cashflow_drop_30d", 0),     0, 1,   invert=True)
    income_score = (s_cv * 0.35 + s_days * 0.30 + s_inc * 0.20 + s_drop * 0.15)

    # 2. Graph Safety (25%)
    g_fraud   = 100 if not row.get("fraud_ring_flag", 0) else 0
    g_device  = _norm(row.get("shared_device_count", 0), 0, 8,  invert=True)
    g_neigh   = _norm(row.get("bad_neighbor_count", 0),  0, 6,  invert=True)
    g_trust   = _norm(row.get("trust_score", 0.5),       0, 1)
    g_risk    = _norm(row.get("graph_risk_score", 0.5),  0, 1,  invert=True)
    graph_score = (g_fraud * 0.30 + g_device * 0.20 + g_neigh * 0.20
                   + g_trust * 0.15 + g_risk * 0.15)

    # 3. Digital Transaction History (20%)
    d_bill   = _norm(row.get("bill_on_time_ratio", 0.5),          0, 1)
    d_wallet = _norm(row.get("wallet_activity_consistency", 0.5), 0, 1)
    d_surp   = _norm(row.get("monthly_surplus_ratio", 0.1),       0, 0.8)
    d_save   = _norm(row.get("saving_buffer", 0),                 0, 50_000_000)
    digital_score = (d_bill * 0.40 + d_wallet * 0.25 + d_surp * 0.20 + d_save * 0.15)

    # 4. Financial Commitment (15%)
    f_delay = _norm(row.get("utility_payment_delay", 0),  0, 15,  invert=True)
    f_late  = _norm(row.get("late_payment_count_6m", 0),  0, 6,   invert=True)
    f_bill  = _norm(row.get("bill_on_time_ratio", 0.5),   0, 1)
    f_conf  = _norm(row.get("data_confidence", 0.5),      0, 1)
    financial_score = (f_delay * 0.30 + f_late * 0.30 + f_bill * 0.25 + f_conf * 0.15)

    # 5. Platform Behavior (5%)
    p_rating  = _norm(row.get("rating_avg", 4),           1, 5)
    p_cancel  = _norm(row.get("cancel_rate", 0.1),        0, 0.5, invert=True)
    p_complete= _norm(row.get("completion_rate", 0.9),    0.5, 1)
    p_tenure  = _norm(row.get("platform_tenure_months", 6), 0, 60)
    platform_score = (p_rating * 0.35 + p_cancel * 0.25 + p_complete * 0.25 + p_tenure * 0.15)

    # Weighted composite
    raw = (
        0.35 * income_score
        + 0.25 * graph_score
        + 0.20 * digital_score
        + 0.15 * financial_score
        + 0.05 * platform_score
    )
    flexiscore = round(raw * 10, 1)

    # Risk tier
    if row.get("fraud_ring_flag", 0):
        tier = "Fraud Flag"
    elif flexiscore >= 800:
        tier = "Tier 1 — Very Low Risk"
    elif flexiscore >= 600:
        tier = "Tier 2 — Medium-Low Risk"
    elif flexiscore >= 450:
        tier = "Tier 3 — Borderline"
    else:
        tier = "Tier 4 — High Risk"

    return {
        "flexiscore": flexiscore,
        "risk_tier": tier,
        "income_stability_score": round(income_score, 1),
        "graph_safety_score":     round(graph_score, 1),
        "digital_transaction_score": round(digital_score, 1),
        "financial_commitment_score": round(financial_score, 1),
        "platform_behavior_score": round(platform_score, 1),
    }


# ── Model training ────────────────────────────────────────────────────────────

def _build_model():
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        print("Using LightGBM")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_SEED,
        )
        print("LightGBM not found — using GradientBoostingClassifier")
    return model


def train(df: pd.DataFrame, save=True):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = df[FEATURE_COLS].fillna(0)
    y = df["default_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    model = _build_model()
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
    }
    print("Model metrics:", metrics)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    return model, metrics


def load_or_train(df=None):
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH!s}")
        return model, {}
    if df is None:
        from data_generator import generate_dataset
        df = generate_dataset()
    model, metrics = train(df)
    return model, metrics


def predict_pd(model, customer: dict) -> float:
    """Return Probability of Default for a single customer dict."""
    row = {col: customer.get(col, 0) for col in FEATURE_COLS}
    X = pd.DataFrame([row])
    pd_score = float(model.predict_proba(X)[:, 1][0])
    return round(pd_score, 4)


def pd_risk_band(pd_score: float) -> str:
    if pd_score < 0.05:
        return "LOW"
    elif pd_score < 0.15:
        return "MEDIUM"
    else:
        return "HIGH"


if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset()
    model, metrics = train(df, save=True)
    print(metrics)
