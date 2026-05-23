"""
data_generator.py — Sinh synthetic dataset 3.000 khách hàng cho FlexiScore.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 3000

CUSTOMER_TYPES = ["gig_worker", "seller_online", "freelancer", "small_merchant", "thin_file_customer"]
TYPE_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# Thu nhập trung bình theo loại khách hàng (VND)
INCOME_BY_TYPE = {
    "gig_worker":        (12_000_000, 5_000_000),
    "seller_online":     (20_000_000, 10_000_000),
    "freelancer":        (15_000_000, 8_000_000),
    "small_merchant":    (18_000_000, 9_000_000),
    "thin_file_customer":(8_000_000,  4_000_000),
}


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_dataset(n=N_SAMPLES, seed=RANDOM_SEED, output_path=None):
    rng = np.random.default_rng(seed)

    ctype = rng.choice(CUSTOMER_TYPES, size=n, p=TYPE_WEIGHTS)

    # ── Customer profile ──────────────────────────────────────────────────────
    age = rng.integers(20, 59, size=n).astype(float)
    platform_tenure_months = _clip(rng.exponential(18, n), 1, 60).astype(int)
    data_months_available  = _clip(platform_tenure_months + rng.integers(-2, 4, n), 1, 60)
    identity_verified = (rng.random(n) > 0.1).astype(int)

    # ── Income / cashflow ─────────────────────────────────────────────────────
    avg_monthly_income = np.array([
        max(1_000_000, rng.normal(*INCOME_BY_TYPE[t])) for t in ctype
    ])

    # income_cv: gig_worker thường ổn hơn freelancer/thin_file
    base_cv = np.where(ctype == "gig_worker",        0.15,
              np.where(ctype == "seller_online",      0.25,
              np.where(ctype == "freelancer",         0.45,
              np.where(ctype == "small_merchant",     0.30,
                                                      0.60))))
    income_cv = _clip(base_cv + rng.normal(0, 0.10, n), 0.05, 1.0)

    cashflow_drop_30d = _clip(rng.beta(1.5, 8, n), 0, 0.9)
    active_days_per_week = _clip(
        np.where(ctype == "gig_worker", rng.normal(5.5, 0.8, n),
        np.where(ctype == "freelancer", rng.normal(3.5, 1.0, n),
                                        rng.normal(4.5, 1.0, n))),
        0, 7
    )
    monthly_inflow_count = _clip(rng.poisson(25, n), 1, 150).astype(int)
    expense_to_income_ratio = _clip(rng.beta(4, 3, n) * 0.9, 0.2, 0.95)
    monthly_surplus_ratio   = _clip(1 - expense_to_income_ratio + rng.normal(0, 0.05, n), 0, 0.8)
    saving_buffer = _clip(avg_monthly_income * rng.beta(1.5, 3, n) * 3, 0, avg_monthly_income * 12)

    # ── Payment discipline ────────────────────────────────────────────────────
    bill_on_time_ratio = _clip(rng.beta(7, 1.5, n), 0.3, 1.0)
    utility_payment_delay = _clip(rng.poisson(0.8, n), 0, 30).astype(int)
    late_payment_count_6m  = _clip(rng.poisson(0.6, n), 0, 12).astype(int)
    avg_payment_delay_days = _clip(utility_payment_delay * rng.uniform(0.8, 1.5, n), 0, 30)
    wallet_activity_consistency = _clip(rng.beta(6, 2, n), 0.2, 1.0)

    # ── Business / platform behavior ──────────────────────────────────────────
    seller_revenue_growth = np.where(
        ctype == "seller_online",
        _clip(rng.normal(0.10, 0.15, n), -0.5, 0.8),
        _clip(rng.normal(0.00, 0.10, n), -0.5, 0.5)
    )
    refund_rate = _clip(
        np.where(ctype == "seller_online", rng.beta(2, 10, n), rng.beta(1, 15, n)),
        0, 0.5
    )
    order_frequency = _clip(rng.normal(4, 1.5, n), 0.5, 12)
    repeat_customer_ratio = _clip(rng.beta(3, 2, n), 0, 1)
    rating_avg = _clip(rng.normal(4.5, 0.3, n), 1, 5)
    cancel_rate = _clip(rng.beta(1.5, 8, n), 0, 0.5)
    completion_rate = _clip(1 - cancel_rate + rng.normal(0, 0.02, n), 0.5, 1.0)

    # ── Graph / fraud ─────────────────────────────────────────────────────────
    shared_device_count = _clip(rng.poisson(0.5, n), 0, 10).astype(int)
    shared_ip_count     = _clip(rng.poisson(0.8, n), 0, 15).astype(int)
    bad_neighbor_count  = _clip(rng.poisson(0.3, n), 0, 8).astype(int)

    # ~5% khách có fraud_ring_flag
    fraud_ring_flag = (rng.random(n) < 0.05).astype(int)
    # Nếu fraud, đẩy shared_device và bad_neighbor lên
    shared_device_count = np.where(fraud_ring_flag, np.maximum(shared_device_count, rng.integers(3, 8, n)), shared_device_count)
    bad_neighbor_count  = np.where(fraud_ring_flag, np.maximum(bad_neighbor_count,  rng.integers(2, 6, n)), bad_neighbor_count)

    circular_transaction_ratio = _clip(
        np.where(fraud_ring_flag, rng.beta(5, 3, n), rng.beta(1, 10, n)),
        0, 0.9
    )
    trust_score = _clip(
        np.where(fraud_ring_flag, rng.beta(2, 6, n), rng.beta(6, 2, n)),
        0, 1
    )
    graph_risk_score = _clip(
        0.3 * (shared_device_count / 10)
        + 0.3 * (bad_neighbor_count / 8)
        + 0.2 * fraud_ring_flag
        + 0.2 * circular_transaction_ratio
        + rng.normal(0, 0.05, n),
        0, 1
    )

    # ── Data confidence ───────────────────────────────────────────────────────
    data_confidence = _clip(
        np.where(ctype == "thin_file_customer", rng.beta(2, 4, n), rng.beta(6, 2, n))
        + rng.normal(0, 0.05, n),
        0.1, 1.0
    )
    missing_data_ratio     = _clip(1 - data_confidence + rng.normal(0, 0.05, n), 0, 0.9)
    source_reliability_score = _clip(data_confidence * rng.beta(8, 2, n), 0.1, 1.0)

    # ── Loan request ──────────────────────────────────────────────────────────
    requested_amount = _clip(
        avg_monthly_income * rng.uniform(0.5, 3, n) / 1_000_000,
        2, 100
    ) * 1_000_000
    requested_tenor_months = rng.choice([3, 6, 9, 12, 18, 24], size=n).astype(int)

    # ── Hidden risk score & default label ────────────────────────────────────
    hidden_risk = (
        0.25 * income_cv
        + 0.20 * expense_to_income_ratio
        + 0.15 * (1 - bill_on_time_ratio)
        + 0.15 * graph_risk_score
        + 0.10 * cashflow_drop_30d
        + 0.10 * missing_data_ratio
        + 0.05 * refund_rate
    )
    hidden_risk = _clip(hidden_risk + rng.normal(0, 0.05, n), 0, 1)

    # Sigmoid → xác suất default
    pd_sim = 1 / (1 + np.exp(-8 * (hidden_risk - 0.45)))
    default_label = (rng.random(n) < pd_sim).astype(int)

    df = pd.DataFrame({
        # profile
        "customer_id":              [f"CUST{i:05d}" for i in range(n)],
        "customer_type":            ctype,
        "age":                      age.astype(int),
        "platform_tenure_months":   platform_tenure_months,
        "data_months_available":    data_months_available,
        "identity_verified":        identity_verified,
        # income
        "avg_monthly_income":       avg_monthly_income.round(0),
        "income_cv":                income_cv.round(4),
        "cashflow_drop_30d":        cashflow_drop_30d.round(4),
        "active_days_per_week":     active_days_per_week.round(2),
        "monthly_inflow_count":     monthly_inflow_count,
        "expense_to_income_ratio":  expense_to_income_ratio.round(4),
        "monthly_surplus_ratio":    monthly_surplus_ratio.round(4),
        "saving_buffer":            saving_buffer.round(0),
        # payment
        "bill_on_time_ratio":       bill_on_time_ratio.round(4),
        "utility_payment_delay":    utility_payment_delay,
        "late_payment_count_6m":    late_payment_count_6m,
        "avg_payment_delay_days":   avg_payment_delay_days.round(2),
        "wallet_activity_consistency": wallet_activity_consistency.round(4),
        # business
        "seller_revenue_growth":    seller_revenue_growth.round(4),
        "refund_rate":              refund_rate.round(4),
        "order_frequency":          order_frequency.round(2),
        "repeat_customer_ratio":    repeat_customer_ratio.round(4),
        "rating_avg":               rating_avg.round(2),
        "cancel_rate":              cancel_rate.round(4),
        "completion_rate":          completion_rate.round(4),
        # graph
        "shared_device_count":      shared_device_count,
        "shared_ip_count":          shared_ip_count,
        "bad_neighbor_count":       bad_neighbor_count,
        "fraud_ring_flag":          fraud_ring_flag,
        "circular_transaction_ratio": circular_transaction_ratio.round(4),
        "trust_score":              trust_score.round(4),
        "graph_risk_score":         graph_risk_score.round(4),
        # data quality
        "data_confidence":          data_confidence.round(4),
        "missing_data_ratio":       missing_data_ratio.round(4),
        "source_reliability_score": source_reliability_score.round(4),
        # loan
        "requested_amount":         requested_amount.round(0),
        "requested_tenor_months":   requested_tenor_months,
        # target
        "default_label":            default_label,
    })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}  ({len(df):,} rows, default rate {default_label.mean():.1%})")

    return df


if __name__ == "__main__":
    df = generate_dataset(output_path="data/training_data.csv")
    print(df.head())
    print(df["default_label"].value_counts(normalize=True))
