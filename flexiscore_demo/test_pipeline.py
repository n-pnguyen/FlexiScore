"""Test end-to-end pipeline for all 3 demo cases."""
from data_generator import generate_dataset
from scoring_model import train
from decision_engine import make_decision

print("Generating dataset...")
df = generate_dataset()
print(f"Dataset: {len(df)} rows, default rate: {df['default_label'].mean():.1%}")

print("\nTraining model...")
model, metrics = train(df, save=True)
print("Metrics:", metrics)

# Happy Path
happy = {
    "customer_id": "DEMO_001", "name": "Nguyen Van A", "customer_type": "gig_worker",
    "avg_monthly_income": 15_000_000, "income_cv": 0.15, "active_days_per_week": 6.0,
    "cashflow_drop_30d": 0.0, "bill_on_time_ratio": 0.98, "utility_payment_delay": 0,
    "late_payment_count_6m": 0, "avg_payment_delay_days": 0, "wallet_activity_consistency": 0.92,
    "expense_to_income_ratio": 0.55, "monthly_surplus_ratio": 0.40, "saving_buffer": 8_000_000,
    "seller_revenue_growth": 0, "refund_rate": 0, "order_frequency": 6, "repeat_customer_ratio": 0.75,
    "rating_avg": 4.8, "cancel_rate": 0.03, "completion_rate": 0.97, "platform_tenure_months": 18,
    "shared_device_count": 0, "shared_ip_count": 0, "bad_neighbor_count": 0, "fraud_ring_flag": 0,
    "circular_transaction_ratio": 0.02, "trust_score": 0.92, "graph_risk_score": 0.08,
    "data_confidence": 0.88, "missing_data_ratio": 0.08, "source_reliability_score": 0.90,
    "requested_amount": 20_000_000, "requested_tenor_months": 12, "monthly_inflow_count": 48,
}
r1 = make_decision(happy, model)
print(f"\nHappy Path => FlexiScore={r1['flexiscore']} F0={r1['f0_status']} F1={r1['f1_status']} Decision={r1['decision']}")
rec = r1["recommended_offer"]
if rec:
    print(f"  Offer: {rec.amount/1e6:.1f}M / {rec.tenor_months}T / {rec.schedule} / EP={rec.expected_profit:,.0f}")

# Risk-First
risk = {
    "customer_id": "DEMO_002", "name": "Tran Thi B", "customer_type": "seller_online",
    "avg_monthly_income": 25_000_000, "income_cv": 0.20, "active_days_per_week": 6.0,
    "cashflow_drop_30d": 0.05, "bill_on_time_ratio": 0.96, "utility_payment_delay": 0,
    "late_payment_count_6m": 0, "avg_payment_delay_days": 0, "wallet_activity_consistency": 0.88,
    "expense_to_income_ratio": 0.55, "monthly_surplus_ratio": 0.40, "saving_buffer": 15_000_000,
    "seller_revenue_growth": 0.18, "refund_rate": 0.08, "order_frequency": 8, "repeat_customer_ratio": 0.65,
    "rating_avg": 4.7, "cancel_rate": 0.05, "completion_rate": 0.96, "platform_tenure_months": 24,
    "shared_device_count": 4, "shared_ip_count": 6, "bad_neighbor_count": 3, "fraud_ring_flag": 1,
    "circular_transaction_ratio": 0.28, "trust_score": 0.28, "graph_risk_score": 0.91,
    "data_confidence": 0.80, "missing_data_ratio": 0.20, "source_reliability_score": 0.75,
    "requested_amount": 40_000_000, "requested_tenor_months": 24, "monthly_inflow_count": 120,
}
r2 = make_decision(risk, model)
print(f"\nRisk-First => F0={r2['f0_status']} Decision={r2['decision']}")

# Credit Coach
coach = {
    "customer_id": "DEMO_003", "name": "Le Van C", "customer_type": "freelancer",
    "avg_monthly_income": 12_000_000, "income_cv": 0.65, "active_days_per_week": 3.5,
    "cashflow_drop_30d": 0.42, "bill_on_time_ratio": 0.70, "utility_payment_delay": 2,
    "late_payment_count_6m": 3, "avg_payment_delay_days": 4.5, "wallet_activity_consistency": 0.55,
    "expense_to_income_ratio": 0.72, "monthly_surplus_ratio": 0.20, "saving_buffer": 2_000_000,
    "seller_revenue_growth": 0, "refund_rate": 0, "order_frequency": 3, "repeat_customer_ratio": 0.40,
    "rating_avg": 4.2, "cancel_rate": 0.15, "completion_rate": 0.82, "platform_tenure_months": 8,
    "shared_device_count": 0, "shared_ip_count": 1, "bad_neighbor_count": 0, "fraud_ring_flag": 0,
    "circular_transaction_ratio": 0.03, "trust_score": 0.75, "graph_risk_score": 0.20,
    "data_confidence": 0.55, "missing_data_ratio": 0.38, "source_reliability_score": 0.60,
    "requested_amount": 15_000_000, "requested_tenor_months": 12, "monthly_inflow_count": 8,
}
r3 = make_decision(coach, model)
print(f"\nCredit Coach => FlexiScore={r3['flexiscore']} F1={r3['f1_status']} Decision={r3['decision']}")
print(f"  Coach plan items: {len(r3['credit_coach_plan'])}")

print("\n--- Acceptance Criteria Check ---")
assert r1["f0_status"] == "PASS", "Happy Path F0 must PASS"
assert "APPROVE" in r1["decision"], f"Happy Path must APPROVE, got {r1['decision']}"
assert r1["recommended_offer"] is not None, "Happy Path needs a recommended offer"
assert r1["recommended_offer"].ep_positive, "Happy Path EP must be positive"

assert r2["f0_status"] == "FAIL", "Risk-First F0 must FAIL"
assert "FRAUD" in r2["decision"], f"Risk-First must be FRAUD decision, got {r2['decision']}"

assert r3["f1_status"] in ("FAIL", "WARN"), f"Credit Coach F1 must FAIL/WARN, got {r3['f1_status']}"
assert r3["decision"] == "CREDIT_COACH", f"Credit Coach decision must be CREDIT_COACH, got {r3['decision']}"
assert len(r3["credit_coach_plan"]) >= 4, "Coach plan must have at least 4 items"

print("ALL ACCEPTANCE CRITERIA PASSED")
