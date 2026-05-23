"""Test end-to-end pipeline — 4 decisions."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_generator import generate_dataset
from scoring_model import load_or_train
from decision_engine import make_decision

print("Loading model...")
df = generate_dataset()
model, _ = load_or_train(df)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _base():
    """Hồ sơ cơ bản tốt — dùng làm nền để mutate."""
    return {
        "customer_id": "TEST", "name": "Test", "customer_type": "gig_worker",
        "avg_monthly_income": 15_000_000, "income_cv": 0.15,
        "active_days_per_week": 6.0, "cashflow_drop_30d": 0.0,
        "bill_on_time_ratio": 0.98, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0,
        "wallet_activity_consistency": 0.92, "expense_to_income_ratio": 0.50,
        "monthly_surplus_ratio": 0.45, "saving_buffer": 8_000_000,
        "seller_revenue_growth": 0, "refund_rate": 0, "order_frequency": 6,
        "repeat_customer_ratio": 0.75, "rating_avg": 4.8, "cancel_rate": 0.03,
        "completion_rate": 0.97, "platform_tenure_months": 18,
        "shared_device_count": 0, "shared_ip_count": 0, "bad_neighbor_count": 0,
        "fraud_ring_flag": 0, "circular_transaction_ratio": 0.02,
        "trust_score": 0.92, "graph_risk_score": 0.08,
        "data_confidence": 0.88, "missing_data_ratio": 0.08,
        "source_reliability_score": 0.90,
        "requested_amount": 20_000_000, "requested_tenor_months": 12,
        "monthly_inflow_count": 48,
    }


# ── Test 1: AUTO_APPROVE ──────────────────────────────────────────────────────
c = _base()  # clean profile
r = make_decision(c, model)
print(f"\nAUTO_APPROVE test => FlexiScore={r['flexiscore']} F0={r['f0_status']} F1={r['f1_status']} => {r['decision']}")
assert r["decision"] == "AUTO_APPROVE", f"Expected AUTO_APPROVE, got {r['decision']}"
assert r["recommended_offer"] is not None
print("  PASS")

# ── Test 2: AUTO_REJECT (fraud_ring) ─────────────────────────────────────────
c = {**_base(), "fraud_ring_flag": 1, "shared_device_count": 4,
     "bad_neighbor_count": 3, "trust_score": 0.28, "graph_risk_score": 0.91}
r = make_decision(c, model)
print(f"\nAUTO_REJECT test => F0={r['f0_status']} => {r['decision']}")
assert r["decision"] == "AUTO_REJECT", f"Expected AUTO_REJECT, got {r['decision']}"
print("  PASS")

# ── Test 3: AUTO_REJECT (income_cv hard rule) ─────────────────────────────────
c = {**_base(), "income_cv": 0.80}
r = make_decision(c, model)
print(f"\nAUTO_REJECT (income_cv>0.75) => F0={r['f0_status']} => {r['decision']}")
assert r["decision"] == "AUTO_REJECT", f"Expected AUTO_REJECT, got {r['decision']}"
print("  PASS")

# ── Test 4: CREDIT_COACH (F1 fail — income_cv) ───────────────────────────────
c = {**_base(), "income_cv": 0.65, "cashflow_drop_30d": 0.42,
     "bill_on_time_ratio": 0.70, "active_days_per_week": 3.5,
     "data_confidence": 0.65}   # >=0.60 → bypass data-conf coach branch
r = make_decision(c, model)
print(f"\nCREDIT_COACH (F1 fail) => F1={r['f1_status']} => {r['decision']}")
assert r["decision"] == "CREDIT_COACH", f"Expected CREDIT_COACH, got {r['decision']}"
assert len(r["credit_coach_plan"]) >= 3
print("  PASS")

# ── Test 5: CREDIT_COACH (low data_confidence) ───────────────────────────────
c = {**_base(), "data_confidence": 0.45, "missing_data_ratio": 0.55}
r = make_decision(c, model)
print(f"\nCREDIT_COACH (low conf) => data_conf={c['data_confidence']} => {r['decision']}")
assert r["decision"] == "CREDIT_COACH", f"Expected CREDIT_COACH, got {r['decision']}"
print("  PASS")

# ── Test 6: HUMAN_REVIEW (borderline score) ───────────────────────────────────
c = {**_base(), "income_cv": 0.38, "bill_on_time_ratio": 0.82,
     "data_confidence": 0.65, "shared_device_count": 1, "bad_neighbor_count": 1,
     "trust_score": 0.65, "graph_risk_score": 0.32, "rating_avg": 4.3,
     "cancel_rate": 0.10, "avg_monthly_income": 10_000_000}
r = make_decision(c, model)
print(f"\nHUMAN_REVIEW test => FlexiScore={r['flexiscore']:.0f} F0={r['f0_status']} F1={r['f1_status']} => {r['decision']}")
assert r["decision"] == "HUMAN_REVIEW", f"Expected HUMAN_REVIEW, got {r['decision']}"
print("  PASS")

# ── Test 7: Demo cases (just verify no crash + correct bucket) ────────────────
from app import DEMO_CASES
case_expected = {
    "Happy Path — Nguyễn Văn A": "AUTO_APPROVE",
    "Risk-First — Trần Thị B":   "AUTO_REJECT",
    "Credit Coach — Lê Văn C":   "CREDIT_COACH",
    "Vùng xám — Nguyễn Thị D":  "HUMAN_REVIEW",
}
print()
for name, expected in case_expected.items():
    c = {**DEMO_CASES[name]}
    r = make_decision(c, model)
    status = "PASS" if r["decision"] == expected else "FAIL"
    print(f"  [{status}] {name} => {r['decision']}  (expected {expected})")
    assert r["decision"] == expected, f"{name}: got {r['decision']}, expected {expected}"

print("\n=== ALL TESTS PASSED ===")
