"""
test_pipeline.py — FlexiScore end-to-end tests (4 quyết định).

  AUTO_REJECT  — F0 fail (bất kỳ vi phạm)
  CREDIT_COACH — F0 pass nhưng F1 fail hoặc data_confidence < 60%
  AUTO_APPROVE — FlexiScore≥700, PD<10%, conf≥70%, EP>0
  HUMAN_REVIEW — Vùng xám (pass gates nhưng chưa đủ auto-approve)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_generator import generate_dataset
from scoring_model import load_or_train
from decision_engine import run_flexiscore_pipeline

print("Loading model...")
df = generate_dataset()
model, _ = load_or_train(df)


def _base():
    """Hồ sơ cơ bản tốt — baseline cho các test."""
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


def _run(label, customer):
    r = run_flexiscore_pipeline(customer, model)
    print(
        f"\n[{label}]"
        f"  Score={r['flexiscore']:.0f}  PD={r['pd']:.1%}"
        f"  F0={r['f0_status']}  F1={r['f1_status']}"
        f"  conf={customer.get('data_confidence',1):.0%}"
        f"  => {r['decision']}"
    )
    return r


def _ok(r, expected, label):
    if r["decision"] == expected:
        print("  PASS")
    else:
        raise AssertionError(f"[{label}] Expected {expected}, got {r['decision']}")


# ── Test 1: AUTO_APPROVE ──────────────────────────────────────────────────────
r = _run("AUTO_APPROVE — base sạch", _base())
_ok(r, "AUTO_APPROVE", "T1")
assert r["recommended_offer"] is not None

# ── Test 2: AUTO_REJECT — fraud ring ─────────────────────────────────────────
r = _run("AUTO_REJECT — fraud_ring=1", {
    **_base(),
    "fraud_ring_flag": 1, "shared_device_count": 4, "bad_neighbor_count": 3,
})
_ok(r, "AUTO_REJECT", "T2")

# ── Test 3: AUTO_REJECT — income_cv F0 ───────────────────────────────────────
r = _run("AUTO_REJECT — income_cv=0.80 (>0.75)", {**_base(), "income_cv": 0.80})
_ok(r, "AUTO_REJECT", "T3")

# ── Test 4: CREDIT_COACH — F1 fail (income_cv) ───────────────────────────────
r = _run("CREDIT_COACH — income_cv=0.65 → F1 fail", {
    **_base(),
    "income_cv": 0.65, "cashflow_drop_30d": 0.42,
    "bill_on_time_ratio": 0.70, "active_days_per_week": 3.5,
    "data_confidence": 0.65,
})
_ok(r, "CREDIT_COACH", "T4")
assert len(r["credit_coach_plan"]) >= 3

# ── Test 5: CREDIT_COACH — low data_confidence ───────────────────────────────
r = _run("CREDIT_COACH — data_confidence=0.45 (<60%)", {
    **_base(), "data_confidence": 0.45, "missing_data_ratio": 0.55,
})
_ok(r, "CREDIT_COACH", "T5")

# ── Test 6: HUMAN_REVIEW — borderline ────────────────────────────────────────
r = _run("HUMAN_REVIEW — borderline", {
    **_base(),
    "income_cv": 0.38, "bill_on_time_ratio": 0.82,
    "data_confidence": 0.65,  # >= 0.60 (not coach) but < 0.70 (not auto-approve)
    "avg_monthly_income": 10_000_000,
    "shared_device_count": 1, "bad_neighbor_count": 1,
})
_ok(r, "HUMAN_REVIEW", "T6")

# ── Test 7: Demo cases ────────────────────────────────────────────────────────
from app import DEMO_CASES

EXPECTED = {
    "Happy Path — Nguyễn Văn A": "AUTO_APPROVE",
    "Risk-First — Trần Thị B":   "AUTO_REJECT",    # fraud_ring=1 → F0 fail
    "Credit Coach — Lê Văn C":   "CREDIT_COACH",   # income_cv=0.65 → F1 fail
    "Human Review — Nguyễn Thị D":  "HUMAN_REVIEW",   # borderline score/conf
}

print("\n── Demo cases ────────────────────────────────────────────────────────")
for name, expected in EXPECTED.items():
    r = run_flexiscore_pipeline({**DEMO_CASES[name]}, model)
    ok = r["decision"] == expected
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}"
        f"  Score={r['flexiscore']:.0f}  PD={r['pd']:.1%}"
        f"  F0={r['f0_status']}  F1={r['f1_status']}"
        f"  => {r['decision']}  (expected {expected})"
    )
    assert ok, f"{name}: got {r['decision']}, expected {expected}"

print("\n=== ALL TESTS PASSED ===")
