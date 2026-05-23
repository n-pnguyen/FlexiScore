"""
test_pipeline.py — FlexiScore end-to-end tests.

Decision types under test:
  AUTO_APPROVE               — score>=800, PD<10%, conf>=70%, EP>0
  FRAUD_REJECT               — F0 fail + fraud_ring or graph_risk>0.70
  FRAUD_REVIEW               — F0 fail (other hard rule, e.g. income_cv>0.75)
  CREDIT_COACH               — F1 fail (income_cv>0.55 or DTI>35%)
  CREDIT_COACH (low conf)    — data_confidence < 60%
  HUMAN_REVIEW               — borderline score / conf / graph risk
  Demo-case verification     — 4 preset scenarios, decisions purely value-driven
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from data_generator import generate_dataset
from scoring_model import load_or_train
from decision_engine import run_flexiscore_pipeline

print("Loading model...")
df = generate_dataset()
model, _ = load_or_train(df)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _base():
    """Clean base profile — expected outcome: AUTO_APPROVE."""
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
        f"\n  FlexiScore={r['flexiscore']:.0f}  PD={r['pd']:.2%}"
        f"  F0={r['f0_status']}  F1={r['f1_status']}"
        f"  graph_risk={r['graph_risk_score']:.2f}"
        f"  data_conf={customer.get('data_confidence',1):.0%}"
        f"\n  => DECISION: {r['decision']}"
    )
    return r


def _assert(r, expected, label):
    if r["decision"] == expected:
        print(f"  PASS")
    else:
        raise AssertionError(
            f"[{label}] Expected {expected}, got {r['decision']}\n"
            f"  reasons: {r.get('reason_codes', [])[:3]}"
        )


# ── Test 1: AUTO_APPROVE ──────────────────────────────────────────────────────
r = _run("AUTO_APPROVE", _base())
_assert(r, "AUTO_APPROVE", "Test 1")
assert r["recommended_offer"] is not None, "AUTO_APPROVE must have an offer"

# ── Test 2: FRAUD_REJECT (fraud_ring + high graph_risk) ──────────────────────
c = {**_base(),
     "fraud_ring_flag": 1, "shared_device_count": 4,
     "bad_neighbor_count": 3, "trust_score": 0.28, "graph_risk_score": 0.91}
r = _run("FRAUD_REJECT — fraud ring + high graph risk", c)
_assert(r, "FRAUD_REJECT", "Test 2")

# ── Test 3: FRAUD_REVIEW (income_cv F0 — no fraud ring) ──────────────────────
c = {**_base(), "income_cv": 0.80}
r = _run("FRAUD_REVIEW — income_cv=0.80 (>0.75), no fraud ring", c)
_assert(r, "FRAUD_REVIEW", "Test 3")

# ── Test 4: CREDIT_COACH (F1 fail — income_cv>0.55) ─────────────────────────
c = {**_base(),
     "income_cv": 0.65, "cashflow_drop_30d": 0.42,
     "bill_on_time_ratio": 0.70, "active_days_per_week": 3.5,
     "data_confidence": 0.65}
r = _run("CREDIT_COACH — F1 fail (income_cv=0.65)", c)
_assert(r, "CREDIT_COACH", "Test 4")
assert len(r["credit_coach_plan"]) >= 3, "Coach plan must have >= 3 steps"

# ── Test 5: CREDIT_COACH (low data_confidence 0.40–0.60) ─────────────────────
c = {**_base(), "data_confidence": 0.45, "missing_data_ratio": 0.55}
r = _run("CREDIT_COACH — low data_confidence=0.45", c)
_assert(r, "CREDIT_COACH", "Test 5")

# ── Test 6: HUMAN_REVIEW (borderline profile) ─────────────────────────────────
c = {**_base(),
     "income_cv": 0.38, "bill_on_time_ratio": 0.82,
     "data_confidence": 0.65, "shared_device_count": 1, "bad_neighbor_count": 1,
     "trust_score": 0.65, "graph_risk_score": 0.32, "rating_avg": 4.3,
     "cancel_rate": 0.10, "avg_monthly_income": 10_000_000}
r = _run("HUMAN_REVIEW — borderline", c)
_assert(r, "HUMAN_REVIEW", "Test 6")

# ── Test 7: Demo cases — 4 preset scenarios ───────────────────────────────────
from app import DEMO_CASES

# Expected decisions for each demo case.
# All decisions emerge purely from feature values — none are hard-coded by name.
case_expected = {
    "Happy Path — Nguyễn Văn A": "AUTO_APPROVE",
    "Risk-First — Trần Thị B":   "FRAUD_REJECT",   # fraud_ring=1, graph_risk=0.91
    "Credit Coach — Lê Văn C":   "CREDIT_COACH",   # income_cv=0.65 → F1 fail
    "Vùng xám — Nguyễn Thị D":  "HUMAN_REVIEW",   # borderline score/conf
}

print("\n── Demo case verification ─────────────────────────────────────────────")
all_pass = True
for name, expected in case_expected.items():
    c = {**DEMO_CASES[name]}
    r = run_flexiscore_pipeline(c, model)
    ok = r["decision"] == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         FlexiScore={r['flexiscore']:.0f}  PD={r['pd']:.2%}"
          f"  F0={r['f0_status']}  F1={r['f1_status']}  => {r['decision']}"
          f"  (expected {expected})")
    if not ok:
        all_pass = False

if not all_pass:
    raise AssertionError("One or more demo cases failed — see output above.")

print("\n=== ALL TESTS PASSED ===")
