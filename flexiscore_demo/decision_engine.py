"""
decision_engine.py — FlexiScore Pipeline, 4 quyết định đầu ra.

  AUTO_REJECT  — F0 hard rule fail (gian lận / dữ liệu thiếu / thu nhập quá biến động)
  CREDIT_COACH — F0 pass nhưng F1 fail hoặc data_confidence < 60%
  AUTO_APPROVE — Tất cả pass, điểm đủ cao, PD thấp, EP dương
  HUMAN_REVIEW — Vùng xám: pass gates nhưng chưa đủ tiêu chí tự động duyệt
"""

from scoring_model   import compute_flexiscore, predict_pd, pd_risk_band
from risk_gates      import run_risk_gates
from graph_risk      import compute_graph_risk
from expected_profit import compute_offers

# ── Ngưỡng quyết định ────────────────────────────────────────────────────────
# Trong production, các giá trị này được calibrate trên dữ liệu thật + backtest.
SCORE_AUTO_APPROVE = 700    # FlexiScore tối thiểu để AUTO_APPROVE
PD_AUTO_APPROVE    = 0.10   # PD tối đa để AUTO_APPROVE
DATA_CONF_APPROVE  = 0.70   # data_confidence tối thiểu để AUTO_APPROVE
DATA_CONF_MIN      = 0.60   # Dưới ngưỡng này → CREDIT_COACH


def run_flexiscore_pipeline(customer: dict, model) -> dict:
    """
    Pipeline duy nhất — luôn chạy bất kể use-case nào.
    Mọi quyết định tính từ giá trị tham số, không hard-code theo tên.
    """
    # Stage 1 — Risk Gates (F0 + F1)
    gates = run_risk_gates(customer)

    # Stage 2 — FlexiScore (rule-based, 5 nhóm tiêu chí)
    score_result = compute_flexiscore(customer)
    flexiscore   = score_result["flexiscore"]

    # Stage 3 — PD từ ML model
    pd_score = predict_pd(model, customer)

    # Stage 4 — Graph Risk
    graph = compute_graph_risk(customer)

    # Stage 5 — Expected Profit (3 offers)
    ep_result = compute_offers(customer, pd_score)

    # Stage 6 — Decision
    data_conf = float(customer.get("data_confidence", 1.0))
    decision, reason_codes, coach_plan = _decide(
        f0_fail   = gates["f0_status"] == "FAIL",
        f1_fail = gates["f1_status"] in ("FAIL", "WARN"),
        flexiscore = flexiscore,
        pd_score  = pd_score,
        data_conf = data_conf,
        has_offer = ep_result["has_viable_offer"],
        rec       = ep_result["recommended_offer"],
        customer  = customer,
        gates     = gates,
    )

    return {
        # Identity
        "customer_id":  customer.get("customer_id", "DEMO"),
        "name":         customer.get("name", "Khách hàng"),
        "customer_type":customer.get("customer_type", ""),
        # Scores
        "flexiscore":   flexiscore,
        "risk_tier":    score_result["risk_tier"],
        "pd":           pd_score,
        "pd_band":      pd_risk_band(pd_score),
        # Sub-scores
        "income_stability_score":     score_result["income_stability_score"],
        "graph_safety_score":         score_result["graph_safety_score"],
        "digital_transaction_score":  score_result["digital_transaction_score"],
        "financial_commitment_score": score_result["financial_commitment_score"],
        "platform_behavior_score":    score_result["platform_behavior_score"],
        # Gates
        "f0_status":        gates["f0_status"],
        "f0_reasons":       gates["f0_reasons"],
        "f1_status":        gates["f1_status"],
        "f1_reasons":       gates["f1_reasons"],
        "dti_stress_ratio": gates.get("dti_stress_ratio", 0),
        # Graph
        "graph_risk_score": graph["graph_risk_score"],
        "trust_score":      graph["trust_score"],
        "fraud_ring_flag":  graph["fraud_ring_flag"],
        "graph_risk_label": graph["graph_risk_label"],
        "graph_reasons":    graph["graph_reasons"],
        # EP
        "offers":            ep_result["offers"],
        "recommended_offer": ep_result["recommended_offer"],
        "has_viable_offer":  ep_result["has_viable_offer"],
        # Decision
        "decision":          decision,
        "reason_codes":      reason_codes,
        "credit_coach_plan": coach_plan,
    }


# Alias backward-compat
make_decision = run_flexiscore_pipeline


# ── Decision tree (4 nhánh, theo thứ tự ưu tiên) ────────────────────────────

def _decide(f0_fail, f1_fail, flexiscore, pd_score, data_conf,
            has_offer, rec, customer, gates):

    # Nhánh 1 — AUTO_REJECT: bất kỳ vi phạm F0 nào
    if f0_fail:
        reasons = ["Hồ sơ vi phạm quy tắc cứng (F0) — không thể xử lý tự động."]
        reasons += [f"• {r}" for r in gates["f0_reasons"]]
        return "AUTO_REJECT", reasons, []

    # Nhánh 2 — CREDIT_COACH: F1 fail hoặc dữ liệu không đủ tin cậy
    if f1_fail or data_conf < DATA_CONF_MIN:
        reasons = []
        if f1_fail:
            reasons.append("F1 Stress Test không đạt:")
            reasons += [f"• {r}" for r in gates["f1_reasons"]]
        if data_conf < DATA_CONF_MIN:
            reasons.append(
                f"• Data confidence {data_conf:.0%} < {DATA_CONF_MIN:.0%} "
                f"— cần kết nối thêm nguồn dữ liệu."
            )
        plan = _build_coach_plan(customer, gates)
        return "CREDIT_COACH", reasons, plan

    # Nhánh 3 — AUTO_APPROVE: tất cả điều kiện đạt
    if (flexiscore >= SCORE_AUTO_APPROVE
            and pd_score < PD_AUTO_APPROVE
            and data_conf >= DATA_CONF_APPROVE
            and has_offer):
        reasons = [
            f"FlexiScore {flexiscore:.0f} ≥ {SCORE_AUTO_APPROVE} — hồ sơ tốt.",
            f"PD {pd_score:.1%} < {PD_AUTO_APPROVE:.0%} — rủi ro vỡ nợ thấp.",
            f"Data confidence {data_conf:.0%} ≥ {DATA_CONF_APPROVE:.0%} — dữ liệu tin cậy.",
            "Expected Profit dương — khoản vay khả thi về mặt tài chính.",
        ]
        if rec:
            reasons.append(
                f"Offer được chọn: {rec.label} — "
                f"{rec.amount/1e6:.1f}M / {rec.tenor_months}T / {rec.schedule}."
            )
        return "AUTO_APPROVE", reasons, []

    # Nhánh 4 — HUMAN_REVIEW: vùng xám
    reasons = ["Hồ sơ cần thẩm định thêm từ cán bộ tín dụng:"]
    if flexiscore < SCORE_AUTO_APPROVE:
        reasons.append(
            f"• FlexiScore {flexiscore:.0f} chưa đạt ngưỡng tự động ({SCORE_AUTO_APPROVE})"
        )
    if pd_score >= PD_AUTO_APPROVE:
        reasons.append(
            f"• PD {pd_score:.1%} — rủi ro vỡ nợ cao hơn ngưỡng ({PD_AUTO_APPROVE:.0%})"
        )
    if data_conf < DATA_CONF_APPROVE:
        reasons.append(
            f"• Data confidence {data_conf:.0%} — cần xác thực thêm (cần ≥ {DATA_CONF_APPROVE:.0%})"
        )
    if not has_offer:
        reasons.append("• Không tìm được offer nào có Expected Profit dương")
    return "HUMAN_REVIEW", reasons, []


# ── Lộ trình cải thiện 90 ngày (cá nhân hoá) ────────────────────────────────

def _build_coach_plan(customer: dict, gates: dict) -> list:
    plan     = []
    step     = 1
    income_cv = float(customer.get("income_cv", 0.5))
    act_days  = float(customer.get("active_days_per_week", 4))
    bill      = float(customer.get("bill_on_time_ratio", 0.7))
    data_conf = float(customer.get("data_confidence", 0.5))
    cf_drop   = float(customer.get("cashflow_drop_30d", 0))

    if income_cv > 0.45:
        plan.append(
            f"Bước {step}: Ổn định thu nhập — Giảm biến động về income_cv < 0.45 "
            f"trong 2 tháng liên tiếp (hiện tại {income_cv:.2f})."
        )
        step += 1

    if act_days < 5:
        plan.append(
            f"Bước {step}: Tăng ngày hoạt động — Duy trì ít nhất 5 ngày/tuần "
            f"tạo thu nhập (hiện tại {act_days:.1f} ngày/tuần)."
        )
        step += 1

    if bill < 0.90:
        plan.append(
            f"Bước {step}: Kỷ luật thanh toán — Thanh toán 100% hóa đơn đúng hạn "
            f"trong 2 kỳ tới (tỷ lệ hiện tại {bill:.0%})."
        )
        step += 1

    if cf_drop > 0.25:
        plan.append(
            f"Bước {step}: Phục hồi dòng tiền — Tránh để dòng tiền giảm quá 25% "
            f"(mức giảm gần nhất {cf_drop:.0%})."
        )
        step += 1

    if data_conf < 0.70:
        plan.append(
            f"Bước {step}: Bổ sung dữ liệu — Kết nối EVN/VNPT, ví điện tử hoặc BHXH "
            f"để tăng data confidence lên ≥ 70% (hiện tại {data_conf:.0%})."
        )
        step += 1

    plan.append(
        f"Bước {step}: Tái nộp hồ sơ sau 90 ngày — "
        f"Hạn mức dự kiến 5–10 triệu đồng, kỳ hạn 6 tháng."
    )
    plan.append("Tip: Nộp sao kê BHXH hoặc eTax có thể tăng điểm trong vòng 30 ngày.")

    return plan
