"""
decision_engine.py — 4 quyết định tín dụng rõ ràng.

AUTO_REJECT   : F0 fail (fraud / hard rule)
CREDIT_COACH  : F0 pass nhưng F1 fail hoặc data_confidence thấp
AUTO_APPROVE  : F0+F1 pass, điểm cao, tin cậy cao, EP dương
HUMAN_REVIEW  : Vùng xám — không rõ ràng đủ để auto
"""

from scoring_model import compute_flexiscore, predict_pd, pd_risk_band
from risk_gates    import run_risk_gates
from graph_risk    import compute_graph_risk
from expected_profit import compute_offers

# ── Ngưỡng quyết định ────────────────────────────────────────────────────────
APPROVE_FLEXISCORE_MIN   = 700    # điểm tối thiểu để auto approve
APPROVE_PD_MAX           = 0.15   # PD tối đa để auto approve
APPROVE_DATA_CONF_MIN    = 0.70   # data confidence tối thiểu để auto approve
COACH_DATA_CONF_MAX      = 0.60   # data confidence thấp → coach thay vì score
# (F1 fail cũng → coach, xem risk_gates.py)


def make_decision(customer: dict, model) -> dict:
    """
    Chạy toàn bộ pipeline và trả kết quả.

    Decision tree:
        F0 FAIL                         → AUTO_REJECT
        F1 FAIL  |  data_conf < 0.60   → CREDIT_COACH
        score>=700, PD<0.15, EP>0, conf>=0.70 → AUTO_APPROVE
        else                            → HUMAN_REVIEW
    """

    # 1. Risk Gates
    gates = run_risk_gates(customer)

    # 2. FlexiScore
    score_result = compute_flexiscore(customer)
    flexiscore   = score_result["flexiscore"]
    risk_tier    = score_result["risk_tier"]

    # 3. PD
    pd_score = predict_pd(model, customer)
    pd_band  = pd_risk_band(pd_score)

    # 4. Graph Risk
    graph = compute_graph_risk(customer)

    # 5. Expected Profit
    ep_result = compute_offers(customer, pd_score)

    # 6. Extract gate flags
    f0_fail   = (gates["f0_status"] == "FAIL")
    f1_fail   = (gates["f1_status"] == "FAIL")
    data_conf = float(customer.get("data_confidence", 1.0))

    # ── Decision tree ─────────────────────────────────────────────────────────
    if f0_fail:
        decision, reason_codes, credit_coach_plan = _auto_reject(customer, gates, graph)

    elif f1_fail or data_conf < COACH_DATA_CONF_MAX:
        decision, reason_codes, credit_coach_plan = _credit_coach(customer, gates, score_result, data_conf)

    elif (flexiscore >= APPROVE_FLEXISCORE_MIN
          and pd_score < APPROVE_PD_MAX
          and ep_result["has_viable_offer"]
          and data_conf >= APPROVE_DATA_CONF_MIN):
        decision, reason_codes, credit_coach_plan = _auto_approve(
            flexiscore, pd_score, graph, ep_result, data_conf)

    else:
        decision, reason_codes, credit_coach_plan = _human_review(
            flexiscore, pd_score, ep_result, gates, data_conf)

    return {
        # Identity
        "customer_id":   customer.get("customer_id", "DEMO"),
        "name":          customer.get("name", "Khách hàng"),
        "customer_type": customer.get("customer_type", ""),
        # Score
        "flexiscore":    flexiscore,
        "risk_tier":     risk_tier,
        "pd":            pd_score,
        "pd_band":       pd_band,
        # Sub-scores
        **{k: v for k, v in score_result.items() if k not in ("flexiscore", "risk_tier")},
        # Gates
        "f0_status":        gates["f0_status"],
        "f0_reasons":       gates["f0_reasons"],
        "f1_status":        gates["f1_status"],
        "f1_reasons":       gates["f1_reasons"],
        "risk_gate_status": gates["risk_gate_status"],
        "dti_stress_ratio": gates.get("dti_stress_ratio", 0),
        "all_gate_reasons": gates["all_reasons"],
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
        "credit_coach_plan": credit_coach_plan,
    }


# ── Builders ──────────────────────────────────────────────────────────────────

def _auto_reject(customer, gates, graph):
    reasons = [
        "Hồ sơ vi phạm quy tắc cứng (F0) — không thể phê duyệt tự động.",
        "ML model không được sử dụng để override quyết định này.",
    ]
    if graph["fraud_ring_flag"]:
        reasons.append("Phát hiện liên kết với vòng gian lận (Fraud Ring).")
    reasons += [f"  • {r}" for r in gates["f0_reasons"]]
    return "AUTO_REJECT", reasons, []


def _credit_coach(customer, gates, score_result, data_conf):
    reasons = ["Hồ sơ chưa đủ điều kiện phê duyệt — xem lộ trình cải thiện."]

    if gates["f1_status"] == "FAIL":
        reasons.append("F1 Stress Test thất bại:")
        reasons += [f"  • {r}" for r in gates["f1_reasons"]]

    if data_conf < COACH_DATA_CONF_MAX:
        reasons.append(
            f"Data confidence thấp ({data_conf:.0%}) — cần bổ sung thêm nguồn dữ liệu "
            f"(EVN, VNPT, ví điện tử, BHXH/eTax)."
        )

    plan = _build_coach_plan(customer, gates, score_result)
    return "CREDIT_COACH", reasons, plan


def _auto_approve(flexiscore, pd_score, graph, ep_result, data_conf):
    rec = ep_result["recommended_offer"]
    reasons = [
        f"FlexiScore {flexiscore:.0f} — vượt ngưỡng phê duyệt ({APPROVE_FLEXISCORE_MIN}).",
        f"PD {pd_score:.1%} — nằm trong giới hạn chấp nhận ({APPROVE_PD_MAX:.0%}).",
        f"Data confidence {data_conf:.0%} — đủ tin cậy.",
        f"Graph Risk {graph['graph_risk_label']} — mạng lưới an toàn.",
    ]
    if rec:
        reasons.append(
            f"Offer tối ưu: {rec.amount/1e6:.1f}M / {rec.tenor_months}T / "
            f"{rec.schedule} — EP dương {rec.expected_profit:,.0f}đ."
        )
    return "AUTO_APPROVE", reasons, []


def _human_review(flexiscore, pd_score, ep_result, gates, data_conf):
    reasons = ["Hồ sơ nằm trong vùng xám — cần thẩm định thêm từ cán bộ tín dụng."]

    if flexiscore < APPROVE_FLEXISCORE_MIN:
        reasons.append(
            f"FlexiScore {flexiscore:.0f} chưa đạt ngưỡng tự động ({APPROVE_FLEXISCORE_MIN})."
        )
    if pd_score >= APPROVE_PD_MAX:
        reasons.append(f"PD {pd_score:.1%} vượt ngưỡng ({APPROVE_PD_MAX:.0%}).")
    if not ep_result["has_viable_offer"]:
        reasons.append("Chưa tìm được offer nào có Expected Profit dương.")
    if COACH_DATA_CONF_MAX <= data_conf < APPROVE_DATA_CONF_MIN:
        reasons.append(
            f"Data confidence {data_conf:.0%} ở mức trung bình — cần xác minh thêm."
        )
    if gates["f1_status"] == "WARN":
        reasons.append("Cảnh báo F1: dòng tiền có dấu hiệu stress nhẹ.")

    return "HUMAN_REVIEW", reasons, []


def _build_coach_plan(customer, gates, score_result):
    plan = ["Lộ trình 90 ngày để đủ điều kiện tái xét:"]

    income_cv   = customer.get("income_cv", 0.5)
    active_days = customer.get("active_days_per_week", 4)
    bill_ratio  = customer.get("bill_on_time_ratio", 0.7)
    data_conf   = customer.get("data_confidence", 0.5)
    cf_drop     = customer.get("cashflow_drop_30d", 0)

    step = 1

    if income_cv > 0.45:
        plan.append(
            f"Bước {step}: Ổn định nguồn thu — giảm biến động thu nhập "
            f"(income_cv hiện tại: {income_cv:.2f}, mục tiêu: <0.45)."
        )
        step += 1

    if active_days < 5:
        plan.append(
            f"Bước {step}: Tăng số ngày làm việc lên ≥5 ngày/tuần "
            f"(hiện tại: {active_days:.1f} ngày)."
        )
        step += 1

    if bill_ratio < 0.90:
        plan.append(
            f"Bước {step}: Thanh toán đúng hạn toàn bộ hóa đơn trong 2 kỳ tiếp theo "
            f"(tỷ lệ hiện tại: {bill_ratio:.0%})."
        )
        step += 1

    if cf_drop > 0.25:
        plan.append(
            f"Bước {step}: Phục hồi dòng tiền — tránh để cashflow giảm >25% "
            f"(mức giảm hiện tại: {cf_drop:.0%})."
        )
        step += 1

    if data_conf < 0.70:
        plan.append(
            f"Bước {step}: Kết nối thêm nguồn dữ liệu — EVN/VNPT, ví điện tử, "
            f"BHXH hoặc eTax để tăng data confidence "
            f"(hiện tại: {data_conf:.0%}, mục tiêu: ≥70%)."
        )
        step += 1

    plan.append(
        f"Bước {step}: Sau 90 ngày cải thiện, nộp lại hồ sơ — "
        f"có thể được xét hạn mức 5–8 triệu, kỳ hạn 6 tháng."
    )
    plan.append("Mẹo: Nộp sao kê BHXH hoặc eTax có thể tăng điểm nhanh trong 30 ngày.")

    return plan
