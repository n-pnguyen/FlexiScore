"""
decision_engine.py
==================
Hàm duy nhất: run_flexiscore_pipeline(customer, model)
  Không hard-code quyết định theo tên use-case.
  Mọi output đều tính từ feature values.

Decision tree (7 nhánh, theo thứ tự ưu tiên):
  1. F0 fail (fraud/hard rule)            → FRAUD_REJECT | FRAUD_REVIEW
  2. F1 fail (stress test)                → CREDIT_COACH | HUMAN_REVIEW
  3. Score>=800, PD<0.10, EP>0, conf>=0.70 → AUTO_APPROVE
  4. Score>=800, EP>0, conf>=0.65          → APPROVE_WITH_OPTIMIZED_OFFER
  5. Score 600-800, EP>0, conf>=0.70, risk_low → APPROVE_WITH_OPTIMIZED_OFFER
  6. Score 600-800 (borderline)           → HUMAN_REVIEW
  7. Score 450-600                        → HUMAN_REVIEW | CREDIT_COACH
  8. Score < 450 | no EP                 → CREDIT_COACH | REJECT_EP_NEGATIVE
"""

from scoring_model   import compute_flexiscore, predict_pd, pd_risk_band
from risk_gates      import run_risk_gates
from graph_risk      import compute_graph_risk
from expected_profit import compute_offers

# ── Ngưỡng quyết định (dễ điều chỉnh) ───────────────────────────────────────
# Trong production các giá trị này được calibrate trên dữ liệu thật + backtest.
PD_AUTO_APPROVE    = 0.10   # PD tối đa để AUTO_APPROVE
SCORE_AUTO_APPROVE = 800    # FlexiScore tối thiểu để AUTO_APPROVE
SCORE_MEDIUM_LOW   = 600    # Ranh giới Tier 2
SCORE_BORDERLINE   = 450    # Ranh giới Tier 3 / Tier 4
GRAPH_RISK_MEDIUM  = 0.40   # Graph risk threshold (trên = elevated)
GRAPH_RISK_HIGH    = 0.70   # Graph risk threshold (trên = fraud review)
DATA_CONF_APPROVE  = 0.70   # data_confidence tối thiểu để auto approve
DATA_CONF_MIN      = 0.60   # data_confidence dưới ngưỡng này → coach
INCOME_CV_F1       = 0.55   # income_cv ngưỡng F1 (sync với risk_gates.py)


def run_flexiscore_pipeline(customer: dict, model) -> dict:
    """
    Pipeline duy nhất — luôn được gọi bất kể use-case nào.
    Input  : customer dict (feature values) + trained model.
    Output : full result dict gồm scores, gates, EP, decision, reason codes.
    """
    # ── Stage 1: Risk Gates ──────────────────────────────────────────────────
    gates      = run_risk_gates(customer)
    f0_fail    = gates["f0_status"] == "FAIL"
    f1_fail    = gates["f1_status"] == "FAIL"

    # ── Stage 2: FlexiScore (rule-based scoring) ─────────────────────────────
    score_result = compute_flexiscore(customer)
    flexiscore   = score_result["flexiscore"]

    # ── Stage 3: PD từ ML model ──────────────────────────────────────────────
    pd_score = predict_pd(model, customer)
    pd_band  = pd_risk_band(pd_score)

    # ── Stage 4: Graph Risk ──────────────────────────────────────────────────
    graph      = compute_graph_risk(customer)
    fraud_flag = graph["fraud_ring_flag"]
    graph_risk = graph["graph_risk_score"]

    # ── Stage 5: Expected Profit (3 offers) ──────────────────────────────────
    ep_result = compute_offers(customer, pd_score)
    has_offer = ep_result["has_viable_offer"]
    rec       = ep_result["recommended_offer"]
    orig_ep   = ep_result["offers"][0].ep_positive  # offer gốc (as-requested)

    # ── Stage 6: Decision Tree ────────────────────────────────────────────────
    data_conf = float(customer.get("data_confidence", 1.0))
    income_cv = float(customer.get("income_cv", 0.0))

    decision, reason_codes, coach_plan = _decide(
        f0_fail, f1_fail, fraud_flag,
        flexiscore, pd_score, graph_risk,
        data_conf, income_cv,
        has_offer, orig_ep, rec,
        customer, gates, score_result, ep_result, graph,
    )

    # ── Build result ──────────────────────────────────────────────────────────
    return {
        "customer_id":    customer.get("customer_id", "DEMO"),
        "name":           customer.get("name", "Khách hàng"),
        "customer_type":  customer.get("customer_type", ""),
        # Scores
        "flexiscore":     flexiscore,
        "risk_tier":      score_result["risk_tier"],
        "pd":             pd_score,
        "pd_band":        pd_band,
        # Sub-scores (5 nhóm)
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
        "risk_gate_status": gates["risk_gate_status"],
        "dti_stress_ratio": gates.get("dti_stress_ratio", 0),
        "all_gate_reasons": gates["all_reasons"],
        # Graph
        "graph_risk_score": graph_risk,
        "trust_score":      graph["trust_score"],
        "fraud_ring_flag":  fraud_flag,
        "graph_risk_label": graph["graph_risk_label"],
        "graph_reasons":    graph["graph_reasons"],
        # EP
        "offers":            ep_result["offers"],
        "recommended_offer": rec,
        "has_viable_offer":  has_offer,
        "original_ep_positive": orig_ep,
        # Decision
        "decision":          decision,
        "reason_codes":      reason_codes,
        "credit_coach_plan": coach_plan,
    }


# ── Alias cho code cũ ────────────────────────────────────────────────────────
make_decision = run_flexiscore_pipeline


# ── Decision tree ─────────────────────────────────────────────────────────────

def _decide(f0_fail, f1_fail, fraud_flag,
            flexiscore, pd_score, graph_risk,
            data_conf, income_cv,
            has_offer, orig_ep, rec,
            customer, gates, score_result, ep_result, graph):
    """
    Pure decision function — không nhìn vào tên use-case, chỉ nhìn giá trị.
    """

    # ── Nhánh 1: F0 Hard Rule fail ────────────────────────────────────────────
    if f0_fail:
        # Phân biệt fraud ring vs vi phạm hard rule khác
        if fraud_flag or graph_risk > GRAPH_RISK_HIGH:
            return _r_fraud_reject(gates, graph)
        else:
            return _r_fraud_review(gates)

    # ── Nhánh 2: F1 Stress Test fail ─────────────────────────────────────────
    if f1_fail:
        # income_cv cao hoặc FlexiScore thấp → cần coach, không chỉ review
        if income_cv > INCOME_CV_F1 or flexiscore < 550:
            return _r_credit_coach(customer, gates, score_result)
        # FlexiScore trung bình → human review có thể xử lý
        return _r_human_review(
            flexiscore, pd_score, ep_result, data_conf,
            "F1 stress test borderline — dòng tiền cần xem xét thêm"
        )

    # ── Nhánh 3: Điều kiện AUTO_APPROVE đầy đủ ───────────────────────────────
    if (flexiscore >= SCORE_AUTO_APPROVE
            and pd_score < PD_AUTO_APPROVE
            and data_conf >= DATA_CONF_APPROVE
            and has_offer):
        return _r_auto_approve(flexiscore, pd_score, graph_risk, data_conf, rec)

    # ── Nhánh 4: FlexiScore cao nhưng chưa đủ auto (PD hơi cao / conf hơi thấp)
    if flexiscore >= SCORE_AUTO_APPROVE and has_offer and data_conf >= 0.65:
        return _r_approve_optimized(
            flexiscore, ep_result,
            f"FlexiScore {flexiscore:.0f} tốt nhưng PD {pd_score:.1%} hoặc "
            f"data_confidence {data_conf:.0%} cần offer cẩn thận hơn"
        )

    # ── Nhánh 5: Tier 2 (600-800), đủ điều kiện approve với offer tối ưu ──────
    if (SCORE_MEDIUM_LOW <= flexiscore < SCORE_AUTO_APPROVE
            and has_offer
            and data_conf >= DATA_CONF_APPROVE
            and graph_risk <= GRAPH_RISK_MEDIUM):
        label = ("offer gốc chưa tối ưu EP" if not orig_ep
                 else "offer tốt nhất được chọn tự động")
        return _r_approve_optimized(flexiscore, ep_result, label)

    # ── Nhánh 6: Tier 2 vùng xám (conf thấp hoặc graph risk trung bình) ──────
    if SCORE_MEDIUM_LOW <= flexiscore < SCORE_AUTO_APPROVE:
        return _r_human_review(
            flexiscore, pd_score, ep_result, data_conf,
            _human_review_reason(flexiscore, pd_score, graph_risk, data_conf, ep_result)
        )

    # ── Nhánh 7: Tier 3 (450-600) ────────────────────────────────────────────
    if SCORE_BORDERLINE <= flexiscore < SCORE_MEDIUM_LOW:
        # Còn cửa nếu EP dương và data đủ
        if has_offer and data_conf >= DATA_CONF_MIN and flexiscore >= 500:
            return _r_human_review(
                flexiscore, pd_score, ep_result, data_conf,
                f"Tier 3 borderline — FlexiScore {flexiscore:.0f}, cần thẩm định"
            )
        return _r_credit_coach(customer, gates, score_result)

    # ── Nhánh 8: Tier 4 (< 450) hoặc không có EP ─────────────────────────────
    if not has_offer:
        return _r_reject_ep(ep_result, flexiscore)

    return _r_credit_coach(customer, gates, score_result)


def _human_review_reason(flexiscore, pd_score, graph_risk, data_conf, ep_result):
    """Tạo lý do rõ ràng tại sao cần human review."""
    parts = []
    if flexiscore < SCORE_AUTO_APPROVE:
        parts.append(f"FlexiScore {flexiscore:.0f} chưa đạt ngưỡng auto ({SCORE_AUTO_APPROVE})")
    if pd_score >= PD_AUTO_APPROVE:
        parts.append(f"PD {pd_score:.1%} trên ngưỡng ({PD_AUTO_APPROVE:.0%})")
    if graph_risk > GRAPH_RISK_MEDIUM:
        parts.append(f"Graph risk elevated ({graph_risk:.2f} > {GRAPH_RISK_MEDIUM})")
    if data_conf < DATA_CONF_APPROVE:
        parts.append(f"Data confidence {data_conf:.0%} cần xác thực thêm")
    if not ep_result["has_viable_offer"]:
        parts.append("Không có offer nào đủ EP dương")
    return " · ".join(parts) if parts else "Hồ sơ vùng xám cần cán bộ xem xét"


# ── Result builders ───────────────────────────────────────────────────────────

def _r_fraud_reject(gates, graph):
    reasons = [
        "Hồ sơ bị từ chối do phát hiện rủi ro gian lận mạng lưới (F0 FAIL).",
        "Pipeline ML không chạy — quyết định dựa trên hard rules.",
        "Hạn mức đề xuất: 0đ.",
    ]
    reasons += [f"  • {r}" for r in gates["f0_reasons"]]
    if graph["graph_risk_score"] > GRAPH_RISK_HIGH:
        reasons.append(
            f"  • Graph risk score {graph['graph_risk_score']:.2f} > {GRAPH_RISK_HIGH} "
            f"— ngưỡng fraud review tự động."
        )
    return "FRAUD_REJECT", reasons, []


def _r_fraud_review(gates):
    reasons = [
        "Hồ sơ vi phạm quy tắc cứng F0 (không phải fraud ring).",
        "Cần cán bộ xác minh thủ công trước khi tiếp tục.",
    ]
    reasons += [f"  • {r}" for r in gates["f0_reasons"]]
    return "FRAUD_REVIEW", reasons, []


def _r_auto_approve(flexiscore, pd_score, graph_risk, data_conf, rec):
    reasons = [
        f"FlexiScore {flexiscore:.0f} >= {SCORE_AUTO_APPROVE} (Tier 1).",
        f"PD {pd_score:.1%} < {PD_AUTO_APPROVE:.0%} — rủi ro thấp.",
        f"Data confidence {data_conf:.0%} >= {DATA_CONF_APPROVE:.0%} — dữ liệu tin cậy.",
        f"Graph risk {graph_risk:.2f} — mạng lưới an toàn.",
    ]
    if rec:
        reasons.append(
            f"Offer tối ưu: {rec.amount/1e6:.1f}M / {rec.tenor_months}T / "
            f"{rec.schedule} | EP = +{rec.expected_profit:,.0f}đ."
        )
    return "AUTO_APPROVE", reasons, []


def _r_approve_optimized(flexiscore, ep_result, context_note):
    rec     = ep_result["recommended_offer"]
    orig    = ep_result["offers"][0]
    reasons = [
        f"FlexiScore {flexiscore:.0f} — đủ điều kiện duyệt với offer tối ưu.",
        f"Lý do tối ưu: {context_note}.",
    ]
    if rec and not orig.ep_positive:
        reasons.append(
            f"Offer gốc ({orig.amount/1e6:.1f}M/{orig.tenor_months}T) EP âm — "
            f"hệ thống tự điều chỉnh sang offer {rec.label}."
        )
    if rec:
        reasons.append(
            f"Offer được chọn: {rec.amount/1e6:.1f}M / {rec.tenor_months}T / "
            f"{rec.schedule} | EP = +{rec.expected_profit:,.0f}đ."
        )
    return "APPROVE_WITH_OPTIMIZED_OFFER", reasons, []


def _r_human_review(flexiscore, pd_score, ep_result, data_conf, main_reason):
    reasons = [
        "Hồ sơ cần thẩm định thêm từ cán bộ tín dụng.",
        main_reason,
    ]
    if ep_result["has_viable_offer"]:
        rec = ep_result["recommended_offer"]
        if rec:
            reasons.append(
                f"Offer tiềm năng để cán bộ xem xét: "
                f"{rec.amount/1e6:.1f}M / {rec.tenor_months}T | EP = +{rec.expected_profit:,.0f}đ."
            )
    else:
        reasons.append("Không tìm được offer nào có EP dương — cần tái cấu trúc khoản vay.")
    return "HUMAN_REVIEW", reasons, []


def _r_credit_coach(customer, gates, score_result):
    reasons = [
        "Hồ sơ chưa đủ điều kiện phê duyệt lần này.",
        "Xem lộ trình cải thiện 90 ngày bên dưới.",
    ]
    if gates["f1_reasons"]:
        reasons.append("Nguyên nhân chính (F1):")
        reasons += [f"  • {r}" for r in gates["f1_reasons"]]
    plan = _build_coach_plan(customer, gates)
    return "CREDIT_COACH", reasons, plan


def _r_reject_ep(ep_result, flexiscore):
    reasons = [
        "Không tìm được cấu trúc khoản vay nào có Expected Profit dương.",
        f"FlexiScore {flexiscore:.0f} — rủi ro quá cao so với lợi nhuận kỳ vọng.",
        "Khuyến nghị: giảm số tiền vay hoặc rút ngắn kỳ hạn và nộp lại.",
    ]
    for o in ep_result["offers"]:
        reasons.append(
            f"  • {o.label}: EP = {o.expected_profit:,.0f}đ — "
            + ("đủ điều kiện" if o.ep_positive else "chưa đủ")
        )
    return "REJECT_EP_NEGATIVE", reasons, []


def _build_coach_plan(customer, gates):
    """Lộ trình 90 ngày cá nhân hoá theo hồ sơ."""
    plan      = []
    income_cv = customer.get("income_cv", 0.5)
    act_days  = customer.get("active_days_per_week", 4)
    bill      = customer.get("bill_on_time_ratio", 0.7)
    data_conf = customer.get("data_confidence", 0.5)
    cf_drop   = customer.get("cashflow_drop_30d", 0)
    step      = 1

    if income_cv > 0.45:
        plan.append(
            f"Bước {step} — Ổn định thu nhập: Giảm biến động về income_cv < 0.45 "
            f"trong 2 tháng liên tiếp (hiện tại: {income_cv:.2f})."
        )
        step += 1

    if act_days < 5:
        plan.append(
            f"Bước {step} — Tăng ngày hoạt động: Duy trì ít nhất 5 ngày/tuần tạo thu nhập "
            f"(hiện tại: {act_days:.1f} ngày/tuần)."
        )
        step += 1

    if bill < 0.90:
        plan.append(
            f"Bước {step} — Kỷ luật thanh toán: Thanh toán đúng hạn 100% hóa đơn "
            f"trong 2 kỳ tới (tỷ lệ hiện tại: {bill:.0%})."
        )
        step += 1

    if cf_drop > 0.25:
        plan.append(
            f"Bước {step} — Phục hồi dòng tiền: Tránh để cashflow giảm > 25% "
            f"(mức giảm gần nhất: {cf_drop:.0%})."
        )
        step += 1

    if data_conf < 0.70:
        plan.append(
            f"Bước {step} — Bổ sung dữ liệu: Kết nối thêm EVN/VNPT, ví điện tử, "
            f"BHXH hoặc eTax để tăng data confidence lên >= 70% "
            f"(hiện tại: {data_conf:.0%})."
        )
        step += 1

    plan.append(
        f"Bước {step} — Tái nộp hồ sơ sau 90 ngày cải thiện: "
        f"Có thể được xét hạn mức 5–8 triệu đồng, kỳ hạn 6 tháng."
    )
    plan.append(
        "Tip nhanh: Nộp sao kê BHXH hoặc eTax có thể tăng điểm trong 30 ngày."
    )
    return plan
