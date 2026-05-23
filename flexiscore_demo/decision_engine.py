"""
decision_engine.py — Quyết định tín dụng cuối cùng.
"""
from scoring_model import compute_flexiscore, predict_pd, pd_risk_band, load_or_train
from risk_gates import run_risk_gates
from graph_risk import compute_graph_risk
from expected_profit import compute_offers


def make_decision(customer: dict, model) -> dict:
    """
    Chạy toàn bộ pipeline và trả kết quả quyết định.

    Returns
    -------
    dict với các keys:
        customer_id, name, flexiscore, risk_tier, pd, pd_band,
        f0_status, f1_status, risk_gate_status, all_reasons,
        graph_risk_score, trust_score, fraud_ring_flag, graph_risk_label, graph_reasons,
        offers, recommended_offer, has_viable_offer,
        decision, decision_label, reason_codes, credit_coach_plan,
        income_stability_score, graph_safety_score, digital_transaction_score,
        financial_commitment_score, platform_behavior_score
    """
    # ── 1. Risk Gates ─────────────────────────────────────────────────────────
    gates = run_risk_gates(customer)

    # ── 2. FlexiScore ─────────────────────────────────────────────────────────
    score_result = compute_flexiscore(customer)
    flexiscore   = score_result["flexiscore"]
    risk_tier    = score_result["risk_tier"]

    # ── 3. PD từ model ────────────────────────────────────────────────────────
    pd_score = predict_pd(model, customer)
    pd_band  = pd_risk_band(pd_score)

    # ── 4. Graph Risk ─────────────────────────────────────────────────────────
    graph = compute_graph_risk(customer)

    # ── 5. Expected Profit ────────────────────────────────────────────────────
    ep_result = compute_offers(customer, pd_score)

    # ── 6. Decision Logic ─────────────────────────────────────────────────────
    reason_codes   = []
    credit_coach   = []
    decision       = ""
    decision_label = ""

    is_fraud   = gates["is_fraud_ring"] or graph["fraud_ring_flag"]
    f0_fail    = gates["f0_status"] == "FAIL"
    f1_fail    = gates["f1_status"] == "FAIL"

    if is_fraud or (f0_fail and gates["is_fraud_ring"]):
        decision       = "FRAUD_REJECT"
        decision_label = "Từ chối — Gian lận mạng lưới"
        reason_codes   = ["Phát hiện liên kết cụm rủi ro/gian lận",
                          "Không sử dụng ML để phê duyệt hồ sơ này",
                          "Hạn mức: 0đ"]
        reason_codes  += gates["all_reasons"]

    elif f0_fail:
        decision       = "FRAUD_REVIEW"
        decision_label = "Chuyển kiểm tra — Vi phạm quy tắc cứng"
        reason_codes   = ["Hồ sơ vi phạm một hoặc nhiều quy tắc F0"] + gates["f0_reasons"]

    elif f1_fail or flexiscore < 450:
        decision       = "CREDIT_COACH"
        decision_label = "Chưa duyệt — Cần cải thiện hồ sơ"
        reason_codes   = ["Hồ sơ chưa đủ điều kiện — xem lộ trình cải thiện 90 ngày"] + gates["f1_reasons"]
        credit_coach   = _build_credit_coach_plan(customer, gates, score_result)

    elif flexiscore >= 800 and pd_score < 0.10 and ep_result["has_viable_offer"]:
        decision       = "AUTO_APPROVE_WITH_OPTIMIZED_OFFER"
        decision_label = "Phê duyệt tự động — Offer tối ưu"
        rec = ep_result["recommended_offer"]
        reason_codes   = [
            f"FlexiScore {flexiscore:.0f} — Tier 1 Very Low Risk",
            f"PD {pd_score:.1%} — Rủi ro thấp",
            f"Graph Risk {graph['graph_risk_label']} — Mạng lưới an toàn",
            f"Offer tối ưu: {rec.amount/1e6:.1f}M / {rec.tenor_months}T / {rec.schedule}",
            f"Expected Profit dương: {rec.expected_profit:,.0f}đ",
        ]

    elif flexiscore >= 600 and ep_result["has_viable_offer"]:
        decision       = "APPROVE_WITH_OPTIMIZED_OFFER"
        decision_label = "Phê duyệt với offer tối ưu"
        rec = ep_result["recommended_offer"]
        reason_codes   = [
            f"FlexiScore {flexiscore:.0f} — Tier 2",
            f"PD {pd_score:.1%}",
            f"Offer đề xuất: {rec.amount/1e6:.1f}M / {rec.tenor_months}T",
            f"Expected Profit: {rec.expected_profit:,.0f}đ",
        ]

    elif flexiscore >= 450:
        decision       = "HUMAN_REVIEW"
        decision_label = "Chuyển thẩm định con người"
        reason_codes   = [
            f"FlexiScore {flexiscore:.0f} — Tier 3 Borderline",
            "Cần thẩm định thêm từ cán bộ tín dụng",
        ]

    else:
        if not ep_result["has_viable_offer"]:
            decision = "REJECT_EP_NEGATIVE"
            decision_label = "Từ chối — Không có offer sinh lời"
            reason_codes = ["Không tìm được cấu trúc khoản vay nào có Expected Profit dương"]
        else:
            decision       = "CREDIT_COACH"
            decision_label = "Chưa duyệt — Cần cải thiện hồ sơ"
            reason_codes   = [f"FlexiScore {flexiscore:.0f} — dưới ngưỡng duyệt"]
            credit_coach   = _build_credit_coach_plan(customer, gates, score_result)

    return {
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
        "all_reasons":      gates["all_reasons"],
        # Graph
        **{f"graph_{k}" if k not in ("trust_score", "fraud_ring_flag") else k: v
           for k, v in graph.items()},
        "graph_risk_score": graph["graph_risk_score"],
        "trust_score":      graph["trust_score"],
        "fraud_ring_flag":  graph["fraud_ring_flag"],
        "graph_risk_label": graph["graph_risk_label"],
        "graph_reasons":    graph["graph_reasons"],
        # EP
        "offers":           ep_result["offers"],
        "recommended_offer": ep_result["recommended_offer"],
        "has_viable_offer": ep_result["has_viable_offer"],
        # Decision
        "decision":         decision,
        "decision_label":   decision_label,
        "reason_codes":     reason_codes,
        "credit_coach_plan": credit_coach,
    }


def _build_credit_coach_plan(customer: dict, gates: dict, score: dict) -> list:
    """Tạo lộ trình 90 ngày cho Credit Coach case."""
    plan = []
    income_cv = customer.get("income_cv", 0.5)
    active_days = customer.get("active_days_per_week", 4)
    bill_ratio = customer.get("bill_on_time_ratio", 0.7)
    data_conf  = customer.get("data_confidence", 0.5)

    plan.append("📋 Lộ trình 90 ngày để đủ điều kiện tái xét:")

    if active_days < 5:
        plan.append(f"1️⃣ Tăng ngày hoạt động tạo thu nhập lên tối thiểu 5 ngày/tuần (hiện tại: {active_days:.1f} ngày)")
    else:
        plan.append("1️⃣ Duy trì hoạt động tạo thu nhập tối thiểu 5 ngày/tuần ✓")

    if income_cv > 0.45:
        plan.append(f"2️⃣ Ổn định thu nhập — giảm income_cv xuống dưới 0.45 trong 2 tháng liên tiếp (hiện tại: {income_cv:.2f})")
    else:
        plan.append(f"2️⃣ Tiếp tục duy trì thu nhập ổn định (income_cv: {income_cv:.2f}) ✓")

    if bill_ratio < 0.90:
        plan.append(f"3️⃣ Thanh toán đúng hạn 2 kỳ hóa đơn tiếp theo (bill_on_time hiện tại: {bill_ratio:.0%})")
    else:
        plan.append("3️⃣ Tiếp tục thanh toán hóa đơn đúng hạn ✓")

    if data_conf < 0.70:
        plan.append("4️⃣ Kết nối thêm nguồn dữ liệu: EVN/VNPT, ví điện tử, hoặc tài khoản ngân hàng số để tăng data_confidence")
    else:
        plan.append("4️⃣ Dữ liệu đã đủ tin cậy — giữ nguyên ✓")

    plan.append("5️⃣ Sau 90 ngày: tái nộp hồ sơ — có thể được xét hạn mức 5–8 triệu với kỳ hạn 6 tháng")
    plan.append("💡 Mẹo: Nộp thêm sao kê BHXH hoặc eTax để tăng điểm nhanh hơn")

    return plan
