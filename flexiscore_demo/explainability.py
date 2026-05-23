"""
explainability.py — Reason codes và SHAP explanation (fallback rule-based).
"""
import numpy as np


# ── Rule-based reason codes ───────────────────────────────────────────────────

RULES = [
    # (feature, condition_fn, positive_msg, negative_msg, weight)
    ("income_cv",           lambda v: v < 0.30,  "Thu nhập ổn định",                   "Thu nhập biến động mạnh",                    "high"),
    ("active_days_per_week",lambda v: v >= 5,    "Hoạt động tạo thu nhập đều đặn",     "Số ngày làm việc/tuần thấp",                 "high"),
    ("cashflow_drop_30d",   lambda v: v < 0.20,  "Dòng tiền ổn định 30 ngày qua",      "Dòng tiền sụt giảm đáng kể 30 ngày qua",    "high"),
    ("bill_on_time_ratio",  lambda v: v >= 0.90, "Thanh toán hóa đơn đúng hạn",        "Lịch sử thanh toán hóa đơn chưa tốt",        "high"),
    ("utility_payment_delay", lambda v: v <= 2,  "Thanh toán điện/nước đúng hạn",      "Chậm thanh toán hóa đơn điện/nước",         "medium"),
    ("late_payment_count_6m", lambda v: v <= 1,  "Ít lần thanh toán trễ hạn",          f"Có nhiều lần thanh toán trễ trong 6 tháng", "medium"),
    ("trust_score",         lambda v: v >= 0.80, "Điểm tin cậy mạng lưới cao",         "Điểm tin cậy mạng lưới thấp",               "high"),
    ("graph_risk_score",    lambda v: v <= 0.20, "Rủi ro mạng lưới thấp",              "Rủi ro mạng lưới cao — cần kiểm tra",       "high"),
    ("fraud_ring_flag",     lambda v: not v,     "Không phát hiện liên kết gian lận",  "Phát hiện liên kết với cụm gian lận",       "critical"),
    ("shared_device_count", lambda v: v < 2,     "Thiết bị không dùng chung",          "Thiết bị liên kết với nhiều hồ sơ khác",    "high"),
    ("bad_neighbor_count",  lambda v: v == 0,    "Không liên kết hồ sơ nợ xấu",        "Có liên kết tới hồ sơ nợ xấu",             "high"),
    ("data_confidence",     lambda v: v >= 0.75, "Dữ liệu đủ tin cậy",                 "Dữ liệu chưa đủ tin cậy — cần bổ sung",    "medium"),
    ("rating_avg",          lambda v: v >= 4.5,  "Đánh giá nền tảng tốt",              "Đánh giá nền tảng trung bình/thấp",          "low"),
    ("completion_rate",     lambda v: v >= 0.90, "Tỷ lệ hoàn thành đơn/cuốc cao",      "Tỷ lệ hoàn thành đơn/cuốc thấp",            "low"),
    ("cancel_rate",         lambda v: v < 0.10,  "Tỷ lệ hủy đơn thấp",                "Tỷ lệ hủy đơn cao",                         "low"),
    ("monthly_surplus_ratio", lambda v: v > 0.20,"Dòng tiền dư hàng tháng tốt",       "Tỷ lệ dòng tiền dư thấp",                   "medium"),
    ("expense_to_income_ratio", lambda v: v < 0.65, "Chi tiêu trong kiểm soát",        "Chi tiêu chiếm tỷ trọng cao trong thu nhập", "medium"),
]


def generate_reason_codes(customer: dict, result: dict) -> dict:
    """
    Trả về positive_factors và negative_factors bằng tiếng Việt.
    Thử dùng SHAP trước, fallback sang rule-based.
    """
    positive = []
    negative = []

    for feat, cond, pos_msg, neg_msg, weight in RULES:
        val = customer.get(feat, None)
        if val is None:
            continue
        if cond(val):
            positive.append(f"✅ {pos_msg}")
        else:
            prefix = "🚨" if weight == "critical" else ("⚠️" if weight == "high" else "💡")
            negative.append(f"{prefix} {neg_msg}")

    # EP signal
    if result.get("has_viable_offer"):
        rec = result.get("recommended_offer")
        if rec:
            positive.append(f"✅ Khoản vay đạt lợi nhuận kỳ vọng (+{rec.expected_profit:,.0f}đ)")
    else:
        negative.append("⚠️ Khoản vay chưa đạt lợi nhuận kỳ vọng với yêu cầu hiện tại")

    return {
        "positive_factors": positive,
        "negative_factors": negative,
    }


def try_shap_explanation(model, customer: dict, feature_cols: list) -> dict:
    """
    Cố gắng lấy SHAP values. Nếu không được thì trả None.
    """
    try:
        import shap
        import pandas as pd

        row = {col: customer.get(col, 0) for col in feature_cols}
        X   = pd.DataFrame([row])

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)

        # Binary classification: shap_vals có thể là list[2 arrays]
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]

        shap_map = {col: float(vals[i]) for i, col in enumerate(feature_cols)}
        top_pos  = sorted([(k, v) for k, v in shap_map.items() if v > 0], key=lambda x: -x[1])[:5]
        top_neg  = sorted([(k, v) for k, v in shap_map.items() if v < 0], key=lambda x: x[1])[:5]

        return {
            "shap_available": True,
            "shap_values":    shap_map,
            "shap_top_risk":  top_pos,
            "shap_top_safe":  top_neg,
        }
    except Exception:
        return {"shap_available": False}
