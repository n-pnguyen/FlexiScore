"""
risk_gates.py — F0 Hard Rules và F1 Stress Test.
"""


def check_f0(customer: dict) -> dict:
    """F0 Hard Rules — chặn trước ML."""
    reasons = []
    status = "PASS"

    if customer.get("fraud_ring_flag", 0):
        status = "FAIL"
        reasons.append("fraud_ring_flag: Phát hiện liên kết với cụm gian lận/rủi ro mạng lưới")

    if customer.get("shared_device_count", 0) >= 4:
        status = "FAIL"
        reasons.append(f"shared_device_count={customer['shared_device_count']}: Thiết bị dùng chung với quá nhiều hồ sơ khác")

    if customer.get("bad_neighbor_count", 0) >= 3:
        status = "FAIL"
        reasons.append(f"bad_neighbor_count={customer['bad_neighbor_count']}: Có liên kết tới hồ sơ nợ xấu/gian lận")

    if customer.get("income_cv", 0) > 0.75:
        status = "FAIL"
        reasons.append(f"income_cv={customer['income_cv']:.2f}: Thu nhập biến động quá mạnh, không đủ điều kiện")

    if customer.get("data_confidence", 1.0) < 0.40:
        status = "FAIL"
        reasons.append(f"data_confidence={customer['data_confidence']:.2f}: Dữ liệu chưa đủ tin cậy — cần bổ sung thêm nguồn")

    is_fraud = bool(customer.get("fraud_ring_flag", 0))

    return {
        "f0_status": status,
        "f0_reasons": reasons,
        "is_fraud_ring": is_fraud,
    }


def check_f1(customer: dict) -> dict:
    """F1 Stress Test — kiểm tra khả năng trả nợ khi thu nhập stress."""
    reasons = []
    status = "PASS"

    avg_income = customer.get("avg_monthly_income", 0)
    requested_amount = customer.get("requested_amount", 0)
    tenor = max(customer.get("requested_tenor_months", 12), 1)

    stressed_income = avg_income * 0.70  # giảm 30%
    estimated_monthly_payment = requested_amount / tenor

    dti_ratio = estimated_monthly_payment / max(stressed_income, 1)

    if dti_ratio > 0.35:
        status = "FAIL"
        reasons.append(
            f"DTI stress={dti_ratio:.0%} > 35%: Khoản vay vượt quá khả năng chi trả khi thu nhập giảm 30%. "
            f"Trả góp ước tính {estimated_monthly_payment:,.0f}đ/tháng trên thu nhập stress "
            f"{stressed_income:,.0f}đ/tháng."
        )

    # Income volatility risky enough to fail stress test
    income_cv = customer.get("income_cv", 0)
    if income_cv > 0.55:
        status = "FAIL"
        reasons.append(
            f"income_cv={income_cv:.2f} > 0.55: Thu nhập biến động quá cao — không đáp ứng stress test. "
            f"Cần ổn định thu nhập ít nhất 2 tháng trước khi tái xét."
        )

    # Cashflow drop cảnh báo thêm
    if customer.get("cashflow_drop_30d", 0) > 0.30:
        reasons.append(
            f"cashflow_drop_30d={customer['cashflow_drop_30d']:.0%}: Dòng tiền đã giảm mạnh trong 30 ngày qua"
        )
        if status == "PASS":
            status = "WARN"

    return {
        "f1_status": status,
        "f1_reasons": reasons,
        "stressed_income": stressed_income,
        "estimated_monthly_payment": estimated_monthly_payment,
        "dti_stress_ratio": round(dti_ratio, 4),
    }


def run_risk_gates(customer: dict) -> dict:
    """Chạy toàn bộ risk gates và trả kết quả tổng hợp."""
    f0 = check_f0(customer)
    f1 = check_f1(customer)

    overall = "PASS"
    if f0["f0_status"] == "FAIL":
        overall = "FAIL"
    elif f1["f1_status"] == "FAIL":
        overall = "FAIL"

    return {
        **f0,
        **f1,
        "risk_gate_status": overall,
        "all_reasons": f0["f0_reasons"] + f1["f1_reasons"],
    }
