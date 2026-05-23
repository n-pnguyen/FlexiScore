"""
app.py — FlexiScore Demo Dashboard (Streamlit).
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Lazy init để không import nặng khi chưa cần ──────────────────────────────
@st.cache_resource(show_spinner="Đang tải model FlexiScore...")
def _load_model():
    from data_generator import generate_dataset
    from scoring_model import load_or_train
    df = generate_dataset()
    model, metrics = load_or_train(df)
    return model, metrics, df


def get_model():
    return _load_model()


# ── Demo cases ────────────────────────────────────────────────────────────────
DEMO_CASES = {
    "😊 Happy Path — Nguyễn Văn A (Tài xế Grab)": {
        "customer_id": "DEMO_001",
        "name": "Nguyễn Văn A",
        "customer_type": "gig_worker",
        "age": 32,
        "platform_tenure_months": 18,
        "data_months_available": 18,
        "identity_verified": 1,
        "avg_monthly_income": 15_000_000,
        "income_cv": 0.15,
        "cashflow_drop_30d": 0.0,
        "active_days_per_week": 6.0,
        "monthly_inflow_count": 48,
        "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40,
        "saving_buffer": 8_000_000,
        "bill_on_time_ratio": 0.98,
        "utility_payment_delay": 0,
        "late_payment_count_6m": 0,
        "avg_payment_delay_days": 0,
        "wallet_activity_consistency": 0.92,
        "seller_revenue_growth": 0.0,
        "refund_rate": 0.0,
        "order_frequency": 6.0,
        "repeat_customer_ratio": 0.75,
        "rating_avg": 4.8,
        "cancel_rate": 0.03,
        "completion_rate": 0.97,
        "shared_device_count": 0,
        "shared_ip_count": 0,
        "bad_neighbor_count": 0,
        "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.02,
        "trust_score": 0.92,
        "graph_risk_score": 0.08,
        "data_confidence": 0.88,
        "missing_data_ratio": 0.08,
        "source_reliability_score": 0.90,
        "requested_amount": 20_000_000,
        "requested_tenor_months": 12,
    },
    "🚨 Risk-First — Trần Thị B (Seller Online)": {
        "customer_id": "DEMO_002",
        "name": "Trần Thị B",
        "customer_type": "seller_online",
        "age": 28,
        "platform_tenure_months": 24,
        "data_months_available": 20,
        "identity_verified": 1,
        "avg_monthly_income": 25_000_000,
        "income_cv": 0.20,
        "cashflow_drop_30d": 0.05,
        "active_days_per_week": 6.0,
        "monthly_inflow_count": 120,
        "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40,
        "saving_buffer": 15_000_000,
        "bill_on_time_ratio": 0.96,
        "utility_payment_delay": 0,
        "late_payment_count_6m": 0,
        "avg_payment_delay_days": 0,
        "wallet_activity_consistency": 0.88,
        "seller_revenue_growth": 0.18,
        "refund_rate": 0.08,
        "order_frequency": 8.0,
        "repeat_customer_ratio": 0.65,
        "rating_avg": 4.7,
        "cancel_rate": 0.05,
        "completion_rate": 0.96,
        "shared_device_count": 4,
        "shared_ip_count": 6,
        "bad_neighbor_count": 3,
        "fraud_ring_flag": 1,
        "circular_transaction_ratio": 0.28,
        "trust_score": 0.28,
        "graph_risk_score": 0.91,
        "data_confidence": 0.80,
        "missing_data_ratio": 0.20,
        "source_reliability_score": 0.75,
        "requested_amount": 40_000_000,
        "requested_tenor_months": 24,
    },
    "📚 Credit Coach — Lê Văn C (Freelancer)": {
        "customer_id": "DEMO_003",
        "name": "Lê Văn C",
        "customer_type": "freelancer",
        "age": 26,
        "platform_tenure_months": 8,
        "data_months_available": 6,
        "identity_verified": 1,
        "avg_monthly_income": 12_000_000,
        "income_cv": 0.65,
        "cashflow_drop_30d": 0.42,
        "active_days_per_week": 3.5,
        "monthly_inflow_count": 8,
        "expense_to_income_ratio": 0.72,
        "monthly_surplus_ratio": 0.20,
        "saving_buffer": 2_000_000,
        "bill_on_time_ratio": 0.70,
        "utility_payment_delay": 2,
        "late_payment_count_6m": 3,
        "avg_payment_delay_days": 4.5,
        "wallet_activity_consistency": 0.55,
        "seller_revenue_growth": 0.0,
        "refund_rate": 0.0,
        "order_frequency": 3.0,
        "repeat_customer_ratio": 0.40,
        "rating_avg": 4.2,
        "cancel_rate": 0.15,
        "completion_rate": 0.82,
        "shared_device_count": 0,
        "shared_ip_count": 1,
        "bad_neighbor_count": 0,
        "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.03,
        "trust_score": 0.75,
        "graph_risk_score": 0.20,
        "data_confidence": 0.55,
        "missing_data_ratio": 0.38,
        "source_reliability_score": 0.60,
        "requested_amount": 15_000_000,
        "requested_tenor_months": 12,
    },
}

TYPE_LABELS = {
    "gig_worker": "Tài xế công nghệ",
    "seller_online": "Seller online",
    "freelancer": "Freelancer",
    "small_merchant": "Tiểu thương",
    "thin_file_customer": "Khách hàng hồ sơ mỏng",
}

DECISION_COLORS = {
    "AUTO_APPROVE_WITH_OPTIMIZED_OFFER": "#2e7d32",
    "APPROVE_WITH_OPTIMIZED_OFFER":      "#388e3c",
    "HUMAN_REVIEW":                      "#f57c00",
    "CREDIT_COACH":                      "#e65100",
    "FRAUD_REJECT":                      "#c62828",
    "FRAUD_REVIEW":                      "#b71c1c",
    "REJECT_EP_NEGATIVE":                "#6d4c41",
}

DECISION_EMOJI = {
    "AUTO_APPROVE_WITH_OPTIMIZED_OFFER": "✅",
    "APPROVE_WITH_OPTIMIZED_OFFER":      "✅",
    "HUMAN_REVIEW":                      "🔍",
    "CREDIT_COACH":                      "📚",
    "FRAUD_REJECT":                      "🚫",
    "FRAUD_REVIEW":                      "⚠️",
    "REJECT_EP_NEGATIVE":                "❌",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color_badge(label: str, color: str) -> str:
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-weight:bold;font-size:0.85em">{label}</span>'


def _score_gauge(value: float, max_val: float = 1000, title: str = "FlexiScore"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1},
            "bar":  {"color": "#1565c0"},
            "steps": [
                {"range": [0, 450],   "color": "#ffcdd2"},
                {"range": [450, 600], "color": "#fff9c4"},
                {"range": [600, 800], "color": "#c8e6c9"},
                {"range": [800, 1000],"color": "#a5d6a7"},
            ],
            "threshold": {
                "line": {"color": "#c62828", "width": 3},
                "thickness": 0.75,
                "value": 600,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=10, r=10))
    return fig


def _bar_subscores(result: dict):
    labels = ["Thu nhập", "Graph Safety", "Giao dịch số", "Tài chính", "Nền tảng"]
    values = [
        result.get("income_stability_score", 0),
        result.get("graph_safety_score", 0),
        result.get("digital_transaction_score", 0),
        result.get("financial_commitment_score", 0),
        result.get("platform_behavior_score", 0),
    ]
    weights = [35, 25, 20, 15, 5]
    colors = ["#1565c0", "#6a1b9a", "#00695c", "#e65100", "#37474f"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.0f}/100<br>({w}%)" for v, w in zip(values, weights)],
        textposition="outside",
    ))
    fig.update_layout(
        title="Điểm thành phần FlexiScore",
        yaxis=dict(range=[0, 120], title="Điểm (0–100)"),
        height=300,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def _ep_chart(offers):
    names = [o.label for o in offers]
    eps   = [o.expected_profit / 1000 for o in offers]  # đơn vị nghìn đồng
    colors = ["#43a047" if o.ep_positive else "#e53935" for o in offers]

    fig = go.Figure(go.Bar(
        x=names, y=eps, marker_color=colors,
        text=[f"{o.expected_profit/1e3:,.0f}K" for o in offers],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Expected Profit so sánh các offer (nghìn đồng)",
        yaxis_title="EP (nghìn đồng)",
        height=280,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def _graph_risk_gauge(score: float):
    color = "#c62828" if score >= 0.70 else ("#f57c00" if score >= 0.40 else "#2e7d32")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": "Graph Risk Score", "font": {"size": 14}},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "#e8f5e9"},
                {"range": [40, 70], "color": "#fff8e1"},
                {"range": [70, 100],"color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="FlexiScore Demo",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <h1 style='text-align:center;color:#1565c0'>💳 FlexiScore Demo</h1>
    <p style='text-align:center;color:#555;font-size:1.05em'>
    Alternative Credit Scoring for Thin-file Customers &nbsp;|&nbsp;
    <em>Risk-first · Cash-flow optimized · Explainable · &lt; 3 phút</em>
    </p>
    <hr>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Cấu hình Demo")

        case_choice = st.selectbox("Chọn use-case:", list(DEMO_CASES.keys()))
        base = DEMO_CASES[case_choice].copy()

        st.markdown("---")
        st.subheader("Chỉnh thông số")

        base["avg_monthly_income"] = st.number_input(
            "Thu nhập TB tháng (VND)", value=int(base["avg_monthly_income"]),
            min_value=1_000_000, max_value=100_000_000, step=1_000_000, format="%d")

        base["income_cv"] = st.slider(
            "Income CV (biến động)", 0.0, 1.0, float(base["income_cv"]), 0.01)

        base["requested_amount"] = st.number_input(
            "Số tiền vay (VND)", value=int(base["requested_amount"]),
            min_value=1_000_000, max_value=200_000_000, step=1_000_000, format="%d")

        base["requested_tenor_months"] = st.selectbox(
            "Kỳ hạn (tháng)", [3, 6, 9, 12, 18, 24],
            index=[3,6,9,12,18,24].index(int(base["requested_tenor_months"])))

        base["shared_device_count"] = st.slider(
            "Shared device count", 0, 10, int(base["shared_device_count"]), 1)

        base["bad_neighbor_count"] = st.slider(
            "Bad neighbor count", 0, 8, int(base["bad_neighbor_count"]), 1)

        base["data_confidence"] = st.slider(
            "Data confidence", 0.0, 1.0, float(base["data_confidence"]), 0.01)

        st.markdown("---")
        run_btn = st.button("🚀 Chạy FlexiScore Pipeline", use_container_width=True, type="primary")

    # ── Load model ────────────────────────────────────────────────────────────
    model, metrics, train_df = get_model()

    # ── Run pipeline on button press or first load ────────────────────────────
    if run_btn or "result" not in st.session_state:
        with st.spinner("Đang chạy pipeline..."):
            from decision_engine import make_decision
            from explainability import generate_reason_codes
            result = make_decision(base, model)
            explain = generate_reason_codes(base, result)
            st.session_state["result"]  = result
            st.session_state["explain"] = explain
            st.session_state["customer"] = base

    result  = st.session_state["result"]
    explain = st.session_state["explain"]
    customer = st.session_state["customer"]

    decision = result["decision"]
    dec_color = DECISION_COLORS.get(decision, "#555")
    dec_emoji = DECISION_EMOJI.get(decision, "❓")

    # ── Decision Banner ───────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{dec_color};color:white;padding:14px 20px;border-radius:10px;"
        f"text-align:center;font-size:1.3em;font-weight:bold;margin-bottom:18px'>"
        f"{dec_emoji} {result['decision_label']}</div>",
        unsafe_allow_html=True
    )

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_customer, col_bank = st.columns([1, 1], gap="large")

    # ────────────────────────── CỘT TRÁI: CUSTOMER VIEW ──────────────────────
    with col_customer:
        st.subheader("👤 Customer View")

        ctype_label = TYPE_LABELS.get(result["customer_type"], result["customer_type"])
        st.markdown(f"""
        | Thông tin | Giá trị |
        |-----------|---------|
        | **Tên** | {result['name']} |
        | **Loại KH** | {ctype_label} |
        | **Thu nhập TB** | {customer['avg_monthly_income']:,.0f} đ/tháng |
        | **Số tiền vay** | {customer['requested_amount']:,.0f} đ |
        | **Kỳ hạn** | {customer['requested_tenor_months']} tháng |
        """)

        st.markdown("---")

        # Kết quả cho khách
        if "APPROVE" in decision:
            st.success("🎉 Chúc mừng! Khoản vay của bạn được phê duyệt.")
            rec = result.get("recommended_offer")
            if rec:
                st.markdown(f"""
                **Offer được đề xuất: {rec.label}**

                | | |
                |--|--|
                | Số tiền | **{rec.amount:,.0f} đ** |
                | Kỳ hạn | **{rec.tenor_months} tháng** |
                | Lịch trả | **{rec.schedule}** |
                | Lãi suất | **{rec.interest_rate:.0%}/năm** |
                | Trả góp ước tính | **{rec.monthly_payment:,.0f} đ/lần** |
                """)

        elif decision in ("FRAUD_REJECT", "FRAUD_REVIEW"):
            st.error("🚫 Hồ sơ cần được xác minh bổ sung trước khi xử lý.")
            st.warning("Vui lòng liên hệ bộ phận hỗ trợ để được hướng dẫn cụ thể.")
            st.markdown("**Lưu ý:** Kết quả này được tạo ra tự động dựa trên phân tích mạng lưới. "
                        "Bạn có quyền yêu cầu xem xét lại từ cán bộ tín dụng.")

        elif decision == "CREDIT_COACH":
            st.warning("📚 Hồ sơ chưa đủ điều kiện — nhưng bạn có thể cải thiện!")
            st.markdown("**Lộ trình 90 ngày để đủ điều kiện tái xét:**")
            plan = result.get("credit_coach_plan", [])
            for item in plan:
                st.markdown(item)

        elif decision == "HUMAN_REVIEW":
            st.info("🔍 Hồ sơ đang được chuyển sang thẩm định thêm từ cán bộ tín dụng.")

        else:
            st.error("❌ Rất tiếc, khoản vay không được phê duyệt lần này.")

        # Positive & negative factors for customer
        st.markdown("---")
        st.markdown("**Điểm mạnh của hồ sơ:**")
        for f in explain["positive_factors"][:5]:
            st.markdown(f)
        if explain["negative_factors"]:
            st.markdown("**Cần cải thiện:**")
            for f in explain["negative_factors"][:5]:
                st.markdown(f)

    # ────────────────────────── CỘT PHẢI: BANK/RISK VIEW ────────────────────
    with col_bank:
        st.subheader("🏦 Bank / Risk Officer View")

        # FlexiScore gauge
        st.plotly_chart(_score_gauge(result["flexiscore"]), use_container_width=True)

        # Key metrics in 3 cols
        m1, m2, m3 = st.columns(3)
        m1.metric("FlexiScore", f"{result['flexiscore']:.0f}", help="Thang 0–1000")
        m2.metric("PD",  f"{result['pd']:.1%}", help="Probability of Default")
        m3.metric("Risk Band", result["pd_band"])

        # Gates
        g1, g2 = st.columns(2)
        f0_icon = "✅" if result["f0_status"] == "PASS" else "❌"
        f1_icon = "✅" if result["f1_status"] == "PASS" else ("⚠️" if result["f1_status"] == "WARN" else "❌")
        g1.markdown(f"**F0 Hard Rules:** {f0_icon} {result['f0_status']}")
        g2.markdown(f"**F1 Stress Test:** {f1_icon} {result['f1_status']}")

        dti = result.get("dti_stress_ratio", 0)
        if dti:
            st.caption(f"DTI stress: {dti:.1%} (ngưỡng 35%)")

        st.markdown(f"**Risk Tier:** {result['risk_tier']}")

        # Graph risk
        st.markdown("---")
        st.plotly_chart(_graph_risk_gauge(result["graph_risk_score"]), use_container_width=True)

        gr1, gr2 = st.columns(2)
        gr_label = result.get("graph_risk_label", "")
        gr_color = "🔴" if gr_label == "HIGH" else ("🟡" if gr_label == "MEDIUM" else "🟢")
        gr1.markdown(f"**Graph Risk:** {gr_color} {gr_label}")
        gr2.metric("Trust Score", f"{result['trust_score']:.2f}")

        # Graph reasons
        with st.expander("Chi tiết Graph Risk"):
            for r in result.get("graph_reasons", []):
                st.markdown(f"• {r}")

        # Subscores bar chart
        st.markdown("---")
        st.plotly_chart(_bar_subscores(result), use_container_width=True)

        # Offer table
        st.markdown("---")
        st.markdown("**📊 So sánh Loan Offers**")
        offers = result.get("offers", [])
        if offers:
            offer_rows = []
            for o in offers:
                rec_marker = " ⭐" if (result["recommended_offer"] and o.label == result["recommended_offer"].label) else ""
                offer_rows.append({
                    "Offer": o.label + rec_marker,
                    "Số tiền (đ)": f"{o.amount:,.0f}",
                    "Kỳ hạn (T)": o.tenor_months,
                    "Lịch trả": o.schedule,
                    "EP (đ)": f"{o.expected_profit:,.0f}",
                    "Khả thi": "✅" if o.ep_positive else "❌",
                })
            st.dataframe(pd.DataFrame(offer_rows), use_container_width=True, hide_index=True)
            st.plotly_chart(_ep_chart(offers), use_container_width=True)

        # Reason codes
        st.markdown("---")
        st.markdown("**📋 Reason Codes (Bank View)**")
        for rc in result.get("reason_codes", []):
            st.markdown(f"• {rc}")

        if result.get("f0_reasons"):
            with st.expander("Chi tiết F0 reasons"):
                for r in result["f0_reasons"]:
                    st.markdown(f"• {r}")

        if result.get("f1_reasons"):
            with st.expander("Chi tiết F1 reasons"):
                for r in result["f1_reasons"]:
                    st.markdown(f"• {r}")

    # ── Model metrics (collapsible) ───────────────────────────────────────────
    with st.expander("📈 Model Performance Metrics"):
        if metrics:
            mc = st.columns(4)
            mc[0].metric("ROC-AUC",   metrics.get("roc_auc", "—"))
            mc[1].metric("Accuracy",  metrics.get("accuracy", "—"))
            mc[2].metric("Precision", metrics.get("precision", "—"))
            mc[3].metric("Recall",    metrics.get("recall", "—"))
        else:
            st.info("Model đã load từ file lưu — chạy `python scoring_model.py` để xem metrics.")

    # ── Pipeline log ──────────────────────────────────────────────────────────
    with st.expander("🔧 Pipeline Debug Log"):
        import json
        debug = {k: v for k, v in result.items()
                 if k not in ("offers", "recommended_offer", "credit_coach_plan")}
        # Convert non-serializable
        for k, v in debug.items():
            if hasattr(v, "__float__"):
                debug[k] = float(v)
        st.json(debug)

    st.markdown(
        "<hr><p style='text-align:center;color:#aaa;font-size:0.8em'>"
        "FlexiScore Demo — Dữ liệu giả lập, không phải dữ liệu thật của SHB/ngân hàng.</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
