"""
app.py — FlexiScore Demo Dashboard (Streamlit).
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Model (cached, chỉ load một lần) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi tạo FlexiScore model...")
def _bootstrap():
    from data_generator import generate_dataset
    from scoring_model import load_or_train
    df = generate_dataset()
    model, metrics = load_or_train(df)
    return model, metrics


# ── Demo cases (chỉ là điểm khởi đầu — sliders mới quyết định) ───────────────
DEMO_CASES = {
    "Happy Path — Nguyễn Văn A": {
        "customer_id": "DEMO_001",
        "name": "Nguyễn Văn A",
        "customer_type": "gig_worker",
        "age": 32,
        "platform_tenure_months": 18,
        "data_months_available": 18,
        "identity_verified": 1,
        "avg_monthly_income": 15_000_000,
        "income_cv": 0.15,
        "cashflow_drop_30d": 0.00,
        "active_days_per_week": 6.0,
        "monthly_inflow_count": 48,
        "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40,
        "saving_buffer": 8_000_000,
        "bill_on_time_ratio": 0.98,
        "utility_payment_delay": 0,
        "late_payment_count_6m": 0,
        "avg_payment_delay_days": 0.0,
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
    "Risk-First — Trần Thị B": {
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
        "avg_payment_delay_days": 0.0,
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
    "Credit Coach — Lê Văn C": {
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
    "Human Review — Nguyễn Thị D": {
        "customer_id": "DEMO_004",
        "name": "Nguyễn Thị D",
        "customer_type": "small_merchant",
        "age": 35,
        "platform_tenure_months": 10,
        "data_months_available": 9,
        "identity_verified": 1,
        "avg_monthly_income": 18_000_000,
        "income_cv": 0.38,
        "cashflow_drop_30d": 0.18,
        "active_days_per_week": 5.0,
        "monthly_inflow_count": 30,
        "expense_to_income_ratio": 0.65,
        "monthly_surplus_ratio": 0.28,
        "saving_buffer": 5_000_000,
        "bill_on_time_ratio": 0.82,
        "utility_payment_delay": 3,
        "late_payment_count_6m": 2,
        "avg_payment_delay_days": 2.5,
        "wallet_activity_consistency": 0.68,
        "seller_revenue_growth": 0.05,
        "refund_rate": 0.07,
        "order_frequency": 4.5,
        "repeat_customer_ratio": 0.55,
        "rating_avg": 4.3,
        "cancel_rate": 0.10,
        "completion_rate": 0.88,
        "shared_device_count": 1,
        "shared_ip_count": 2,
        "bad_neighbor_count": 1,
        "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.08,
        "trust_score": 0.65,
        "graph_risk_score": 0.32,
        "data_confidence": 0.65,
        "missing_data_ratio": 0.25,
        "source_reliability_score": 0.70,
        "requested_amount": 25_000_000,
        "requested_tenor_months": 18,
    },
}

TYPE_LABELS = {
    "gig_worker":        "Tài xế công nghệ",
    "seller_online":     "Seller online",
    "freelancer":        "Freelancer",
    "small_merchant":    "Tiểu thương",
    "thin_file_customer":"Khách hàng hồ sơ mỏng",
}

# ── Decision metadata ─────────────────────────────────────────────────────────
DECISIONS = {
    "AUTO_REJECT": {
        "label": "Từ chối tự động",
        "emoji": "🚫",
        "color": "#b71c1c",
        "bg":    "#ffebee",
        "desc":  "Vi phạm quy tắc cứng F0 — không thể phê duyệt.",
    },
    "AUTO_APPROVE": {
        "label": "Phê duyệt tự động",
        "emoji": "✅",
        "color": "#1b5e20",
        "bg":    "#e8f5e9",
        "desc":  "Đủ điều kiện toàn diện — phê duyệt tự động với offer tối ưu.",
    },
    "HUMAN_REVIEW": {
        "label": "Chuyển thẩm định",
        "emoji": "🔍",
        "color": "#e65100",
        "bg":    "#fff3e0",
        "desc":  "Vùng xám rủi ro — cần cán bộ tín dụng xem xét thêm.",
    },
    "CREDIT_COACH": {
        "label": "Chưa duyệt — Lộ trình cải thiện",
        "emoji": "📚",
        "color": "#4a148c",
        "bg":    "#f3e5f5",
        "desc":  "Hồ sơ chưa đủ điều kiện — có lộ trình cụ thể để quay lại.",
    },
}


# ── Charts ────────────────────────────────────────────────────────────────────

def _gauge(value, max_val, title, color="#1565c0", steps=None):
    steps = steps or [
        {"range": [0, max_val * 0.45], "color": "#ffcdd2"},
        {"range": [max_val * 0.45, max_val * 0.60], "color": "#fff9c4"},
        {"range": [max_val * 0.60, max_val * 0.80], "color": "#c8e6c9"},
        {"range": [max_val * 0.80, max_val], "color": "#a5d6a7"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis":  {"range": [0, max_val]},
            "bar":   {"color": color},
            "steps": steps,
            "threshold": {
                "line":      {"color": "#c62828", "width": 2},
                "thickness": 0.7,
                "value":     max_val * 0.70,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=5, l=5, r=5))
    return fig


def _bar_subscores(result):
    labels = ["Thu nhập\n(35%)", "Graph\nSafety\n(25%)", "Giao dịch\nsố (20%)",
              "Tài chính\n(15%)", "Nền tảng\n(5%)"]
    values = [
        result.get("income_stability_score", 0),
        result.get("graph_safety_score", 0),
        result.get("digital_transaction_score", 0),
        result.get("financial_commitment_score", 0),
        result.get("platform_behavior_score", 0),
    ]
    colors = ["#1565c0", "#6a1b9a", "#00695c", "#e65100", "#37474f"]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Điểm thành phần (mỗi nhóm 0–100)",
        yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def _ep_chart(offers):
    names   = [o.label for o in offers]
    eps     = [o.expected_profit / 1_000 for o in offers]
    colors  = ["#2e7d32" if o.ep_positive else "#c62828" for o in offers]
    fig = go.Figure(go.Bar(
        x=names, y=eps, marker_color=colors,
        text=[f"{o.expected_profit/1_000:,.0f}K" for o in offers],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        title="Expected Profit từng offer (nghìn đồng)",
        yaxis_title="EP (nghìn đồng)",
        plot_bgcolor="white",
        height=260,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def _gate_badge(status):
    if status == "PASS":
        return "✅ PASS"
    elif status == "WARN":
        return "⚠️ WARN"
    else:
        return "❌ FAIL"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="FlexiScore Demo",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.markdown("""
    <h1 style='text-align:center;color:#1565c0;margin-bottom:2px'>💳 FlexiScore</h1>
    <p style='text-align:center;color:#666;font-size:1em;margin-top:0'>
    Alternative Credit Scoring for Thin-file Customers &nbsp;·&nbsp;
    Risk-first &nbsp;·&nbsp; Cash-flow optimised &nbsp;·&nbsp; Explainable
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    # Load model
    model, metrics = _bootstrap()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Hồ sơ khách hàng")
        st.caption("Chọn ví dụ để nạp giá trị mặc định, sau đó chỉnh trực tiếp bên dưới.")

        case_choice = st.selectbox(
            "Ví dụ minh họa:",
            list(DEMO_CASES.keys()),
            help="Chỉ là điểm khởi đầu — các thông số bên dưới mới quyết định kết quả."
        )
        base = DEMO_CASES[case_choice].copy()

        st.divider()
        st.markdown("**Thông tin cơ bản**")
        base["avg_monthly_income"] = st.number_input(
            "Thu nhập TB tháng (đ)", value=int(base["avg_monthly_income"]),
            min_value=1_000_000, max_value=100_000_000, step=500_000, format="%d")

        base["income_cv"] = st.slider(
            "Biến động thu nhập (income_cv)",
            0.0, 1.0, float(base["income_cv"]), 0.01,
            help="Thấp = ổn định. >0.55 → F1 fail. >0.75 → F0 fail.")

        base["data_confidence"] = st.slider(
            "Độ tin cậy dữ liệu",
            0.0, 1.0, float(base["data_confidence"]), 0.01,
            help="<0.40 → F0 fail. <0.60 → Credit Coach. ≥0.70 cần để Auto Approve.")

        st.divider()
        st.markdown("**Khoản vay**")
        base["requested_amount"] = st.number_input(
            "Số tiền vay (đ)", value=int(base["requested_amount"]),
            min_value=1_000_000, max_value=200_000_000, step=1_000_000, format="%d")

        base["requested_tenor_months"] = st.selectbox(
            "Kỳ hạn (tháng)", [3, 6, 9, 12, 18, 24],
            index=[3, 6, 9, 12, 18, 24].index(int(base["requested_tenor_months"])))

        st.divider()
        st.markdown("**Rủi ro mạng lưới**")
        base["fraud_ring_flag"] = int(st.checkbox(
            "Fraud ring flag", value=bool(base["fraud_ring_flag"]),
            help="True → F0 FAIL → AUTO_REJECT"))

        base["shared_device_count"] = st.slider(
            "Shared device count", 0, 10, int(base["shared_device_count"]), 1,
            help="≥4 → F0 FAIL")

        base["bad_neighbor_count"] = st.slider(
            "Bad neighbor count", 0, 8, int(base["bad_neighbor_count"]), 1,
            help="≥3 → F0 FAIL")

        st.divider()
        st.markdown("**Thanh toán & hành vi**")
        base["bill_on_time_ratio"] = st.slider(
            "Tỷ lệ thanh toán đúng hạn", 0.0, 1.0,
            float(base["bill_on_time_ratio"]), 0.01)

        base["active_days_per_week"] = st.slider(
            "Ngày làm việc/tuần", 0.0, 7.0,
            float(base["active_days_per_week"]), 0.5)

        st.divider()
        run = st.button("Chạy FlexiScore Pipeline", use_container_width=True, type="primary")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run or "result" not in st.session_state or st.session_state.get("last_case") != case_choice:
        from decision_engine import make_decision
        from explainability import generate_reason_codes
        with st.spinner("Đang chạy pipeline..."):
            result  = make_decision(base, model)
            explain = generate_reason_codes(base, result)
        st.session_state["result"]    = result
        st.session_state["explain"]   = explain
        st.session_state["customer"]  = base
        st.session_state["last_case"] = case_choice

    result   = st.session_state["result"]
    explain  = st.session_state["explain"]
    customer = st.session_state["customer"]
    decision = result["decision"]
    dmeta    = DECISIONS.get(decision, DECISIONS["HUMAN_REVIEW"])

    # ── Decision Banner ───────────────────────────────────────────────────────
    d_color = dmeta["color"]
    d_emoji = dmeta["emoji"]
    d_label = dmeta["label"]
    d_desc  = dmeta["desc"]
    st.markdown(
        f"<div style='background:{d_color};color:white;padding:16px 24px;"
        f"border-radius:10px;margin-bottom:20px'>"
        f"<span style='font-size:1.5em'>{d_emoji}</span>&nbsp;&nbsp;"
        f"<strong style='font-size:1.2em'>{d_label}</strong><br>"
        f"<span style='font-size:0.9em;opacity:0.9'>{d_desc}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 4-metric strip ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FlexiScore", f"{result['flexiscore']:.0f} / 1000")
    c2.metric("PD",  f"{result['pd']:.1%}", delta=f"{result['pd_band']}", delta_color="off")
    c3.metric("F0", _gate_badge(result["f0_status"]))
    c4.metric("F1", _gate_badge(result["f1_status"]))

    st.divider()

    # ── Two-column main layout ────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ─────────────────────────── CỘT TRÁI: Customer View ────────────────────
    with left:
        st.subheader("👤 Customer View")

        ctype_label = TYPE_LABELS.get(result["customer_type"], result["customer_type"])
        st.markdown(f"""
| | |
|--|--|
| **Tên** | {result['name']} |
| **Loại KH** | {ctype_label} |
| **Thu nhập TB** | {customer['avg_monthly_income']:,.0f} đ/tháng |
| **Yêu cầu vay** | {customer['requested_amount']:,.0f} đ / {customer['requested_tenor_months']} tháng |
| **Risk tier** | {result['risk_tier']} |
""")
        st.divider()

        # Decision-specific customer message
        if decision == "AUTO_APPROVE":
            st.success("🎉 Chúc mừng! Khoản vay của bạn được phê duyệt tự động.")
            rec = result.get("recommended_offer")
            if rec:
                st.markdown(f"""
**Offer được đề xuất — {rec.label}**

| | |
|--|--|
| Số tiền giải ngân | **{rec.amount:,.0f} đ** |
| Kỳ hạn | **{rec.tenor_months} tháng** |
| Lịch trả | **{rec.schedule}** |
| Lãi suất | **{rec.interest_rate:.0%} / năm** |
| Góp ước tính | **{rec.monthly_payment:,.0f} đ / lần** |
| Expected Profit (ngân hàng) | **+{rec.expected_profit:,.0f} đ** |
""")

        elif decision == "AUTO_REJECT":
            st.error("🚫 Hồ sơ không đủ điều kiện xử lý.")
            st.markdown(
                "Kết quả dựa trên phân tích tự động. "
                "Bạn có quyền yêu cầu cán bộ tín dụng xem xét lại."
            )
            st.markdown("**Lý do cụ thể:**")
            for r in result["reason_codes"]:
                st.markdown(f"- {r}")

        elif decision == "CREDIT_COACH":
            st.warning("📚 Hồ sơ chưa đủ điều kiện — nhưng bạn có thể cải thiện!")
            plan = result.get("credit_coach_plan", [])
            if plan:
                st.markdown(f"**{plan[0]}**")
                for item in plan[1:]:
                    st.markdown(f"- {item}")

        elif decision == "HUMAN_REVIEW":
            st.info(
                "🔍 Hồ sơ đang được chuyển sang thẩm định thêm từ cán bộ tín dụng.\n\n"
                "Bạn sẽ nhận được phản hồi trong vòng 1–2 ngày làm việc."
            )
            rec = result.get("recommended_offer")
            if rec and rec.ep_positive:
                st.markdown(
                    f"Cán bộ tín dụng sẽ xem xét offer tiềm năng: "
                    f"**{rec.amount/1e6:.1f}M / {rec.tenor_months}T**."
                )

        st.divider()
        st.markdown("**Điểm mạnh hồ sơ**")
        pos = explain.get("positive_factors", [])
        for f in pos[:6]:
            st.markdown(f)
        if explain.get("negative_factors"):
            st.markdown("**Cần cải thiện**")
            for f in explain["negative_factors"][:6]:
                st.markdown(f)

    # ─────────────────────────── CỘT PHẢI: Bank/Risk View ───────────────────
    with right:
        st.subheader("🏦 Bank / Risk View")

        # FlexiScore gauge
        st.plotly_chart(
            _gauge(result["flexiscore"], 1000, "FlexiScore (0–1000)", color="#1565c0"),
            use_container_width=True,
        )

        # Graph risk gauge
        gr_color = (
            "#c62828" if result["graph_risk_label"] == "HIGH"
            else ("#f57c00" if result["graph_risk_label"] == "MEDIUM" else "#2e7d32")
        )
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.plotly_chart(
                _gauge(
                    result["graph_risk_score"] * 100, 100,
                    "Graph Risk (%)",
                    color=gr_color,
                    steps=[
                        {"range": [0, 40],  "color": "#e8f5e9"},
                        {"range": [40, 70], "color": "#fff8e1"},
                        {"range": [70, 100],"color": "#ffebee"},
                    ],
                ),
                use_container_width=True,
            )
        with gcol2:
            st.plotly_chart(
                _gauge(
                    result["trust_score"] * 100, 100,
                    "Trust Score (%)",
                    color="#00695c",
                    steps=[
                        {"range": [0, 40],  "color": "#ffebee"},
                        {"range": [40, 70], "color": "#fff8e1"},
                        {"range": [70, 100],"color": "#e8f5e9"},
                    ],
                ),
                use_container_width=True,
            )

        # Graph risk detail
        with st.expander("Chi tiết Graph Risk"):
            for r in result.get("graph_reasons", []):
                st.markdown(f"- {r}")

        # Subscores
        st.plotly_chart(_bar_subscores(result), use_container_width=True)

        # Offers table + EP chart
        offers = result.get("offers", [])
        if offers:
            st.markdown("**So sánh Loan Offers**")
            rec_label = result["recommended_offer"].label if result["recommended_offer"] else ""
            rows = []
            for o in offers:
                star = " ⭐" if o.label == rec_label else ""
                rows.append({
                    "Offer": o.label + star,
                    "Số tiền": f"{o.amount/1e6:.1f}M",
                    "Kỳ hạn": f"{o.tenor_months}T",
                    "Lịch": o.schedule,
                    "EP": f"{o.expected_profit/1_000:,.0f}K đ",
                    "OK?": "✅" if o.ep_positive else "❌",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.plotly_chart(_ep_chart(offers), use_container_width=True)

    # ── Reason codes & gate details (full width) ──────────────────────────────
    st.divider()
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        with st.expander("📋 Reason Codes — Bank View"):
            for rc in result.get("reason_codes", []):
                st.markdown(f"- {rc}")

    with exp_col2:
        with st.expander("🛡 F0 — Hard Rule Detail"):
            f0r = result.get("f0_reasons", [])
            if f0r:
                for r in f0r:
                    st.markdown(f"- {r}")
            else:
                st.success("Không có vi phạm F0.")

    with exp_col3:
        with st.expander("⚡ F1 — Stress Test Detail"):
            f1r = result.get("f1_reasons", [])
            dti = result.get("dti_stress_ratio", 0)
            st.markdown(f"**DTI stress:** {dti:.1%}  *(ngưỡng 35%)*")
            if f1r:
                for r in f1r:
                    st.markdown(f"- {r}")
            else:
                st.success("Không có vi phạm F1.")

    # ── Model info ────────────────────────────────────────────────────────────
    with st.expander("📈 Model Info"):
        info_cols = st.columns(4)
        if metrics:
            info_cols[0].metric("ROC-AUC",   metrics.get("roc_auc",   "—"))
            info_cols[1].metric("Accuracy",  metrics.get("accuracy",  "—"))
            info_cols[2].metric("Precision", metrics.get("precision", "—"))
            info_cols[3].metric("Recall",    metrics.get("recall",    "—"))
        else:
            st.info("Model đã load từ cache.")

        st.caption(
            f"Auto Approve ngưỡng: FlexiScore ≥ 700  |  PD < 15%  |  "
            f"Data confidence ≥ 70%  |  EP > 0"
        )

    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:0.78em;margin-top:20px'>"
        "FlexiScore Demo — Dữ liệu giả lập, không phải dữ liệu thật của SHB.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
