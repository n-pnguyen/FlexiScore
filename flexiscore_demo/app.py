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


# ── Demo cases (chỉ là điểm khởi đầu — controls trong sidebar mới quyết định) ─
DEMO_CASES = {
    "Happy Path — Nguyễn Văn A": {
        "customer_id": "DEMO_001", "name": "Nguyễn Văn A",
        "customer_type": "gig_worker", "age": 32,
        "platform_tenure_months": 18, "data_months_available": 18,
        "identity_verified": 1,
        "avg_monthly_income": 15_000_000, "income_cv": 0.15,
        "cashflow_drop_30d": 0.00, "active_days_per_week": 6.0,
        "monthly_inflow_count": 48, "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40, "saving_buffer": 8_000_000,
        "bill_on_time_ratio": 0.98, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0.0,
        "wallet_activity_consistency": 0.92,
        "seller_revenue_growth": 0.0, "refund_rate": 0.0,
        "order_frequency": 6.0, "repeat_customer_ratio": 0.75,
        "rating_avg": 4.8, "cancel_rate": 0.03, "completion_rate": 0.97,
        "shared_device_count": 0, "shared_ip_count": 0,
        "bad_neighbor_count": 0, "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.02,
        "trust_score": 0.92, "graph_risk_score": 0.08,
        "data_confidence": 0.88, "missing_data_ratio": 0.08,
        "source_reliability_score": 0.90,
        "requested_amount": 20_000_000, "requested_tenor_months": 12,
    },
    "Risk-First — Trần Thị B": {
        "customer_id": "DEMO_002", "name": "Trần Thị B",
        "customer_type": "seller_online", "age": 28,
        "platform_tenure_months": 24, "data_months_available": 20,
        "identity_verified": 1,
        "avg_monthly_income": 25_000_000, "income_cv": 0.20,
        "cashflow_drop_30d": 0.05, "active_days_per_week": 6.0,
        "monthly_inflow_count": 120, "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40, "saving_buffer": 15_000_000,
        "bill_on_time_ratio": 0.96, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0.0,
        "wallet_activity_consistency": 0.88,
        "seller_revenue_growth": 0.18, "refund_rate": 0.08,
        "order_frequency": 8.0, "repeat_customer_ratio": 0.65,
        "rating_avg": 4.7, "cancel_rate": 0.05, "completion_rate": 0.96,
        "shared_device_count": 4, "shared_ip_count": 6,
        "bad_neighbor_count": 3, "fraud_ring_flag": 1,
        "circular_transaction_ratio": 0.28,
        "trust_score": 0.28, "graph_risk_score": 0.91,
        "data_confidence": 0.80, "missing_data_ratio": 0.20,
        "source_reliability_score": 0.75,
        "requested_amount": 40_000_000, "requested_tenor_months": 24,
    },
    "Credit Coach — Lê Văn C": {
        "customer_id": "DEMO_003", "name": "Lê Văn C",
        "customer_type": "freelancer", "age": 26,
        "platform_tenure_months": 8, "data_months_available": 6,
        "identity_verified": 1,
        "avg_monthly_income": 12_000_000, "income_cv": 0.65,
        "cashflow_drop_30d": 0.42, "active_days_per_week": 3.5,
        "monthly_inflow_count": 8, "expense_to_income_ratio": 0.72,
        "monthly_surplus_ratio": 0.20, "saving_buffer": 2_000_000,
        "bill_on_time_ratio": 0.70, "utility_payment_delay": 2,
        "late_payment_count_6m": 3, "avg_payment_delay_days": 4.5,
        "wallet_activity_consistency": 0.55,
        "seller_revenue_growth": 0.0, "refund_rate": 0.0,
        "order_frequency": 3.0, "repeat_customer_ratio": 0.40,
        "rating_avg": 4.2, "cancel_rate": 0.15, "completion_rate": 0.82,
        "shared_device_count": 0, "shared_ip_count": 1,
        "bad_neighbor_count": 0, "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.03,
        "trust_score": 0.75, "graph_risk_score": 0.20,
        "data_confidence": 0.55, "missing_data_ratio": 0.38,
        "source_reliability_score": 0.60,
        "requested_amount": 15_000_000, "requested_tenor_months": 12,
    },
    "Human Review — Nguyễn Thị D": {
        "customer_id": "DEMO_004", "name": "Nguyễn Thị D",
        "customer_type": "small_merchant", "age": 35,
        "platform_tenure_months": 10, "data_months_available": 9,
        "identity_verified": 1,
        "avg_monthly_income": 18_000_000, "income_cv": 0.38,
        "cashflow_drop_30d": 0.18, "active_days_per_week": 5.0,
        "monthly_inflow_count": 30, "expense_to_income_ratio": 0.65,
        "monthly_surplus_ratio": 0.28, "saving_buffer": 5_000_000,
        "bill_on_time_ratio": 0.82, "utility_payment_delay": 3,
        "late_payment_count_6m": 2, "avg_payment_delay_days": 2.5,
        "wallet_activity_consistency": 0.68,
        "seller_revenue_growth": 0.05, "refund_rate": 0.07,
        "order_frequency": 4.5, "repeat_customer_ratio": 0.55,
        "rating_avg": 4.3, "cancel_rate": 0.10, "completion_rate": 0.88,
        "shared_device_count": 1, "shared_ip_count": 2,
        "bad_neighbor_count": 1, "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.08,
        "trust_score": 0.65, "graph_risk_score": 0.32,
        "data_confidence": 0.65, "missing_data_ratio": 0.25,
        "source_reliability_score": 0.70,
        "requested_amount": 25_000_000, "requested_tenor_months": 18,
    },
}

TYPE_OPTS   = ["gig_worker", "seller_online", "freelancer",
               "small_merchant", "thin_file_customer"]
TYPE_LABELS = {
    "gig_worker":         "Tài xế công nghệ",
    "seller_online":      "Seller online",
    "freelancer":         "Freelancer",
    "small_merchant":     "Tiểu thương",
    "thin_file_customer": "Khách hàng hồ sơ mỏng",
}
TENOR_OPTS = [3, 6, 9, 12, 18, 24]

DECISIONS = {
    "AUTO_REJECT": {
        "label": "Từ chối tự động",
        "emoji": "🚫",
        "color": "#b71c1c",
        "desc":  "Vi phạm quy tắc cứng F0 — không thể phê duyệt.",
    },
    "AUTO_APPROVE": {
        "label": "Phê duyệt tự động",
        "emoji": "✅",
        "color": "#1b5e20",
        "desc":  "Đủ điều kiện toàn diện — phê duyệt tự động với offer tối ưu.",
    },
    "HUMAN_REVIEW": {
        "label": "Chuyển thẩm định",
        "emoji": "🔍",
        "color": "#e65100",
        "desc":  "Vùng xám rủi ro — cần cán bộ tín dụng xem xét thêm.",
    },
    "CREDIT_COACH": {
        "label": "Chưa duyệt — Lộ trình cải thiện",
        "emoji": "📚",
        "color": "#4a148c",
        "desc":  "Hồ sơ chưa đủ điều kiện — có lộ trình cụ thể để quay lại.",
    },
}


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _gauge(value, max_val, title, color="#1565c0", steps=None):
    steps = steps or [
        {"range": [0,              max_val * 0.45], "color": "#ffcdd2"},
        {"range": [max_val * 0.45, max_val * 0.60], "color": "#fff9c4"},
        {"range": [max_val * 0.60, max_val * 0.80], "color": "#c8e6c9"},
        {"range": [max_val * 0.80, max_val],         "color": "#a5d6a7"},
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
    labels = ["Thu nhập\n(35%)", "Graph\nSafety\n(25%)",
              "Giao dịch\nsố (20%)", "Tài chính\n(15%)", "Nền tảng\n(5%)"]
    values = [
        result.get("income_stability_score",    0),
        result.get("graph_safety_score",        0),
        result.get("digital_transaction_score", 0),
        result.get("financial_commitment_score",0),
        result.get("platform_behavior_score",   0),
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
        plot_bgcolor="white", height=280,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def _ep_chart(offers):
    names  = [o.label for o in offers]
    eps    = [o.expected_profit / 1_000 for o in offers]
    colors = ["#2e7d32" if o.ep_positive else "#c62828" for o in offers]
    fig = go.Figure(go.Bar(
        x=names, y=eps, marker_color=colors,
        text=[f"{o.expected_profit/1_000:,.0f}K" for o in offers],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        title="Expected Profit từng offer (nghìn đồng)",
        yaxis_title="EP (nghìn đồng)", plot_bgcolor="white",
        height=260, margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def _gate_badge(status):
    if status == "PASS": return "✅ PASS"
    if status == "WARN": return "⚠️ WARN"
    return "❌ FAIL"


# ── Sidebar — tất cả tham số ──────────────────────────────────────────────────

def _build_sidebar(base: dict) -> dict:
    """Render toàn bộ controls trong sidebar, trả về dict customer đã chỉnh."""
    p = base.copy()

    with st.sidebar:
        st.markdown("### Hồ sơ khách hàng")
        st.caption(
            "Chọn ví dụ để nạp giá trị mặc định, "
            "sau đó chỉnh trực tiếp bên dưới."
        )

        case_name = st.selectbox(
            "Ví dụ minh họa:",
            list(DEMO_CASES.keys()),
            help="Chỉ là điểm khởi đầu — các thông số bên dưới mới quyết định kết quả.",
            key="case_select",
        )

        st.divider()

        # ── 1. Hồ sơ cơ bản ──────────────────────────────────────────────────
        with st.expander("👤 Hồ sơ cơ bản", expanded=True):
            p["name"] = st.text_input("Tên khách hàng", value=str(p.get("name", "")))

            ctype_idx = TYPE_OPTS.index(p.get("customer_type", "gig_worker"))
            chosen    = st.selectbox(
                "Loại khách hàng",
                options=TYPE_OPTS,
                format_func=lambda x: TYPE_LABELS[x],
                index=ctype_idx,
            )
            p["customer_type"] = chosen

            p["age"] = st.slider(
                "Tuổi", 18, 65, int(p.get("age", 30)), 1,
            )
            p["identity_verified"] = int(st.checkbox(
                "Đã xác thực eKYC / định danh",
                value=bool(p.get("identity_verified", 1)),
            ))
            p["platform_tenure_months"] = st.slider(
                "Thâm niên nền tảng (tháng)",
                0, 60, int(p.get("platform_tenure_months", 12)), 1,
                help="Tốt: >6 tháng. Rủi ro: <4 tháng.",
            )
            p["data_months_available"] = st.slider(
                "Số tháng dữ liệu sẵn có",
                0, 60, int(p.get("data_months_available", 12)), 1,
                help="Tốt: >=6. Không auto approve nếu <3.",
            )

        # ── 2. Thu nhập & Dòng tiền ───────────────────────────────────────────
        with st.expander("💰 Thu nhập & Dòng tiền", expanded=True):
            p["avg_monthly_income"] = st.number_input(
                "Thu nhập TB tháng (đ)",
                min_value=1_000_000, max_value=200_000_000,
                value=int(p.get("avg_monthly_income", 10_000_000)),
                step=500_000, format="%d",
            )
            p["income_cv"] = st.slider(
                "Biến động thu nhập (income_cv)",
                0.0, 1.0, float(p.get("income_cv", 0.3)), 0.01,
                help="Thấp = ổn định tốt. >0.55 → F1 fail. >0.75 → F0 fail.",
            )
            p["cashflow_drop_30d"] = st.slider(
                "Sụt giảm dòng tiền 30 ngày",
                0.0, 0.9, float(p.get("cashflow_drop_30d", 0.0)), 0.01,
                help="Cảnh báo: >0.30. Ảnh hưởng FlexiScore thu nhập.",
            )
            p["active_days_per_week"] = st.slider(
                "Ngày tạo thu nhập / tuần",
                0.0, 7.0, float(p.get("active_days_per_week", 5.0)), 0.5,
                help="Gig tốt: >=5 ngày/tuần.",
            )
            p["monthly_inflow_count"] = st.slider(
                "Số giao dịch tiền vào / tháng",
                1, 200, int(p.get("monthly_inflow_count", 20)), 1,
                help="Tần suất cao = dòng tiền số chứng minh được.",
            )
            p["expense_to_income_ratio"] = st.slider(
                "Tỷ lệ chi tiêu / thu nhập",
                0.1, 1.0, float(p.get("expense_to_income_ratio", 0.6)), 0.01,
                help="Tốt: <0.60. Review: >0.75.",
            )
            p["monthly_surplus_ratio"] = st.slider(
                "Tỷ lệ dòng tiền dư tháng",
                0.0, 0.9, float(p.get("monthly_surplus_ratio", 0.3)), 0.01,
                help="Tốt: >0.25.",
            )
            p["saving_buffer"] = st.number_input(
                "Dự phòng / số dư buffer (đ)",
                min_value=0, max_value=500_000_000,
                value=int(p.get("saving_buffer", 3_000_000)),
                step=500_000, format="%d",
                help="Càng cao càng an toàn.",
            )

        # ── 3. Kỷ luật thanh toán ─────────────────────────────────────────────
        with st.expander("📋 Kỷ luật thanh toán"):
            p["bill_on_time_ratio"] = st.slider(
                "Tỷ lệ thanh toán đúng hạn",
                0.0, 1.0, float(p.get("bill_on_time_ratio", 0.9)), 0.01,
                help="Tốt: >0.90.",
            )
            p["utility_payment_delay"] = st.slider(
                "Số ngày trễ hóa đơn điện/nước",
                0, 30, int(p.get("utility_payment_delay", 0)), 1,
                help="Tốt: 0–2. Rủi ro: >7.",
            )
            p["late_payment_count_6m"] = st.slider(
                "Số lần thanh toán trễ (6 tháng)",
                0, 12, int(p.get("late_payment_count_6m", 0)), 1,
                help="Tốt: 0–1. Rủi ro: >3.",
            )
            p["avg_payment_delay_days"] = st.slider(
                "Số ngày trễ trung bình",
                0.0, 30.0, float(p.get("avg_payment_delay_days", 0.0)), 0.5,
                help="Tốt: <3 ngày.",
            )
            p["wallet_activity_consistency"] = st.slider(
                "Độ đều hoạt động ví / ngân hàng số",
                0.0, 1.0, float(p.get("wallet_activity_consistency", 0.7)), 0.01,
                help="Tốt: >0.75.",
            )

        # ── 4. Hành vi nền tảng & kinh doanh ─────────────────────────────────
        with st.expander("🛒 Hành vi nền tảng & kinh doanh"):
            p["rating_avg"] = st.slider(
                "Đánh giá trung bình nền tảng",
                1.0, 5.0, float(p.get("rating_avg", 4.5)), 0.1,
                help="Tốt: >4.5.",
            )
            p["cancel_rate"] = st.slider(
                "Tỷ lệ hủy đơn / cuốc",
                0.0, 0.5, float(p.get("cancel_rate", 0.05)), 0.01,
                help="Rủi ro: >0.20.",
            )
            p["completion_rate"] = st.slider(
                "Tỷ lệ hoàn thành đơn / cuốc",
                0.0, 1.0, float(p.get("completion_rate", 0.95)), 0.01,
                help="Tốt: >0.90.",
            )
            p["order_frequency"] = st.slider(
                "Tần suất đơn hàng / chuyến (ngày)",
                0.0, 15.0, float(p.get("order_frequency", 4.0)), 0.5,
                help="Ổn định / cao là tốt.",
            )
            p["repeat_customer_ratio"] = st.slider(
                "Tỷ lệ khách hàng quay lại",
                0.0, 1.0, float(p.get("repeat_customer_ratio", 0.5)), 0.01,
                help="Càng cao càng tốt.",
            )
            p["seller_revenue_growth"] = st.slider(
                "Tăng trưởng doanh thu (seller)",
                -0.5, 1.0, float(p.get("seller_revenue_growth", 0.0)), 0.01,
                help="Ổn định / dương là tốt.",
            )
            p["refund_rate"] = st.slider(
                "Tỷ lệ hoàn / hủy đơn (seller)",
                0.0, 0.5, float(p.get("refund_rate", 0.0)), 0.01,
                help="Tốt: <0.10. Rủi ro: >0.25.",
            )

        # ── 5. Rủi ro mạng lưới (Graph) ───────────────────────────────────────
        with st.expander("🕸 Rủi ro mạng lưới / Graph", expanded=True):
            p["fraud_ring_flag"] = int(st.checkbox(
                "Fraud ring flag (cụm gian lận)",
                value=bool(p.get("fraud_ring_flag", 0)),
                help="True → F0 FAIL → AUTO_REJECT ngay.",
            ))
            p["shared_device_count"] = st.slider(
                "Số hồ sơ dùng chung thiết bị",
                0, 10, int(p.get("shared_device_count", 0)), 1,
                help="Fraud review: >=4 → F0 FAIL.",
            )
            p["shared_ip_count"] = st.slider(
                "Số hồ sơ dùng chung IP",
                0, 15, int(p.get("shared_ip_count", 0)), 1,
                help="Cảnh báo khi cao.",
            )
            p["bad_neighbor_count"] = st.slider(
                "Số liên kết tới hồ sơ nợ xấu",
                0, 8, int(p.get("bad_neighbor_count", 0)), 1,
                help="Fraud review: >=3 → F0 FAIL.",
            )
            p["circular_transaction_ratio"] = st.slider(
                "Tỷ lệ giao dịch vòng",
                0.0, 0.9, float(p.get("circular_transaction_ratio", 0.02)), 0.01,
                help="Cảnh báo: >0.20 (đảo tiền).",
            )
            p["trust_score"] = st.slider(
                "Điểm tin cậy mạng lưới (trust_score)",
                0.0, 1.0, float(p.get("trust_score", 0.7)), 0.01,
                help="Càng cao càng tốt.",
            )
            p["graph_risk_score"] = st.slider(
                "Điểm rủi ro graph",
                0.0, 1.0, float(p.get("graph_risk_score", 0.2)), 0.01,
                help=">0.70 → Fraud Review.",
            )

        # ── 6. Độ tin cậy dữ liệu ─────────────────────────────────────────────
        with st.expander("📊 Độ tin cậy dữ liệu", expanded=True):
            p["data_confidence"] = st.slider(
                "Độ tin cậy dữ liệu (data_confidence)",
                0.0, 1.0, float(p.get("data_confidence", 0.7)), 0.01,
                help="<0.40 → F0 FAIL. <0.60 → CREDIT_COACH. >=0.70 cần để AUTO_APPROVE.",
            )
            p["missing_data_ratio"] = st.slider(
                "Tỷ lệ dữ liệu thiếu",
                0.0, 0.9, float(p.get("missing_data_ratio", 0.1)), 0.01,
                help="Càng thấp càng tốt.",
            )
            p["source_reliability_score"] = st.slider(
                "Độ tin cậy nguồn dữ liệu",
                0.0, 1.0, float(p.get("source_reliability_score", 0.8)), 0.01,
                help="Càng cao càng tốt.",
            )

        # ── 7. Khoản vay ──────────────────────────────────────────────────────
        with st.expander("🏦 Yêu cầu khoản vay", expanded=True):
            p["requested_amount"] = st.number_input(
                "Số tiền vay (đ)",
                min_value=1_000_000, max_value=500_000_000,
                value=int(p.get("requested_amount", 20_000_000)),
                step=1_000_000, format="%d",
            )
            tenor_val = int(p.get("requested_tenor_months", 12))
            if tenor_val not in TENOR_OPTS:
                tenor_val = 12
            p["requested_tenor_months"] = st.selectbox(
                "Kỳ hạn vay (tháng)", TENOR_OPTS,
                index=TENOR_OPTS.index(tenor_val),
            )

        st.divider()
        run = st.button(
            "Chạy FlexiScore Pipeline",
            use_container_width=True,
            type="primary",
        )

    return case_name, p, run


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="FlexiScore Demo",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.markdown(
        "<h1 style='text-align:center;color:#1565c0;margin-bottom:2px'>💳 FlexiScore</h1>"
        "<p style='text-align:center;color:#666;font-size:1em;margin-top:0'>"
        "Alternative Credit Scoring for Thin-file Customers &nbsp;·&nbsp;"
        "Risk-first &nbsp;·&nbsp; Cash-flow optimised &nbsp;·&nbsp; Explainable"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Load model
    model, metrics = _bootstrap()

    # Lấy base từ session hoặc default case đầu tiên
    default_case = list(DEMO_CASES.keys())[0]
    if "last_case" not in st.session_state:
        st.session_state["last_case"] = default_case

    # Xác định base cho sidebar: nếu case thay đổi thì reset về demo preset
    # (việc này được xử lý bằng cách pass DEMO_CASES[case] vào _build_sidebar)
    # Ta cần biết case trước khi render sidebar nên dùng query params trick:
    # Streamlit re-renders từ top → sidebar selectbox quyết định case_name,
    # ta lấy giá trị hiện tại từ session nếu có.
    prev_case = st.session_state.get("last_case", default_case)
    base = DEMO_CASES.get(prev_case, DEMO_CASES[default_case]).copy()

    # Render sidebar — trả về case_name đã chọn, customer dict, nút run
    case_name, customer, run_clicked = _build_sidebar(base)

    # Nếu case thay đổi → tự động re-run với preset mới
    case_changed = (case_name != st.session_state.get("last_case"))
    if case_changed:
        # Reset lại base sang preset mới rồi rerun để sidebar load đúng giá trị
        st.session_state["last_case"] = case_name
        st.rerun()

    # ── Chạy pipeline ─────────────────────────────────────────────────────────
    need_run = run_clicked or ("result" not in st.session_state)
    if need_run:
        from decision_engine import make_decision
        from explainability import generate_reason_codes
        with st.spinner("Đang chạy pipeline..."):
            result  = make_decision(customer, model)
            explain = generate_reason_codes(customer, result)
        st.session_state["result"]   = result
        st.session_state["explain"]  = explain
        st.session_state["customer"] = customer
    else:
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
        f"border-radius:10px;margin-bottom:18px'>"
        f"<span style='font-size:1.5em'>{d_emoji}</span>&nbsp;&nbsp;"
        f"<strong style='font-size:1.2em'>{d_label}</strong><br>"
        f"<span style='font-size:0.9em;opacity:0.9'>{d_desc}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 4-metric strip ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FlexiScore", f"{result['flexiscore']:.0f} / 1000",
              delta=result["risk_tier"], delta_color="off")
    c2.metric("PD", f"{result['pd']:.1%}",
              delta=result["pd_band"], delta_color="off")
    c3.metric("F0 Hard Rules", _gate_badge(result["f0_status"]))
    c4.metric("F1 Stress Test", _gate_badge(result["f1_status"]))

    st.divider()

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ───────────────── CỘT TRÁI: Customer View ───────────────────────────────
    with left:
        st.subheader("👤 Customer View")

        ctype_label = TYPE_LABELS.get(result["customer_type"], result["customer_type"])
        st.markdown(
            f"| | |\n|--|--|\n"
            f"| **Tên** | {result['name']} |\n"
            f"| **Loại KH** | {ctype_label} |\n"
            f"| **Thu nhập TB** | {customer['avg_monthly_income']:,.0f} đ / tháng |\n"
            f"| **Yêu cầu vay** | {customer['requested_amount']:,.0f} đ / "
            f"{customer['requested_tenor_months']} tháng |\n"
            f"| **Risk tier** | {result['risk_tier']} |"
        )
        st.divider()

        if decision == "AUTO_APPROVE":
            st.success("Chúc mừng! Khoản vay được phê duyệt tự động.")
            rec = result.get("recommended_offer")
            if rec:
                st.markdown(f"**Offer đề xuất — {rec.label}**")
                rows = {
                    "Số tiền giải ngân": f"{rec.amount:,.0f} đ",
                    "Kỳ hạn": f"{rec.tenor_months} tháng",
                    "Lịch trả": rec.schedule,
                    "Lãi suất": f"{rec.interest_rate:.0%} / năm",
                    "Góp ước tính": f"{rec.monthly_payment:,.0f} đ / lần",
                    "Expected Profit (NH)": f"+{rec.expected_profit:,.0f} đ",
                }
                st.table(pd.DataFrame(rows.items(), columns=["", "Giá trị"]).set_index(""))

        elif decision == "AUTO_REJECT":
            st.error("Hồ sơ không đủ điều kiện xử lý.")
            st.markdown("Bạn có quyền yêu cầu cán bộ tín dụng xem xét lại.")
            st.markdown("**Lý do:**")
            for r in result["reason_codes"]:
                st.markdown(f"- {r}")

        elif decision == "CREDIT_COACH":
            st.warning("Hồ sơ chưa đủ điều kiện — nhưng bạn có thể cải thiện!")
            plan = result.get("credit_coach_plan", [])
            if plan:
                st.markdown(f"**{plan[0]}**")
                for item in plan[1:]:
                    st.markdown(f"- {item}")

        elif decision == "HUMAN_REVIEW":
            st.info(
                "Hồ sơ đang được chuyển sang thẩm định thêm từ cán bộ tín dụng.\n\n"
                "Phản hồi dự kiến trong 1–2 ngày làm việc."
            )
            rec = result.get("recommended_offer")
            if rec and rec.ep_positive:
                st.markdown(
                    f"Cán bộ tín dụng sẽ xem xét offer tiềm năng: "
                    f"**{rec.amount/1e6:.1f}M / {rec.tenor_months}T**."
                )

        st.divider()
        pos = explain.get("positive_factors", [])
        neg = explain.get("negative_factors", [])
        if pos:
            st.markdown("**Điểm mạnh hồ sơ**")
            for f in pos[:8]:
                st.markdown(f)
        if neg:
            st.markdown("**Cần cải thiện**")
            for f in neg[:8]:
                st.markdown(f)

    # ───────────────── CỘT PHẢI: Bank / Risk View ────────────────────────────
    with right:
        st.subheader("🏦 Bank / Risk View")

        # FlexiScore gauge
        st.plotly_chart(
            _gauge(result["flexiscore"], 1000, "FlexiScore (0–1000)",
                   color="#1565c0"),
            use_container_width=True,
        )

        # Graph Risk + Trust Score side by side
        gr_label = result.get("graph_risk_label", "LOW")
        gr_color = ("#c62828" if gr_label == "HIGH"
                    else ("#f57c00" if gr_label == "MEDIUM" else "#2e7d32"))
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.plotly_chart(
                _gauge(
                    result["graph_risk_score"] * 100, 100, "Graph Risk (%)",
                    color=gr_color,
                    steps=[
                        {"range": [0,  40], "color": "#e8f5e9"},
                        {"range": [40, 70], "color": "#fff8e1"},
                        {"range": [70,100], "color": "#ffebee"},
                    ],
                ),
                use_container_width=True,
            )
        with gcol2:
            st.plotly_chart(
                _gauge(
                    result["trust_score"] * 100, 100, "Trust Score (%)",
                    color="#00695c",
                    steps=[
                        {"range": [0,  40], "color": "#ffebee"},
                        {"range": [40, 70], "color": "#fff8e1"},
                        {"range": [70,100], "color": "#e8f5e9"},
                    ],
                ),
                use_container_width=True,
            )

        with st.expander("Chi tiết Graph Risk"):
            for r in result.get("graph_reasons", []):
                st.markdown(f"- {r}")

        # Sub-scores bar chart
        st.plotly_chart(_bar_subscores(result), use_container_width=True)

        # Offers
        offers = result.get("offers", [])
        if offers:
            st.markdown("**So sánh Loan Offers**")
            rec_label = result["recommended_offer"].label if result["recommended_offer"] else ""
            rows = []
            for o in offers:
                star = " ⭐" if o.label == rec_label else ""
                rows.append({
                    "Offer":    o.label + star,
                    "Số tiền": f"{o.amount/1e6:.1f}M",
                    "Kỳ hạn":  f"{o.tenor_months}T",
                    "Lịch":    o.schedule,
                    "EP":      f"{o.expected_profit/1_000:,.0f}K đ",
                    "OK?":     "✅" if o.ep_positive else "❌",
                })
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True,
            )
            st.plotly_chart(_ep_chart(offers), use_container_width=True)

    # ── Bottom — Reason codes + Gate details + Model info ────────────────────
    st.divider()
    b1, b2, b3 = st.columns(3)

    with b1:
        with st.expander("📋 Reason Codes — Bank View"):
            for rc in result.get("reason_codes", []):
                st.markdown(f"- {rc}")

    with b2:
        with st.expander("🛡 F0 Hard Rules — Chi tiết"):
            f0r = result.get("f0_reasons", [])
            if f0r:
                for r in f0r:
                    st.markdown(f"- {r}")
            else:
                st.success("Không có vi phạm F0.")

    with b3:
        with st.expander("⚡ F1 Stress Test — Chi tiết"):
            dti = result.get("dti_stress_ratio", 0)
            st.markdown(f"**DTI stress:** {dti:.1%}  *(ngưỡng 35%)*")
            f1r = result.get("f1_reasons", [])
            if f1r:
                for r in f1r:
                    st.markdown(f"- {r}")
            else:
                st.success("Không có vi phạm F1.")

    with st.expander("📈 Model Info & Thresholds"):
        mi = st.columns(4)
        if metrics:
            mi[0].metric("ROC-AUC",   metrics.get("roc_auc",   "—"))
            mi[1].metric("Accuracy",  metrics.get("accuracy",  "—"))
            mi[2].metric("Precision", metrics.get("precision", "—"))
            mi[3].metric("Recall",    metrics.get("recall",    "—"))
        else:
            st.info("Model load từ cache — chạy scoring_model.py để xem metrics.")
        st.caption(
            "AUTO_APPROVE: FlexiScore >= 700  |  PD < 15%  |  "
            "data_confidence >= 70%  |  EP > 0  ·  "
            "CREDIT_COACH: F1 fail (income_cv>0.55 hoặc DTI>35%)  |  data_confidence < 60%  ·  "
            "AUTO_REJECT: F0 fail"
        )

    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:0.78em;margin-top:16px'>"
        "FlexiScore Demo — Dữ liệu giả lập, không phải dữ liệu thật của SHB.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
