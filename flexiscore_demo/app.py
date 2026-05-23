"""
app.py — FlexiScore Demo Dashboard (Streamlit).
Chạy: streamlit run app.py

Kiến trúc dữ liệu:
  Zone A — RAW INPUTS (nhập tay): customer_type, age, eKYC, loan request
  Zone B — FEATURE-ENGINEERED (auto-computed từ data sources, demo cho override):
    B1: Grab/Shopee API     → platform metrics, active_days, ratings
    B2: VietQR / Bank txns  → income, cashflow, expense ratios
    B3: EVN / VNPT bills    → payment discipline
    B4: Neo4j Graph DB      → fraud, device sharing, trust score
    B5: Pipeline auto-calc  → data_confidence (từ B1–B4)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi tạo FlexiScore model...")
def _bootstrap():
    from data_generator import generate_dataset
    from scoring_model import load_or_train
    model, metrics = load_or_train(generate_dataset())
    return model, metrics


# ── Demo presets ──────────────────────────────────────────────────────────────
DEMO_CASES = {
    "Happy Path — Nguyễn Văn A": {
        "customer_id": "DEMO_001", "name": "Nguyễn Văn A",
        "customer_type": "gig_worker", "age": 32, "identity_verified": 1,
        "requested_amount": 20_000_000, "requested_tenor_months": 12,
        # B1 — Grab API
        "platform_tenure_months": 18, "active_days_per_week": 6.0,
        "rating_avg": 4.8, "cancel_rate": 0.03, "completion_rate": 0.97,
        "order_frequency": 6.0, "repeat_customer_ratio": 0.75,
        "seller_revenue_growth": 0.0, "refund_rate": 0.0,
        # B2 — VietQR/Bank
        "avg_monthly_income": 15_000_000, "income_cv": 0.15,
        "cashflow_drop_30d": 0.00, "monthly_inflow_count": 48,
        "expense_to_income_ratio": 0.55, "monthly_surplus_ratio": 0.40,
        "saving_buffer": 8_000_000, "wallet_activity_consistency": 0.92,
        # B3 — EVN/VNPT
        "bill_on_time_ratio": 0.98, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0.0,
        # B4 — Graph DB
        "shared_device_count": 0, "shared_ip_count": 0,
        "bad_neighbor_count": 0, "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.02,
        "trust_score": 0.92, "graph_risk_score": 0.08,
        # B5 — Auto
        "data_months_available": 18,
        "data_confidence": 0.88, "missing_data_ratio": 0.08,
        "source_reliability_score": 0.90,
    },
    "Risk-First — Trần Thị B": {
        "customer_id": "DEMO_002", "name": "Trần Thị B",
        "customer_type": "seller_online", "age": 28, "identity_verified": 1,
        "requested_amount": 40_000_000, "requested_tenor_months": 24,
        # B1
        "platform_tenure_months": 24, "active_days_per_week": 6.0,
        "rating_avg": 4.7, "cancel_rate": 0.05, "completion_rate": 0.96,
        "order_frequency": 8.0, "repeat_customer_ratio": 0.65,
        "seller_revenue_growth": 0.18, "refund_rate": 0.08,
        # B2
        "avg_monthly_income": 25_000_000, "income_cv": 0.20,
        "cashflow_drop_30d": 0.05, "monthly_inflow_count": 120,
        "expense_to_income_ratio": 0.55, "monthly_surplus_ratio": 0.40,
        "saving_buffer": 15_000_000, "wallet_activity_consistency": 0.88,
        # B3
        "bill_on_time_ratio": 0.96, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0.0,
        # B4
        "shared_device_count": 4, "shared_ip_count": 6,
        "bad_neighbor_count": 3, "fraud_ring_flag": 1,
        "circular_transaction_ratio": 0.28,
        "trust_score": 0.28, "graph_risk_score": 0.91,
        # B5
        "data_months_available": 20,
        "data_confidence": 0.80, "missing_data_ratio": 0.20,
        "source_reliability_score": 0.75,
    },
    "Credit Coach — Lê Văn C": {
        "customer_id": "DEMO_003", "name": "Lê Văn C",
        "customer_type": "freelancer", "age": 26, "identity_verified": 1,
        "requested_amount": 15_000_000, "requested_tenor_months": 12,
        # B1
        "platform_tenure_months": 8, "active_days_per_week": 3.5,
        "rating_avg": 4.2, "cancel_rate": 0.15, "completion_rate": 0.82,
        "order_frequency": 3.0, "repeat_customer_ratio": 0.40,
        "seller_revenue_growth": 0.0, "refund_rate": 0.0,
        # B2
        "avg_monthly_income": 12_000_000, "income_cv": 0.65,
        "cashflow_drop_30d": 0.42, "monthly_inflow_count": 8,
        "expense_to_income_ratio": 0.72, "monthly_surplus_ratio": 0.20,
        "saving_buffer": 2_000_000, "wallet_activity_consistency": 0.55,
        # B3
        "bill_on_time_ratio": 0.70, "utility_payment_delay": 2,
        "late_payment_count_6m": 3, "avg_payment_delay_days": 4.5,
        # B4
        "shared_device_count": 0, "shared_ip_count": 1,
        "bad_neighbor_count": 0, "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.03,
        "trust_score": 0.75, "graph_risk_score": 0.20,
        # B5
        "data_months_available": 6,
        "data_confidence": 0.55, "missing_data_ratio": 0.38,
        "source_reliability_score": 0.60,
    },
    "Vùng xám — Nguyễn Thị D": {
        "customer_id": "DEMO_004", "name": "Nguyễn Thị D",
        "customer_type": "small_merchant", "age": 35, "identity_verified": 1,
        "requested_amount": 25_000_000, "requested_tenor_months": 18,
        # B1
        "platform_tenure_months": 10, "active_days_per_week": 5.0,
        "rating_avg": 4.3, "cancel_rate": 0.10, "completion_rate": 0.88,
        "order_frequency": 4.5, "repeat_customer_ratio": 0.55,
        "seller_revenue_growth": 0.05, "refund_rate": 0.07,
        # B2
        "avg_monthly_income": 18_000_000, "income_cv": 0.38,
        "cashflow_drop_30d": 0.18, "monthly_inflow_count": 30,
        "expense_to_income_ratio": 0.65, "monthly_surplus_ratio": 0.28,
        "saving_buffer": 5_000_000, "wallet_activity_consistency": 0.68,
        # B3
        "bill_on_time_ratio": 0.82, "utility_payment_delay": 3,
        "late_payment_count_6m": 2, "avg_payment_delay_days": 2.5,
        # B4
        "shared_device_count": 1, "shared_ip_count": 2,
        "bad_neighbor_count": 1, "fraud_ring_flag": 0,
        "circular_transaction_ratio": 0.08,
        "trust_score": 0.65, "graph_risk_score": 0.32,
        # B5
        "data_months_available": 9,
        "data_confidence": 0.65, "missing_data_ratio": 0.25,
        "source_reliability_score": 0.70,
    },
}

TYPE_OPTS = ["gig_worker", "seller_online", "freelancer",
             "small_merchant", "thin_file_customer"]
TYPE_LABELS = {
    "gig_worker": "Tài xế công nghệ",
    "seller_online": "Seller online",
    "freelancer": "Freelancer",
    "small_merchant": "Tiểu thương",
    "thin_file_customer": "Khách hàng hồ sơ mỏng",
}
TENOR_OPTS = [3, 6, 9, 12, 18, 24]

DECISIONS = {
    "AUTO_REJECT":  {"label": "Từ chối tự động",            "emoji": "🚫", "color": "#b71c1c",
                     "desc": "Vi phạm quy tắc cứng F0 — không thể phê duyệt."},
    "AUTO_APPROVE": {"label": "Phê duyệt tự động",          "emoji": "✅", "color": "#1b5e20",
                     "desc": "Đủ điều kiện toàn diện — phê duyệt tự động với offer tối ưu."},
    "HUMAN_REVIEW": {"label": "Chuyển thẩm định",           "emoji": "🔍", "color": "#e65100",
                     "desc": "Vùng xám rủi ro — cần cán bộ tín dụng xem xét thêm."},
    "CREDIT_COACH": {"label": "Chưa duyệt — Lộ trình cải thiện", "emoji": "📚", "color": "#4a148c",
                     "desc": "Hồ sơ chưa đủ điều kiện — có lộ trình cụ thể để quay lại."},
}

# ── data_confidence auto-calc từ B5 ──────────────────────────────────────────
SOURCE_WEIGHTS = {
    "src_platform": 0.30,   # Grab/Shopee
    "src_bank":     0.30,   # VietQR/Bank
    "src_utility":  0.20,   # EVN/VNPT
    "src_graph":    0.20,   # Neo4j Graph
}

def _auto_data_confidence(src_flags: dict, months: int) -> tuple:
    """Tính data_confidence và missing_data_ratio tự động từ nguồn đã kết nối."""
    source_score = sum(
        SOURCE_WEIGHTS[k] for k, v in src_flags.items() if v
    )
    # Bonus nếu có nhiều tháng dữ liệu
    months_factor = min(months / 12.0, 1.0)
    confidence = round(source_score * (0.7 + 0.3 * months_factor), 3)
    missing    = round(1.0 - source_score, 3)
    reliability = round(confidence * 0.95, 3)
    return confidence, missing, reliability


# ── Chart helpers ─────────────────────────────────────────────────────────────
def _gauge(value, max_val, title, color="#1565c0", steps=None):
    steps = steps or [
        {"range": [0,              max_val * 0.45], "color": "#ffcdd2"},
        {"range": [max_val * 0.45, max_val * 0.60], "color": "#fff9c4"},
        {"range": [max_val * 0.60, max_val * 0.80], "color": "#c8e6c9"},
        {"range": [max_val * 0.80, max_val],         "color": "#a5d6a7"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar":  {"color": color},
            "steps": steps,
            "threshold": {"line": {"color": "#c62828", "width": 2},
                          "thickness": 0.7, "value": max_val * 0.70},
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
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=["#1565c0","#6a1b9a","#00695c","#e65100","#37474f"],
        text=[f"{v:.0f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Điểm thành phần (0–100 mỗi nhóm)",
        yaxis=dict(range=[0, 115], showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white", height=280,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def _ep_chart(offers):
    fig = go.Figure(go.Bar(
        x=[o.label for o in offers],
        y=[o.expected_profit / 1_000 for o in offers],
        marker_color=["#2e7d32" if o.ep_positive else "#c62828" for o in offers],
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
    return {"PASS": "✅ PASS", "WARN": "⚠️ WARN"}.get(status, "❌ FAIL")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _build_sidebar(base: dict):
    p = base.copy()
    with st.sidebar:
        st.markdown("### Hồ sơ đầu vào")
        st.caption("Chọn ví dụ để nạp preset, sau đó chỉnh trực tiếp.")

        case_name = st.selectbox(
            "Ví dụ minh họa:", list(DEMO_CASES.keys()), key="case_select",
        )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # ZONE A — RAW INPUTS (nhập tay thực sự)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown(
            "<div style='background:#e3f2fd;border-left:4px solid #1565c0;"
            "padding:6px 10px;border-radius:4px;margin-bottom:8px'>"
            "<strong>ZONE A — Đầu vào thực sự</strong><br>"
            "<span style='font-size:0.8em;color:#555'>"
            "Bank officer nhập hoặc lấy từ eKYC / form đăng ký</span></div>",
            unsafe_allow_html=True,
        )

        p["name"] = st.text_input("Tên khách hàng", value=str(p.get("name", "")))

        ctype_idx = TYPE_OPTS.index(p.get("customer_type", "gig_worker"))
        p["customer_type"] = st.selectbox(
            "Loại khách hàng", options=TYPE_OPTS,
            format_func=lambda x: TYPE_LABELS[x], index=ctype_idx,
        )
        p["age"] = st.slider("Tuổi", 18, 65, int(p.get("age", 30)), 1)
        p["identity_verified"] = int(st.checkbox(
            "Đã xác thực eKYC / định danh",
            value=bool(p.get("identity_verified", 1)),
        ))

        st.markdown("**Yêu cầu khoản vay**")
        p["requested_amount"] = st.number_input(
            "Số tiền vay (đ)", min_value=1_000_000, max_value=500_000_000,
            value=int(p.get("requested_amount", 20_000_000)),
            step=1_000_000, format="%d",
        )
        tenor_val = int(p.get("requested_tenor_months", 12))
        if tenor_val not in TENOR_OPTS:
            tenor_val = 12
        p["requested_tenor_months"] = st.selectbox(
            "Kỳ hạn vay (tháng)", TENOR_OPTS, index=TENOR_OPTS.index(tenor_val),
        )

        # ══════════════════════════════════════════════════════════════════════
        # ZONE B — FEATURE-ENGINEERED (auto-pull trong production)
        # ══════════════════════════════════════════════════════════════════════
        st.divider()
        st.markdown(
            "<div style='background:#f3e5f5;border-left:4px solid #6a1b9a;"
            "padding:6px 10px;border-radius:4px;margin-bottom:8px'>"
            "<strong>ZONE B — Feature-engineered từ data sources</strong><br>"
            "<span style='font-size:0.8em;color:#555'>"
            "Production: auto-pull &amp; tính tự động. "
            "Demo: chỉnh tay để mô phỏng.</span></div>",
            unsafe_allow_html=True,
        )

        # ── B5 trước: Data Source Connections (ảnh hưởng data_confidence) ────
        st.markdown("**B5 · Nguồn dữ liệu đã kết nối**")
        st.caption("Tick = nguồn đã kết nối → tự động tính data_confidence")
        src_platform = st.checkbox("📱 Grab / Shopee / Gojek API",  value=True, key="src_p")
        src_bank     = st.checkbox("🏦 VietQR / Bank statement",    value=True, key="src_b")
        src_utility  = st.checkbox("⚡ EVN / VNPT / Hóa đơn",       value=True, key="src_u")
        src_graph    = st.checkbox("🕸 Neo4j Graph DB",              value=True, key="src_g")

        data_months = st.slider(
            "Số tháng dữ liệu có sẵn", 0, 36,
            int(p.get("data_months_available", 12)), 1,
            help="Tốt: >=6. Không auto approve nếu <3.",
        )
        p["data_months_available"] = data_months

        src_flags = {
            "src_platform": src_platform,
            "src_bank":     src_bank,
            "src_utility":  src_utility,
            "src_graph":    src_graph,
        }
        conf, miss, rel = _auto_data_confidence(src_flags, data_months)
        p["data_confidence"]          = conf
        p["missing_data_ratio"]       = miss
        p["source_reliability_score"] = rel

        # ── B1 — Grab / Shopee API ────────────────────────────────────────────
        with st.expander(
            "B1 · 📱 Grab / Shopee API"
            + (" ✅" if src_platform else " ❌ (không có dữ liệu)"),
            expanded=src_platform,
        ):
            if not src_platform:
                st.caption("Nguồn chưa kết nối — dùng giá trị mặc định trung tính.")
            p["platform_tenure_months"] = st.slider(
                "Thâm niên nền tảng (tháng)",
                0, 60, int(p.get("platform_tenure_months", 12)), 1,
                help="Tốt: >6. Rủi ro: <4.",
                disabled=not src_platform,
            )
            p["active_days_per_week"] = st.slider(
                "Ngày tạo thu nhập / tuần  [active_days]",
                0.0, 7.0, float(p.get("active_days_per_week", 5.0)), 0.5,
                help="Gig tốt: >=5 ngày/tuần.",
                disabled=not src_platform,
            )
            p["rating_avg"] = st.slider(
                "Đánh giá TB nền tảng  [rating_avg]",
                1.0, 5.0, float(p.get("rating_avg", 4.5)), 0.1,
                help="Tốt: >4.5.",
                disabled=not src_platform,
            )
            p["cancel_rate"] = st.slider(
                "Tỷ lệ hủy đơn  [cancel_rate]",
                0.0, 0.5, float(p.get("cancel_rate", 0.05)), 0.01,
                help="Rủi ro: >0.20.",
                disabled=not src_platform,
            )
            p["completion_rate"] = st.slider(
                "Tỷ lệ hoàn thành  [completion_rate]",
                0.0, 1.0, float(p.get("completion_rate", 0.90)), 0.01,
                help="Tốt: >0.90.",
                disabled=not src_platform,
            )
            p["order_frequency"] = st.slider(
                "Tần suất đơn hàng / ngày  [order_frequency]",
                0.0, 15.0, float(p.get("order_frequency", 4.0)), 0.5,
                disabled=not src_platform,
            )
            p["repeat_customer_ratio"] = st.slider(
                "Tỷ lệ khách quay lại  [repeat_customer_ratio]",
                0.0, 1.0, float(p.get("repeat_customer_ratio", 0.5)), 0.01,
                disabled=not src_platform,
            )
            p["seller_revenue_growth"] = st.slider(
                "Tăng trưởng doanh thu  [seller_revenue_growth]",
                -0.5, 1.0, float(p.get("seller_revenue_growth", 0.0)), 0.01,
                help="Ổn định / dương là tốt.",
                disabled=not src_platform,
            )
            p["refund_rate"] = st.slider(
                "Tỷ lệ hoàn đơn  [refund_rate]",
                0.0, 0.5, float(p.get("refund_rate", 0.0)), 0.01,
                help="Tốt: <0.10. Rủi ro: >0.25.",
                disabled=not src_platform,
            )

        # ── B2 — VietQR / Bank ────────────────────────────────────────────────
        with st.expander(
            "B2 · 🏦 VietQR / Bank transactions"
            + (" ✅" if src_bank else " ❌ (không có dữ liệu)"),
            expanded=src_bank,
        ):
            if not src_bank:
                st.caption("Nguồn chưa kết nối — thu nhập và dòng tiền không xác thực được.")
            p["avg_monthly_income"] = st.number_input(
                "Thu nhập TB tháng  [avg_monthly_income] (đ)",
                min_value=1_000_000, max_value=200_000_000,
                value=int(p.get("avg_monthly_income", 10_000_000)),
                step=500_000, format="%d",
                help="= mean(monthly_inflows) từ transaction history.",
                disabled=not src_bank,
            )
            p["income_cv"] = st.slider(
                "Biến động thu nhập  [income_cv]",
                0.0, 1.0, float(p.get("income_cv", 0.3)), 0.01,
                help="= std(monthly_income) / mean(monthly_income). "
                     "Tốt: <0.30. >0.55 → F1 fail. >0.75 → F0 fail.",
                disabled=not src_bank,
            )
            p["cashflow_drop_30d"] = st.slider(
                "Sụt giảm dòng tiền 30 ngày  [cashflow_drop_30d]",
                0.0, 0.9, float(p.get("cashflow_drop_30d", 0.0)), 0.01,
                help="= (baseline_income − last30d) / baseline. Cảnh báo: >0.30.",
                disabled=not src_bank,
            )
            p["monthly_inflow_count"] = st.slider(
                "Số giao dịch tiền vào / tháng  [monthly_inflow_count]",
                1, 200, int(p.get("monthly_inflow_count", 20)), 1,
                help="= count(inflow_txns) / months.",
                disabled=not src_bank,
            )
            p["expense_to_income_ratio"] = st.slider(
                "Chi tiêu / thu nhập  [expense_to_income_ratio]",
                0.1, 1.0, float(p.get("expense_to_income_ratio", 0.6)), 0.01,
                help="= sum(outflows) / sum(inflows). Tốt: <0.60.",
                disabled=not src_bank,
            )
            p["monthly_surplus_ratio"] = st.slider(
                "Tỷ lệ dòng tiền dư  [monthly_surplus_ratio]",
                0.0, 0.9, float(p.get("monthly_surplus_ratio", 0.3)), 0.01,
                help="= (inflows − outflows) / inflows. Tốt: >0.25.",
                disabled=not src_bank,
            )
            p["saving_buffer"] = st.number_input(
                "Số dư / dự phòng  [saving_buffer] (đ)",
                min_value=0, max_value=500_000_000,
                value=int(p.get("saving_buffer", 3_000_000)),
                step=500_000, format="%d",
                help="Current balance hoặc cumulative surplus.",
                disabled=not src_bank,
            )
            p["wallet_activity_consistency"] = st.slider(
                "Độ đều hoạt động ví  [wallet_activity_consistency]",
                0.0, 1.0, float(p.get("wallet_activity_consistency", 0.7)), 0.01,
                help="= 1 − std(weekly_tx_count)/mean(weekly_tx_count). Tốt: >0.75.",
                disabled=not src_bank,
            )

        # ── B3 — EVN / VNPT ───────────────────────────────────────────────────
        with st.expander(
            "B3 · ⚡ EVN / VNPT / Hóa đơn"
            + (" ✅" if src_utility else " ❌ (không có dữ liệu)"),
            expanded=False,
        ):
            if not src_utility:
                st.caption("Nguồn chưa kết nối — kỷ luật thanh toán không xác thực được.")
            p["bill_on_time_ratio"] = st.slider(
                "Tỷ lệ thanh toán đúng hạn  [bill_on_time_ratio]",
                0.0, 1.0, float(p.get("bill_on_time_ratio", 0.85)), 0.01,
                help="= on_time_count / total_bills. Tốt: >0.90.",
                disabled=not src_utility,
            )
            p["utility_payment_delay"] = st.slider(
                "Số ngày trễ hóa đơn điện/nước  [utility_payment_delay]",
                0, 30, int(p.get("utility_payment_delay", 0)), 1,
                help="= mean(payment_date − due_date). Tốt: 0–2. Rủi ro: >7.",
                disabled=not src_utility,
            )
            p["late_payment_count_6m"] = st.slider(
                "Số lần trễ trong 6 tháng  [late_payment_count_6m]",
                0, 12, int(p.get("late_payment_count_6m", 0)), 1,
                help="= count(payments where delay > 0) in 6 months. Tốt: 0–1.",
                disabled=not src_utility,
            )
            p["avg_payment_delay_days"] = st.slider(
                "Số ngày trễ trung bình  [avg_payment_delay_days]",
                0.0, 30.0, float(p.get("avg_payment_delay_days", 0.0)), 0.5,
                help="= mean(delay_days) across all late payments. Tốt: <3.",
                disabled=not src_utility,
            )

        # ── B4 — Graph DB (Neo4j) ─────────────────────────────────────────────
        with st.expander(
            "B4 · 🕸 Neo4j Graph DB"
            + (" ✅" if src_graph else " ❌ (không có dữ liệu)"),
            expanded=src_graph,
        ):
            if not src_graph:
                st.caption("Nguồn chưa kết nối — rủi ro mạng lưới không phân tích được.")
            p["fraud_ring_flag"] = int(st.checkbox(
                "Fraud ring flag — thuộc cụm gian lận  [fraud_ring_flag]",
                value=bool(p.get("fraud_ring_flag", 0)),
                help="True → F0 FAIL → AUTO_REJECT ngay.",
                disabled=not src_graph,
            ))
            p["shared_device_count"] = st.slider(
                "Hồ sơ dùng chung thiết bị  [shared_device_count]",
                0, 10, int(p.get("shared_device_count", 0)), 1,
                help="MATCH (u)-[:SHARES_DEVICE]-(o). >=4 → F0 FAIL.",
                disabled=not src_graph,
            )
            p["shared_ip_count"] = st.slider(
                "Hồ sơ dùng chung IP  [shared_ip_count]",
                0, 15, int(p.get("shared_ip_count", 0)), 1,
                help="MATCH (u)-[:SHARES_IP]-(o). Cảnh báo khi cao.",
                disabled=not src_graph,
            )
            p["bad_neighbor_count"] = st.slider(
                "Liên kết hồ sơ nợ xấu  [bad_neighbor_count]",
                0, 8, int(p.get("bad_neighbor_count", 0)), 1,
                help="Graph traversal 1–3 hops tới fraud nodes. >=3 → F0 FAIL.",
                disabled=not src_graph,
            )
            p["circular_transaction_ratio"] = st.slider(
                "Tỷ lệ giao dịch vòng  [circular_transaction_ratio]",
                0.0, 0.9, float(p.get("circular_transaction_ratio", 0.02)), 0.01,
                help="= cycles / total_transactions. Cảnh báo: >0.20.",
                disabled=not src_graph,
            )
            p["trust_score"] = st.slider(
                "Điểm tin cậy mạng lưới  [trust_score]  (PageRank)",
                0.0, 1.0, float(p.get("trust_score", 0.70)), 0.01,
                help="PageRank trên transaction graph. Càng cao càng tốt.",
                disabled=not src_graph,
            )
            p["graph_risk_score"] = st.slider(
                "Điểm rủi ro graph  [graph_risk_score]",
                0.0, 1.0, float(p.get("graph_risk_score", 0.20)), 0.01,
                help="Composite từ graph features. >0.70 → Fraud Review.",
                disabled=not src_graph,
            )

        # ── B5 summary (read-only) ────────────────────────────────────────────
        st.divider()
        conf_color = (
            "#c62828" if conf < 0.40 else
            ("#f57c00" if conf < 0.60 else
             ("#1565c0" if conf < 0.70 else "#1b5e20"))
        )
        st.markdown(
            f"<div style='background:{conf_color}15;border:1px solid {conf_color};"
            f"padding:8px 12px;border-radius:6px'>"
            f"<strong style='color:{conf_color}'>Data confidence (auto): {conf:.0%}</strong><br>"
            f"<span style='font-size:0.8em'>Missing ratio: {miss:.0%} &nbsp;|&nbsp; "
            f"Source reliability: {rel:.0%}</span></div>",
            unsafe_allow_html=True,
        )
        hint = (
            "❌ F0 FAIL — quá ít dữ liệu" if conf < 0.40 else
            ("📚 → CREDIT_COACH (cần kết nối thêm nguồn)" if conf < 0.60 else
             ("✅ Đủ cho AUTO_APPROVE" if conf >= 0.70 else
              "🔍 Có thể HUMAN_REVIEW"))
        )
        st.caption(hint)

        st.divider()
        run = st.button("Chạy FlexiScore Pipeline", use_container_width=True, type="primary")

    return case_name, p, run


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FlexiScore Demo", page_icon="💳",
        layout="wide", initial_sidebar_state="expanded",
    )

    st.markdown(
        "<h1 style='text-align:center;color:#1565c0;margin-bottom:2px'>💳 FlexiScore</h1>"
        "<p style='text-align:center;color:#666;font-size:1em;margin-top:0'>"
        "Alternative Credit Scoring for Thin-file Customers"
        " &nbsp;·&nbsp; Risk-first &nbsp;·&nbsp; Cash-flow optimised &nbsp;·&nbsp; Explainable"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    model, metrics = _bootstrap()

    # Case switching
    if "last_case" not in st.session_state:
        st.session_state["last_case"] = list(DEMO_CASES.keys())[0]
    prev_case = st.session_state["last_case"]
    base = DEMO_CASES.get(prev_case, list(DEMO_CASES.values())[0]).copy()

    case_name, customer, run_clicked = _build_sidebar(base)

    if case_name != prev_case:
        st.session_state["last_case"] = case_name
        st.rerun()

    # Pipeline
    if run_clicked or "result" not in st.session_state:
        from decision_engine import make_decision
        from explainability import generate_reason_codes
        with st.spinner("Đang chạy pipeline..."):
            result  = make_decision(customer, model)
            explain = generate_reason_codes(customer, result)
        st.session_state.update(
            result=result, explain=explain, customer=customer,
        )
    else:
        result   = st.session_state["result"]
        explain  = st.session_state["explain"]
        customer = st.session_state["customer"]

    decision = result["decision"]
    dmeta    = DECISIONS.get(decision, DECISIONS["HUMAN_REVIEW"])

    # Banner
    d_color, d_emoji = dmeta["color"], dmeta["emoji"]
    d_label, d_desc  = dmeta["label"], dmeta["desc"]
    st.markdown(
        f"<div style='background:{d_color};color:white;padding:16px 24px;"
        f"border-radius:10px;margin-bottom:18px'>"
        f"<span style='font-size:1.5em'>{d_emoji}</span>&nbsp;&nbsp;"
        f"<strong style='font-size:1.2em'>{d_label}</strong><br>"
        f"<span style='font-size:0.9em;opacity:0.9'>{d_desc}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Metric strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FlexiScore", f"{result['flexiscore']:.0f} / 1000",
              delta=result["risk_tier"], delta_color="off")
    c2.metric("PD", f"{result['pd']:.1%}",
              delta=result["pd_band"], delta_color="off")
    c3.metric("F0 Hard Rules", _gate_badge(result["f0_status"]))
    c4.metric("F1 Stress Test", _gate_badge(result["f1_status"]))

    # Data confidence indicator
    conf    = customer.get("data_confidence", 1.0)
    n_srcs  = sum([
        st.session_state.get("src_p", True),
        st.session_state.get("src_b", True),
        st.session_state.get("src_u", True),
        st.session_state.get("src_g", True),
    ])
    conf_bar = "🟢" * n_srcs + "⬜" * (4 - n_srcs)
    st.caption(
        f"Data confidence: **{conf:.0%}** &nbsp;|&nbsp; "
        f"Nguồn kết nối: {conf_bar} ({n_srcs}/4)"
    )

    st.divider()

    # Two-column layout
    left, right = st.columns([1, 1], gap="large")

    # ── Customer view ─────────────────────────────────────────────────────────
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
                st.table(pd.DataFrame({
                    "": ["Số tiền", "Kỳ hạn", "Lịch trả",
                         "Lãi suất", "Góp ước tính", "Expected Profit (NH)"],
                    "Giá trị": [
                        f"{rec.amount:,.0f} đ",
                        f"{rec.tenor_months} tháng",
                        rec.schedule,
                        f"{rec.interest_rate:.0%} / năm",
                        f"{rec.monthly_payment:,.0f} đ / lần",
                        f"+{rec.expected_profit:,.0f} đ",
                    ],
                }).set_index(""))

        elif decision == "AUTO_REJECT":
            st.error("Hồ sơ không đủ điều kiện xử lý.")
            st.markdown("Bạn có quyền yêu cầu cán bộ tín dụng xem xét lại.")
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
                "Hồ sơ đang được chuyển sang thẩm định thêm.\n\n"
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

    # ── Bank / Risk view ──────────────────────────────────────────────────────
    with right:
        st.subheader("🏦 Bank / Risk View")

        st.plotly_chart(
            _gauge(result["flexiscore"], 1000, "FlexiScore (0–1000)", color="#1565c0"),
            use_container_width=True,
        )

        gr_label = result.get("graph_risk_label", "LOW")
        gr_color = ("#c62828" if gr_label == "HIGH"
                    else ("#f57c00" if gr_label == "MEDIUM" else "#2e7d32"))

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.plotly_chart(
                _gauge(
                    result["graph_risk_score"] * 100, 100, "Graph Risk (%)",
                    color=gr_color,
                    steps=[{"range":[0,40],"color":"#e8f5e9"},
                           {"range":[40,70],"color":"#fff8e1"},
                           {"range":[70,100],"color":"#ffebee"}],
                ), use_container_width=True,
            )
        with gcol2:
            st.plotly_chart(
                _gauge(
                    result["trust_score"] * 100, 100, "Trust Score (%)",
                    color="#00695c",
                    steps=[{"range":[0,40],"color":"#ffebee"},
                           {"range":[40,70],"color":"#fff8e1"},
                           {"range":[70,100],"color":"#e8f5e9"}],
                ), use_container_width=True,
            )

        with st.expander("Chi tiết Graph Risk"):
            for r in result.get("graph_reasons", []):
                st.markdown(f"- {r}")

        st.plotly_chart(_bar_subscores(result), use_container_width=True)

        offers = result.get("offers", [])
        if offers:
            st.markdown("**So sánh Loan Offers**")
            rec_label = result["recommended_offer"].label if result["recommended_offer"] else ""
            rows = [{
                "Offer":    o.label + (" ⭐" if o.label == rec_label else ""),
                "Số tiền": f"{o.amount/1e6:.1f}M",
                "Kỳ hạn":  f"{o.tenor_months}T",
                "Lịch":    o.schedule,
                "EP":      f"{o.expected_profit/1_000:,.0f}K đ",
                "OK?":     "✅" if o.ep_positive else "❌",
            } for o in offers]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.plotly_chart(_ep_chart(offers), use_container_width=True)

    # ── Bottom expanders ──────────────────────────────────────────────────────
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
                for r in f0r: st.markdown(f"- {r}")
            else:
                st.success("Không có vi phạm F0.")
    with b3:
        with st.expander("⚡ F1 Stress Test — Chi tiết"):
            dti = result.get("dti_stress_ratio", 0)
            st.markdown(f"**DTI stress:** {dti:.1%}  *(ngưỡng 35%)*")
            f1r = result.get("f1_reasons", [])
            if f1r:
                for r in f1r: st.markdown(f"- {r}")
            else:
                st.success("Không có vi phạm F1.")

    with st.expander("📈 Model Info & Decision Thresholds"):
        mi = st.columns(4)
        if metrics:
            mi[0].metric("ROC-AUC",   metrics.get("roc_auc",   "—"))
            mi[1].metric("Accuracy",  metrics.get("accuracy",  "—"))
            mi[2].metric("Precision", metrics.get("precision", "—"))
            mi[3].metric("Recall",    metrics.get("recall",    "—"))
        else:
            st.info("Model load từ cache.")
        st.caption(
            "AUTO_APPROVE: FlexiScore ≥ 700 · PD < 15% · data_confidence ≥ 70% · EP > 0  |  "
            "CREDIT_COACH: F1 fail (income_cv > 0.55 hoặc DTI > 35%) hoặc data_confidence < 60%  |  "
            "AUTO_REJECT: F0 fail  |  HUMAN_REVIEW: còn lại"
        )

    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:0.78em;margin-top:16px'>"
        "FlexiScore Demo — Dữ liệu giả lập, không phải dữ liệu thật của SHB.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
