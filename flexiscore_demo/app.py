"""
app.py — FlexiScore Demo Dashboard
Chạy: streamlit run app.py

PROTOTYPE — chỉ để minh họa logic pipeline, không phải hệ thống vận hành.
Mọi quyết định tính thuần túy từ giá trị tham số — không hard-code theo tên use-case.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Bootstrap: load/train model một lần ──────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi tạo FlexiScore model...")
def _bootstrap():
    from data_generator import generate_dataset
    from scoring_model import load_or_train
    return load_or_train(generate_dataset())


# ── 4 Demo presets ────────────────────────────────────────────────────────────
DEMO_CASES = {
    "Happy Path — Nguyễn Văn A": {
        "customer_id": "DEMO_001", "name": "Nguyễn Văn A",
        "customer_type": "gig_worker",
        "age": 32, "identity_verified": 1,
        "requested_amount": 20_000_000, "requested_tenor_months": 12,
        "avg_monthly_income": 15_000_000, "income_cv": 0.15,
        "cashflow_drop_30d": 0.00, "bill_on_time_ratio": 0.98,
        "fraud_ring_flag": 0, "shared_device_count": 0, "bad_neighbor_count": 0,
        # phụ trợ (B-zone, không chỉnh trực tiếp)
        "platform_tenure_months": 18, "active_days_per_week": 6.0,
        "rating_avg": 4.8, "cancel_rate": 0.03, "completion_rate": 0.97,
        "order_frequency": 6.0, "repeat_customer_ratio": 0.75,
        "seller_revenue_growth": 0.0, "refund_rate": 0.0,
        "monthly_inflow_count": 48, "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40, "saving_buffer": 8_000_000,
        "wallet_activity_consistency": 0.92, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0.0,
        "shared_ip_count": 0, "circular_transaction_ratio": 0.02,
        "trust_score": 0.92, "graph_risk_score": 0.08,
        "data_months_available": 18,
        "data_confidence": 0.88, "missing_data_ratio": 0.08,
        "source_reliability_score": 0.90,
    },
    "Risk-First — Trần Thị B": {
        "customer_id": "DEMO_002", "name": "Trần Thị B",
        "customer_type": "seller_online",
        "age": 28, "identity_verified": 1,
        "requested_amount": 40_000_000, "requested_tenor_months": 24,
        "avg_monthly_income": 25_000_000, "income_cv": 0.20,
        "cashflow_drop_30d": 0.05, "bill_on_time_ratio": 0.96,
        "fraud_ring_flag": 1, "shared_device_count": 4, "bad_neighbor_count": 3,
        "platform_tenure_months": 24, "active_days_per_week": 6.0,
        "rating_avg": 4.7, "cancel_rate": 0.05, "completion_rate": 0.96,
        "order_frequency": 8.0, "repeat_customer_ratio": 0.65,
        "seller_revenue_growth": 0.18, "refund_rate": 0.08,
        "monthly_inflow_count": 120, "expense_to_income_ratio": 0.55,
        "monthly_surplus_ratio": 0.40, "saving_buffer": 15_000_000,
        "wallet_activity_consistency": 0.88, "utility_payment_delay": 0,
        "late_payment_count_6m": 0, "avg_payment_delay_days": 0.0,
        "shared_ip_count": 6, "circular_transaction_ratio": 0.28,
        "trust_score": 0.28, "graph_risk_score": 0.91,
        "data_months_available": 20,
        "data_confidence": 0.80, "missing_data_ratio": 0.20,
        "source_reliability_score": 0.75,
    },
    "Credit Coach — Lê Văn C": {
        "customer_id": "DEMO_003", "name": "Lê Văn C",
        "customer_type": "freelancer",
        "age": 26, "identity_verified": 1,
        "requested_amount": 15_000_000, "requested_tenor_months": 12,
        "avg_monthly_income": 12_000_000, "income_cv": 0.65,
        "cashflow_drop_30d": 0.42, "bill_on_time_ratio": 0.70,
        "fraud_ring_flag": 0, "shared_device_count": 0, "bad_neighbor_count": 0,
        "platform_tenure_months": 8, "active_days_per_week": 3.5,
        "rating_avg": 4.2, "cancel_rate": 0.15, "completion_rate": 0.82,
        "order_frequency": 3.0, "repeat_customer_ratio": 0.40,
        "seller_revenue_growth": 0.0, "refund_rate": 0.0,
        "monthly_inflow_count": 8, "expense_to_income_ratio": 0.72,
        "monthly_surplus_ratio": 0.20, "saving_buffer": 2_000_000,
        "wallet_activity_consistency": 0.55, "utility_payment_delay": 2,
        "late_payment_count_6m": 3, "avg_payment_delay_days": 4.5,
        "shared_ip_count": 1, "circular_transaction_ratio": 0.03,
        "trust_score": 0.75, "graph_risk_score": 0.20,
        "data_months_available": 6,
        "data_confidence": 0.55, "missing_data_ratio": 0.38,
        "source_reliability_score": 0.60,
    },
    "Vùng xám — Nguyễn Thị D": {
        "customer_id": "DEMO_004", "name": "Nguyễn Thị D",
        "customer_type": "small_merchant",
        "age": 35, "identity_verified": 1,
        "requested_amount": 25_000_000, "requested_tenor_months": 18,
        "avg_monthly_income": 18_000_000, "income_cv": 0.38,
        "cashflow_drop_30d": 0.18, "bill_on_time_ratio": 0.82,
        "fraud_ring_flag": 0, "shared_device_count": 1, "bad_neighbor_count": 1,
        "platform_tenure_months": 10, "active_days_per_week": 5.0,
        "rating_avg": 4.3, "cancel_rate": 0.10, "completion_rate": 0.88,
        "order_frequency": 4.5, "repeat_customer_ratio": 0.55,
        "seller_revenue_growth": 0.05, "refund_rate": 0.07,
        "monthly_inflow_count": 30, "expense_to_income_ratio": 0.65,
        "monthly_surplus_ratio": 0.28, "saving_buffer": 5_000_000,
        "wallet_activity_consistency": 0.68, "utility_payment_delay": 3,
        "late_payment_count_6m": 2, "avg_payment_delay_days": 2.5,
        "shared_ip_count": 2, "circular_transaction_ratio": 0.08,
        "trust_score": 0.65, "graph_risk_score": 0.32,
        "data_months_available": 9,
        "data_confidence": 0.65, "missing_data_ratio": 0.25,
        "source_reliability_score": 0.70,
    },
}

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
        "desc":  "Vi phạm quy tắc cứng F0 — pipeline dừng, không tiếp tục ML.",
    },
    "CREDIT_COACH": {
        "label": "Chưa duyệt — Lộ trình cải thiện",
        "emoji": "📚",
        "color": "#4a148c",
        "desc":  "Vượt qua F0, nhưng chưa đủ điều kiện vay. Có lộ trình 90 ngày để quay lại.",
    },
    "AUTO_APPROVE": {
        "label": "Phê duyệt tự động",
        "emoji": "✅",
        "color": "#1b5e20",
        "desc":  "Hồ sơ đạt toàn bộ tiêu chí — phê duyệt tự động với offer tốt nhất.",
    },
    "HUMAN_REVIEW": {
        "label": "Chuyển thẩm định",
        "emoji": "🔍",
        "color": "#e65100",
        "desc":  "Hồ sơ vùng xám — cần cán bộ tín dụng xem xét thêm trước khi quyết định.",
    },
}

# Trọng số nguồn dữ liệu → data_confidence
SOURCE_WEIGHTS = {"src_p": 0.30, "src_b": 0.30, "src_u": 0.20, "src_g": 0.20}


def _calc_confidence(flags: dict, months: int) -> float:
    raw  = sum(SOURCE_WEIGHTS[k] for k, v in flags.items() if v)
    mfac = min(months / 12.0, 1.0)
    return round(raw * (0.7 + 0.3 * mfac), 3)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _sidebar(base: dict):
    p = base.copy()
    with st.sidebar:
        st.markdown("## 📋 Hồ sơ đầu vào")
        case = st.selectbox(
            "Ví dụ minh họa",
            list(DEMO_CASES.keys()),
            key="case_select",
            help="Chọn preset để nạp giá trị mẫu. Chỉnh bất kỳ tham số nào để thử nghiệm.",
        )
        st.caption(
            "💡 Use-case chỉ là preset khởi đầu. "
            "Mọi thay đổi tham số đều ảnh hưởng trực tiếp tới quyết định."
        )
        st.divider()

        # ── Khoản vay ─────────────────────────────────────────────────────────
        st.markdown("#### 💰 Khoản vay")
        p["requested_amount"] = st.number_input(
            "Số tiền vay (đ)",
            min_value=1_000_000, max_value=500_000_000,
            value=int(p.get("requested_amount", 20_000_000)),
            step=1_000_000, format="%d",
        )
        tenor = int(p.get("requested_tenor_months", 12))
        p["requested_tenor_months"] = st.select_slider(
            "Kỳ hạn (tháng)", options=TENOR_OPTS,
            value=tenor if tenor in TENOR_OPTS else 12,
        )

        # ── Thu nhập & Dòng tiền ──────────────────────────────────────────────
        st.markdown("#### 📈 Thu nhập & Dòng tiền")
        p["avg_monthly_income"] = st.number_input(
            "Thu nhập TB tháng (đ)",
            min_value=1_000_000, max_value=200_000_000,
            value=int(p.get("avg_monthly_income", 10_000_000)),
            step=500_000, format="%d",
            help="Trích xuất từ VietQR / sao kê ngân hàng.",
        )
        p["income_cv"] = st.slider(
            "Biến động thu nhập  [income_cv]",
            0.0, 1.0, float(p.get("income_cv", 0.30)), 0.01,
            help="0 = rất ổn định · >0.55 → F1 fail · >0.75 → F0 fail (từ chối ngay)",
        )
        p["cashflow_drop_30d"] = st.slider(
            "Sụt giảm dòng tiền 30 ngày  [cashflow_drop_30d]",
            0.0, 0.90, float(p.get("cashflow_drop_30d", 0.0)), 0.01,
            help="0 = không sụt giảm · >0.30 → cảnh báo F1",
        )

        # ── Kỷ luật thanh toán ────────────────────────────────────────────────
        st.markdown("#### 🧾 Kỷ luật thanh toán")
        p["bill_on_time_ratio"] = st.slider(
            "Tỷ lệ thanh toán đúng hạn  [bill_on_time_ratio]",
            0.0, 1.0, float(p.get("bill_on_time_ratio", 0.85)), 0.01,
            help="Từ EVN / VNPT / hóa đơn định kỳ · >0.90 là tốt",
        )

        # ── Rủi ro mạng lưới ──────────────────────────────────────────────────
        st.markdown("#### 🕸 Rủi ro mạng lưới (Neo4j Graph)")
        p["fraud_ring_flag"] = int(st.checkbox(
            "Phát hiện liên kết cụm gian lận  [fraud_ring_flag]",
            value=bool(p.get("fraud_ring_flag", 0)),
            help="True → F0 FAIL → AUTO_REJECT ngay lập tức",
        ))
        p["shared_device_count"] = st.slider(
            "Thiết bị dùng chung  [shared_device_count]",
            0, 10, int(p.get("shared_device_count", 0)), 1,
            help="≥ 4 → F0 FAIL",
        )
        p["bad_neighbor_count"] = st.slider(
            "Liên kết hồ sơ nợ xấu  [bad_neighbor_count]",
            0, 8, int(p.get("bad_neighbor_count", 0)), 1,
            help="≥ 3 → F0 FAIL",
        )

        # ── Độ tin cậy dữ liệu ────────────────────────────────────────────────
        st.markdown("#### 🔗 Nguồn dữ liệu đã kết nối")
        st.caption("Mỗi nguồn đóng góp vào data_confidence — ảnh hưởng trực tiếp tới quyết định.")

        c1, c2 = st.columns(2)
        src_p = c1.checkbox("📱 Grab/Shopee",  value=True, key="src_p")
        src_b = c2.checkbox("🏦 Ngân hàng",    value=True, key="src_b")
        src_u = c1.checkbox("⚡ EVN/VNPT",      value=True, key="src_u")
        src_g = c2.checkbox("🕸 Graph DB",      value=True, key="src_g")

        flags = {"src_p": src_p, "src_b": src_b, "src_u": src_u, "src_g": src_g}
        months = int(p.get("data_months_available", 12))
        conf   = _calc_confidence(flags, months)

        p["data_confidence"]          = conf
        p["missing_data_ratio"]       = round(1 - conf, 3)
        p["source_reliability_score"] = round(conf * 0.95, 3)

        # Confidence indicator
        n_src = sum(flags.values())
        bar   = "🟢" * n_src + "⬜" * (4 - n_src)
        conf_color = (
            "#c62828" if conf < 0.40 else
            "#f57c00" if conf < 0.60 else
            "#1565c0" if conf < 0.70 else "#1b5e20"
        )
        # Hint dựa trên giá trị confidence, không phải số nguồn
        # (1 nguồn nhỏ như EVN/VNPT = 20% vẫn có thể < 40%)
        if conf < 0.40:
            need = "ngân hàng (30%) hoặc platform (30%)" if n_src > 0 else "ít nhất ngân hàng hoặc platform"
            conf_hint = f"❌ F0 FAIL — confidence {conf:.0%} < 40% · Cần thêm {need}"
        elif conf < 0.60:
            conf_hint = f"📚 → CREDIT_COACH — confidence {conf:.0%} < 60% · Kết nối thêm nguồn"
        elif conf < 0.70:
            conf_hint = f"⚠️ HUMAN_REVIEW — confidence {conf:.0%} < 70% · Cần thêm để auto-approve"
        else:
            conf_hint = f"✅ Đủ điều kiện auto-approve — confidence {conf:.0%} ≥ 70%"
        st.markdown(
            f"<div style='margin-top:6px;padding:8px 12px;"
            f"background:{conf_color}15;border-left:4px solid {conf_color};"
            f"border-radius:4px'>"
            f"<strong style='color:{conf_color}'>Data confidence: {conf:.0%}</strong>"
            f"  {bar}<br>"
            f"<span style='font-size:0.82em;color:#555'>{conf_hint}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Tham số phụ (ít ảnh hưởng hơn) ──────────────────────────────────
        with st.expander("⚙️ Tham số phụ", expanded=False):
            st.caption(
                "Các biến này ảnh hưởng tới FlexiScore nhưng không tác động trực tiếp "
                "vào F0/F1 gates. Giá trị mặc định lấy từ preset."
            )
            p["active_days_per_week"] = st.slider(
                "Ngày hoạt động / tuần", 0.0, 7.0,
                float(p.get("active_days_per_week", 5.0)), 0.5,
                help="Ảnh hưởng tới Income Stability (35% FlexiScore)",
            )
            p["wallet_activity_consistency"] = st.slider(
                "Độ đều giao dịch ví  [wallet_consistency]",
                0.0, 1.0, float(p.get("wallet_activity_consistency", 0.70)), 0.01,
                help="Ảnh hưởng tới Digital Transactions (20% FlexiScore)",
            )
            p["monthly_surplus_ratio"] = st.slider(
                "Tỷ lệ thặng dư tháng  [monthly_surplus_ratio]",
                0.0, 0.9, float(p.get("monthly_surplus_ratio", 0.30)), 0.01,
            )
            p["late_payment_count_6m"] = st.slider(
                "Số lần trễ hạn 6 tháng  [late_payment_count_6m]",
                0, 12, int(p.get("late_payment_count_6m", 0)), 1,
                help="Ảnh hưởng tới Financial Commitment (15% FlexiScore)",
            )
            p["trust_score"] = st.slider(
                "Điểm tin cậy mạng lưới  [trust_score]",
                0.0, 1.0, float(p.get("trust_score", 0.70)), 0.01,
                help="Ảnh hưởng tới Graph Safety (25% FlexiScore)",
            )
            p["graph_risk_score"] = st.slider(
                "Điểm rủi ro graph  [graph_risk_score]",
                0.0, 1.0, float(p.get("graph_risk_score", 0.20)), 0.01,
            )
            st.markdown("**Platform Behavior (5% FlexiScore)**")
            p["rating_avg"] = st.slider(
                "Đánh giá TB", 1.0, 5.0, float(p.get("rating_avg", 4.5)), 0.1,
            )
            p["cancel_rate"] = st.slider(
                "Tỷ lệ hủy đơn", 0.0, 0.5, float(p.get("cancel_rate", 0.05)), 0.01,
            )
            p["platform_tenure_months"] = st.slider(
                "Thâm niên nền tảng (tháng)", 0, 60,
                int(p.get("platform_tenure_months", 12)), 1,
            )

        st.divider()
        run = st.button("▶ Chạy Pipeline", use_container_width=True, type="primary")

    return case, p, run


# ── Chart: FlexiScore gauge ───────────────────────────────────────────────────
def _gauge_flexiscore(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "FlexiScore", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 1000]},
            "bar":  {"color": "#1565c0"},
            "steps": [
                {"range": [0,   450], "color": "#ffcdd2"},
                {"range": [450, 600], "color": "#fff9c4"},
                {"range": [600, 700], "color": "#bbdefb"},
                {"range": [700, 1000], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "#1b5e20", "width": 3},
                "thickness": 0.75, "value": 700,
            },
        },
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=10, r=10))
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FlexiScore Demo", page_icon="💳",
        layout="wide", initial_sidebar_state="expanded",
    )

    # Header
    st.markdown(
        "<h1 style='text-align:center;color:#1565c0;margin-bottom:4px'>💳 FlexiScore Demo</h1>"
        "<p style='text-align:center;color:#888;font-size:0.95em;margin-top:0'>"
        "Alternative Credit Scoring · Thin-file customers · Risk-first · Explainable"
        "</p>",
        unsafe_allow_html=True,
    )
    st.info(
        "**Prototype minh họa logic pipeline FlexiScore** — không phải hệ thống vận hành tín dụng. "
        "Dữ liệu 100% giả lập. Mọi quyết định tính từ giá trị tham số, không hard-code theo use-case.",
        icon="ℹ️",
    )

    model, metrics = _bootstrap()

    # Case switching
    if "last_case" not in st.session_state:
        st.session_state["last_case"] = list(DEMO_CASES.keys())[0]

    prev_case = st.session_state["last_case"]
    base = DEMO_CASES.get(prev_case, list(DEMO_CASES.values())[0]).copy()

    case, customer, run_clicked = _sidebar(base)

    if case != prev_case:
        st.session_state["last_case"] = case
        st.rerun()

    # Run pipeline
    if run_clicked or "result" not in st.session_state:
        from decision_engine import run_flexiscore_pipeline
        from explainability  import generate_reason_codes
        with st.spinner("Đang chạy pipeline..."):
            result  = run_flexiscore_pipeline(customer, model)
            explain = generate_reason_codes(customer, result)
        st.session_state.update(result=result, explain=explain, customer=customer)
    else:
        result   = st.session_state["result"]
        explain  = st.session_state["explain"]
        customer = st.session_state["customer"]

    decision = result["decision"]
    dmeta    = DECISIONS.get(decision, DECISIONS["HUMAN_REVIEW"])

    # ── Decision banner ───────────────────────────────────────────────────────
    dc = dmeta["color"]
    st.markdown(
        f"<div style='background:{dc};color:white;padding:18px 24px;"
        f"border-radius:10px;margin:12px 0'>"
        f"<span style='font-size:1.8em'>{dmeta['emoji']}</span>&nbsp;&nbsp;"
        f"<strong style='font-size:1.3em'>{dmeta['label']}</strong>"
        f"<span style='opacity:0.7;margin-left:12px;font-size:0.9em'>[{decision}]</span><br>"
        f"<span style='font-size:0.92em;opacity:0.92'>{dmeta['desc']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Pipeline steps ────────────────────────────────────────────────────────
    st.markdown("**Pipeline thực thi:**")
    s1, s2, s3, s4 = st.columns(4)
    f0_ok = result["f0_status"] == "PASS"
    f1_ok = result["f1_status"] in ("PASS", "WARN")
    conf  = customer.get("data_confidence", 1.0)

    with s1:
        if f0_ok:
            st.success("**Stage 1 · F0**\n\nQuy tắc cứng\n\n✅ PASS")
        else:
            st.error("**Stage 1 · F0**\n\nQuy tắc cứng\n\n❌ FAIL")

    with s2:
        if not f0_ok:
            st.markdown(
                "<div style='background:#eee;padding:12px;border-radius:6px;"
                "text-align:center;color:#999;height:100%'>"
                "<strong>Stage 2 · F1</strong><br>Stress Test<br>— (bỏ qua)</div>",
                unsafe_allow_html=True,
            )
        elif f1_ok:
            st.success(
                f"**Stage 2 · F1**\n\nStress Test\n\n"
                f"{'✅ PASS' if result['f1_status']=='PASS' else '⚠️ WARN'}"
            )
        else:
            st.error("**Stage 2 · F1**\n\nStress Test\n\n❌ FAIL")

    with s3:
        score_color = (
            "success" if result["flexiscore"] >= 700 else
            "warning" if result["flexiscore"] >= 500 else "error"
        )
        note = (
            f"FlexiScore: **{result['flexiscore']:.0f}** / 1000\n\n"
            f"PD: **{result['pd']:.1%}** ({result['pd_band']})\n\n"
            f"Conf: **{conf:.0%}**"
        )
        if not f0_ok:
            st.markdown(
                "<div style='background:#eee;padding:12px;border-radius:6px;"
                "text-align:center;color:#999'>"
                "<strong>Stage 3 · Score/PD</strong><br>— (bỏ qua)</div>",
                unsafe_allow_html=True,
            )
        elif score_color == "success":
            st.success(f"**Stage 3 · Score/PD**\n\n{note}")
        elif score_color == "warning":
            st.warning(f"**Stage 3 · Score/PD**\n\n{note}")
        else:
            st.error(f"**Stage 3 · Score/PD**\n\n{note}")

    with s4:
        if decision == "AUTO_APPROVE":
            st.success(f"**Quyết định**\n\n{dmeta['emoji']} {dmeta['label']}")
        elif decision == "AUTO_REJECT":
            st.error(f"**Quyết định**\n\n{dmeta['emoji']} {dmeta['label']}")
        elif decision == "CREDIT_COACH":
            st.warning(f"**Quyết định**\n\n{dmeta['emoji']} {dmeta['label']}")
        else:
            st.info(f"**Quyết định**\n\n{dmeta['emoji']} {dmeta['label']}")

    st.divider()

    # ── 2-column layout ───────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: Customer / Decision view ────────────────────────────────────────
    with left:
        st.subheader("👤 Hồ sơ khách hàng")
        ctype_label = TYPE_LABELS.get(result["customer_type"], result["customer_type"])
        amt  = customer.get("requested_amount", 0)
        tenor= customer.get("requested_tenor_months", 0)
        inc  = customer.get("avg_monthly_income", 0)
        st.markdown(
            f"**{result['name']}** &nbsp;·&nbsp; {ctype_label}\n\n"
            f"Thu nhập TB: **{inc:,.0f} đ/tháng** &nbsp;·&nbsp; "
            f"Yêu cầu: **{amt/1e6:.0f}M / {tenor} tháng**\n\n"
            f"Risk tier: *{result['risk_tier']}*"
        )
        st.divider()

        # Decision-specific content
        if decision == "AUTO_APPROVE":
            st.success("Chúc mừng! Khoản vay được phê duyệt tự động.")
            rec = result.get("recommended_offer")
            if rec:
                st.markdown(f"**Offer: {rec.label}**")
                st.table(pd.DataFrame({
                    "": ["Số tiền", "Kỳ hạn", "Lịch trả", "Lãi suất",
                         "Góp ước tính", "Expected Profit (NH)"],
                    "Giá trị": [
                        f"{rec.amount:,.0f} đ",
                        f"{rec.tenor_months} tháng",
                        rec.schedule,
                        f"{rec.interest_rate:.0%} / năm",
                        f"{rec.monthly_payment:,.0f} đ / kỳ",
                        f"+{rec.expected_profit:,.0f} đ",
                    ],
                }).set_index(""))

        elif decision == "AUTO_REJECT":
            st.error("Hồ sơ không đủ điều kiện xử lý tự động.")
            for r in result.get("reason_codes", []):
                st.markdown(r)
            st.caption("Liên hệ cán bộ tín dụng để được tư vấn thêm.")

        elif decision == "CREDIT_COACH":
            st.warning("Hồ sơ chưa đủ điều kiện — nhưng có thể cải thiện trong 90 ngày.")
            for r in result.get("reason_codes", []):
                st.markdown(r)
            plan = result.get("credit_coach_plan", [])
            if plan:
                st.markdown("**Lộ trình cải thiện:**")
                for item in plan:
                    st.markdown(f"- {item}")

        elif decision == "HUMAN_REVIEW":
            st.info("Hồ sơ đang được chuyển sang thẩm định thêm.\n\nPhản hồi dự kiến 1–2 ngày làm việc.")
            for r in result.get("reason_codes", []):
                st.markdown(r)
            rec = result.get("recommended_offer")
            if rec and rec.ep_positive:
                st.markdown(
                    f"Cán bộ sẽ xem xét offer tiềm năng: "
                    f"**{rec.amount/1e6:.1f}M / {rec.tenor_months}T** "
                    f"(EP +{rec.expected_profit:,.0f}đ)."
                )

        st.divider()
        pos = explain.get("positive_factors", [])
        neg = explain.get("negative_factors", [])
        if pos:
            st.markdown("**Điểm mạnh hồ sơ**")
            for f in pos[:5]:
                st.markdown(f)
        if neg:
            st.markdown("**Cần cải thiện**")
            for f in neg[:5]:
                st.markdown(f)

    # ── RIGHT: Model / Risk view ──────────────────────────────────────────────
    with right:
        st.subheader("🏦 Kết quả mô hình")

        st.plotly_chart(_gauge_flexiscore(result["flexiscore"]), use_container_width=True)

        # Key metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("PD (Xác suất vỡ nợ)", f"{result['pd']:.1%}", delta=result["pd_band"], delta_color="off")
        m2.metric("F0 Hard Rules", "✅ PASS" if result["f0_status"]=="PASS" else "❌ FAIL")
        m3.metric("F1 Stress Test", "✅ PASS" if result["f1_status"]=="PASS" else
                  ("⚠️ WARN" if result["f1_status"]=="WARN" else "❌ FAIL"))

        m4, m5, m6 = st.columns(3)
        m4.metric("Data Confidence", f"{customer.get('data_confidence',0):.0%}")
        m5.metric("Graph Risk", result.get("graph_risk_label", "—"))
        m6.metric("DTI Stress", f"{result.get('dti_stress_ratio',0):.1%}", delta="ngưỡng 35%", delta_color="off")

        # Graph risk details
        with st.expander("🕸 Chi tiết Graph Risk"):
            for r in result.get("graph_reasons", []):
                st.markdown(f"- {r}")

        # F0 / F1 details
        with st.expander("🛡 Chi tiết F0 & F1"):
            f0r = result.get("f0_reasons", [])
            f1r = result.get("f1_reasons", [])
            if f0r:
                st.markdown("**F0 — Vi phạm:**")
                for r in f0r: st.markdown(f"- {r}")
            else:
                st.success("F0: Không có vi phạm.")
            if f1r:
                st.markdown("**F1 — Cảnh báo:**")
                for r in f1r: st.markdown(f"- {r}")
            else:
                st.success("F1: Không có vi phạm.")

        # Offers
        offers = result.get("offers", [])
        if offers:
            rec_label = result["recommended_offer"].label if result.get("recommended_offer") else ""
            with st.expander("💼 So sánh 3 Loan Offers"):
                rows = [{
                    "Offer":    o.label + (" ⭐" if o.label == rec_label else ""),
                    "Số tiền": f"{o.amount/1e6:.1f}M",
                    "Kỳ hạn":  f"{o.tenor_months}T",
                    "EP":      f"{o.expected_profit/1_000:,.0f}K đ",
                    "OK?":     "✅" if o.ep_positive else "❌",
                } for o in offers]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model info (collapsed) ────────────────────────────────────────────────
    st.divider()
    with st.expander("📈 Thông tin mô hình & ngưỡng quyết định"):
        if metrics:
            mc = st.columns(4)
            mc[0].metric("ROC-AUC",   metrics.get("roc_auc",   "—"))
            mc[1].metric("Accuracy",  metrics.get("accuracy",  "—"))
            mc[2].metric("Precision", metrics.get("precision", "—"))
            mc[3].metric("Recall",    metrics.get("recall",    "—"))
        else:
            st.caption("Model load từ cache — metrics không hiển thị.")
        st.markdown(
            "| Quyết định | Điều kiện |\n|---|---|\n"
            "| **AUTO_REJECT** | F0 fail: fraud_ring=1, device≥4, bad_neighbor≥3, "
            "income_cv>0.75, hoặc data_conf<40% |\n"
            "| **CREDIT_COACH** | F1 fail (income_cv>0.55 hoặc DTI>35%) "
            "hoặc data_confidence<60% |\n"
            "| **AUTO_APPROVE** | FlexiScore≥700 · PD<10% · "
            "data_confidence≥70% · EP>0 |\n"
            "| **HUMAN_REVIEW** | Tất cả còn lại — vùng xám |"
        )

    st.markdown(
        "<p style='text-align:center;color:#bbb;font-size:0.78em;margin-top:16px'>"
        "FlexiScore Demo · Synthetic data · Prototype only · "
        "Decisions are purely value-driven, not case-name-driven.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
