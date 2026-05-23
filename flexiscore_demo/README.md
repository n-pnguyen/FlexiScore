# FlexiScore — Alternative Credit Scoring Demo

## FlexiScore là gì?

FlexiScore là mô hình chấm điểm tín dụng sử dụng **dữ liệu phi truyền thống** (alternative data) cho nhóm khách hàng "hồ sơ mỏng" (thin-file) — những người không có lịch sử tín dụng ngân hàng truyền thống:

- Tài xế công nghệ (Grab, Gojek)
- Seller online (Shopee, TikTok Shop)
- Freelancer / creative worker
- Tiểu thương / SME nhỏ
- Khách hàng hồ sơ mỏng lần đầu vay

Thay vì dùng bảng lương tĩnh hay lịch sử ngân hàng, FlexiScore phân tích:
- **Dòng tiền số** (VietQR, ví điện tử)
- **Hành vi gig/nền tảng** (Grab, Shopee ratings)
- **Kỷ luật thanh toán** (EVN, VNPT, BHXH)
- **Rủi ro mạng lưới** (Graph AI — Neo4j)

---

## ⚠️ Lưu ý về dữ liệu

**Prototype này dùng 100% synthetic data (dữ liệu giả lập).** Chưa sử dụng dữ liệu thật từ SHB, Grab, Gojek, VietQR, EVN, VNPT, hay BHXH. Mục tiêu là chứng minh **logic end-to-end** của pipeline.

---

## Mục tiêu Demo

### Use Case 1 — Happy Path (Nguyễn Văn A, Tài xế Grab)
- Hồ sơ trắng — chưa từng vay ngân hàng
- Thu nhập gig ổn định (income_cv thấp), graph sạch
- **Kết quả:** Auto Approve với offer tối ưu 15M/9T/bi-weekly — EP dương

### Use Case 2 — Risk-First (Trần Thị B, Seller Online)
- Dòng tiền đẹp nhưng fraud graph rất cao
- fraud_ring_flag = True, shared_device = 4, bad_neighbor = 3
- **Kết quả:** Fraud Reject — không dùng ML để override, hạn mức = 0

### Use Case 3 — Credit Coach (Lê Văn C, Freelancer)
- Thu nhập biến động (income_cv = 0.65), F1 stress test fail
- **Kết quả:** Credit Coach với lộ trình 90 ngày cải thiện hồ sơ

---

## Cách chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Chạy dashboard
streamlit run app.py
```

Lần đầu chạy sẽ sinh dataset và train model tự động (~30 giây).

---

## Cấu trúc project

```
flexiscore_demo/
├── app.py               # Streamlit dashboard (entry point)
├── data_generator.py    # Sinh 3.000 khách synthetic
├── scoring_model.py     # Train LightGBM/GBM + FlexiScore formula
├── risk_gates.py        # F0 Hard Rules + F1 Stress Test
├── graph_risk.py        # Graph risk simulation (NetworkX)
├── expected_profit.py   # EP Engine + Adaptive Loan Architect
├── decision_engine.py   # Decision pipeline tổng hợp
├── explainability.py    # SHAP / rule-based reason codes
├── requirements.txt
└── README.md
```

---

## Pipeline 6 phase

```
Input KH → [F0/F1 Risk Gates] → [LightGBM PD] → [Graph Risk] → [EP Engine] → [Decision] → [Explanation]
```

| Phase | Mô tả |
|-------|-------|
| F0 Hard Rules | Chặn fraud/CV cao trước ML — không override |
| F1 Stress Test | DTI test với income giảm 30% |
| FlexiScore | 5 nhóm tiêu chí, thang 0–1000 |
| PD Model | LightGBM calibrated probability |
| Graph Risk | NetworkX mock — fraud ring, device/IP sharing |
| EP Engine | Expected Profit = Revenue − EL − CoF − OpEx |
| Decision | Auto Approve / Human Review / Credit Coach / Fraud Reject |

---

## Giới hạn của prototype

1. **Dữ liệu giả lập** — chưa reflect distribution thực của gig workers VN
2. **Model chưa calibrated** với dữ liệu thật — PD chỉ mang tính minh họa
3. **Graph risk là mock** — không phải Neo4j production với real transaction graph
4. **EP parameters** (LGD, CoF, OpEx) là giả định — cần điều chỉnh theo portfolio thực
5. **Không có backtesting** với vintage data thực

---

## Roadmap Production

Khi có dữ liệu SHB đã ẩn danh:

1. **Data ingestion** — Kafka pipeline với consent gateway OAuth2
2. **Model retraining** — Train trên real default labels, Platt scaling calibration
3. **Backtest** — Vintage analysis, KS test, PSI monitoring
4. **Shadow mode** — Chạy song song với scorecard hiện tại, so sánh Gini
5. **Champion/Challenger** — A/B test 10% traffic trước khi deploy full
6. **Production monitoring** — Data drift detection, model performance decay alert

---

## Scoring Weights

| Nhóm | Trọng số | Features chính |
|------|----------|----------------|
| Income Stability | 35% | income_cv, active_days, cashflow_drop |
| Graph Safety | 25% | fraud_ring, shared_device, bad_neighbor, trust_score |
| Digital Transactions | 20% | bill_on_time, wallet_consistency, surplus_ratio |
| Financial Commitment | 15% | payment_delay, late_count, data_confidence |
| Platform Behavior | 5% | rating, cancel_rate, completion_rate |
