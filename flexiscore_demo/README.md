# FlexiScore — Alternative Credit Scoring Demo

> **Prototype minh họa — Không phải hệ thống vận hành.**
> Dùng để kiểm tra và trình bày logic mô hình FlexiScore chạy end-to-end.
> Mọi quyết định tính thuần túy từ giá trị tham số — không hard-code theo tên use-case.

---

## FlexiScore là gì?

FlexiScore là mô hình chấm điểm tín dụng thay thế (alternative credit scoring) cho nhóm khách hàng **hồ sơ mỏng (thin-file)** — những người không có lịch sử tín dụng ngân hàng truyền thống:

- Tài xế công nghệ (Grab, Gojek, Be)
- Seller online (Shopee, TikTok Shop, Lazada)
- Freelancer / creative worker
- Tiểu thương / hộ kinh doanh nhỏ
- Khách hàng lần đầu tiếp cận tín dụng

**Thay vì dùng bảng lương tĩnh hay lịch sử ngân hàng**, FlexiScore phân tích:

| Nguồn dữ liệu | Thông tin trích xuất |
|---------------|---------------------|
| Grab/Shopee API | Doanh thu, ngày hoạt động, rating, completion rate |
| VietQR / Bank statement | Thu nhập, biến động dòng tiền, tỷ lệ chi tiêu |
| EVN / VNPT / Hóa đơn | Kỷ luật thanh toán hóa đơn định kỳ |
| Neo4j Graph DB | Fraud ring detection, device/IP sharing, trust score |

---

## ⚠️ Giới hạn prototype

**Prototype này dùng 100% synthetic data (3.000 bản ghi giả lập).** Không sử dụng dữ liệu thật.

Mục tiêu duy nhất: **chứng minh logic end-to-end** của pipeline — từ nhận hồ sơ → tính điểm → ra quyết định → giải thích.

Cụ thể:
- FlexiScore formula và decision thresholds là **giả định minh họa**, chưa được calibrate
- PD model (LightGBM) chỉ mang tính minh họa — ROC-AUC ~0.60 trên synthetic data
- Graph risk là NetworkX mock, không phải Neo4j production
- EP parameters (LGD=65%, CoF=7%, lãi suất 20%/năm) cần điều chỉnh theo portfolio thực

---

## Cách chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Chạy dashboard
streamlit run app.py

# 3. Chạy test pipeline (optional)
python test_pipeline.py
```

Lần đầu chạy sẽ tự động sinh dataset và train model (~30–60 giây).
Từ lần 2 trở đi, model được load từ `models/pd_model.pkl`.

---

## Cấu trúc project

```
flexiscore_demo/
├── app.py               # Streamlit dashboard — entry point
├── data_generator.py    # Sinh 3.000 bản ghi synthetic (5 customer types)
├── scoring_model.py     # FlexiScore formula (0–1000) + LightGBM PD model
├── risk_gates.py        # F0 Hard Rules + F1 Stress Test
├── graph_risk.py        # Graph risk simulation (NetworkX mock Neo4j)
├── expected_profit.py   # EP Engine — 3 loan offers, Adaptive Loan Architect
├── decision_engine.py   # run_flexiscore_pipeline() — decision tree 7 nhánh
├── explainability.py    # Rule-based reason codes (tiếng Việt)
├── test_pipeline.py     # End-to-end tests — 10 scenarios
├── requirements.txt
├── models/              # pd_model.pkl (auto-generated)
└── README.md
```

---

## Pipeline 6 giai đoạn

```
Customer Input
     │
     ▼
[Stage 1] F0 Hard Rules        ← Chặn fraud/hard violations trước ML
     │ FAIL → FRAUD_REJECT | FRAUD_REVIEW
     │ PASS ↓
[Stage 2] FlexiScore            ← 5 nhóm tiêu chí, thang 0–1000
     │
[Stage 3] PD Model (LightGBM)  ← Xác suất vỡ nợ
     │
[Stage 4] Graph Risk           ← Fraud ring, device/IP sharing
     │
[Stage 5] EP Engine            ← Expected Profit = Revenue − EL − CoF − OpEx
     │                            3 loan offers: Original / 75% / 60%
     │
[Stage 6] Decision Tree        ← 7 nhánh thuần giá trị (không hard-code)
     │
     ▼
Decision + Reason Codes + Coach Plan (nếu có)
```

### Các giai đoạn chi tiết

| Stage | Module | Mô tả |
|-------|--------|-------|
| F0 Hard Rules | `risk_gates.py` | fraud_ring, shared_device≥4, bad_neighbor≥3, income_cv>0.75, data_conf<0.40 |
| F1 Stress Test | `risk_gates.py` | DTI test (thu nhập giảm 30%), income_cv>0.55 |
| FlexiScore | `scoring_model.py` | 5 nhóm: Income 35%, Graph 25%, Transactions 20%, Finance 15%, Platform 5% |
| PD Model | `scoring_model.py` | LightGBM (fallback GradientBoosting), 30+ features |
| Graph Risk | `graph_risk.py` | NetworkX graph — fraud ring, PageRank trust score |
| EP Engine | `expected_profit.py` | EP = Revenue − EL(PD×LGD×EAD) − CoF − OpEx; Adaptive Loan Architect |
| Decision | `decision_engine.py` | 7-nhánh decision tree |

---

## 7 loại quyết định

| Decision | Điều kiện | Ý nghĩa |
|----------|-----------|---------|
| `FRAUD_REJECT` | F0 fail + fraud_ring hoặc graph_risk>0.70 | Từ chối — phát hiện gian lận mạng lưới |
| `FRAUD_REVIEW` | F0 fail (vi phạm rule khác) | Cần cán bộ xác minh thủ công |
| `AUTO_APPROVE` | Score≥800, PD<10%, conf≥70%, EP>0 | Phê duyệt tự động, offer tối ưu |
| `APPROVE_WITH_OPTIMIZED_OFFER` | Score≥600, EP>0, conf≥70% | Duyệt với offer hệ thống tự điều chỉnh |
| `HUMAN_REVIEW` | Vùng xám (score 600–800, conf hoặc graph borderline) | Cán bộ tín dụng xem xét thêm |
| `CREDIT_COACH` | F1 fail (income_cv>0.55 hoặc DTI>35%) hoặc conf<60% | Lộ trình cải thiện 90 ngày |
| `REJECT_EP_NEGATIVE` | Score<450, không có offer nào EP>0 | Từ chối — không khả thi về EP |

---

## 4 ví dụ minh họa (Demo Cases)

Use-cases chỉ đóng vai trò **preset để nạp nhanh giá trị ban đầu**. Mọi quyết định đều chạy qua `run_flexiscore_pipeline()` và phụ thuộc 100% vào giá trị tham số — **không hard-code theo tên**.

| Use Case | Profile | Expected Decision |
|----------|---------|------------------|
| Happy Path — Nguyễn Văn A | Tài xế Grab, thu nhập ổn định, graph sạch | `AUTO_APPROVE` |
| Risk-First — Trần Thị B | Seller online, fraud_ring=1, graph_risk=0.91 | `FRAUD_REJECT` |
| Credit Coach — Lê Văn C | Freelancer, income_cv=0.65, cashflow giảm | `CREDIT_COACH` |
| Vùng xám — Nguyễn Thị D | Tiểu thương, borderline score/conf | `HUMAN_REVIEW` |

---

## Kiến trúc dữ liệu: Zone A vs Zone B

### Zone A — Raw Inputs (5 biến nhập tay)
- `customer_type`, `age`, `identity_verified`
- `requested_amount`, `requested_tenor_months`

### Zone B — Feature-Engineered (auto-compute từ data sources)
Trong production, các biến này được tính tự động từ API. Demo cho phép chỉnh tay để mô phỏng:

| Group | Nguồn | Features chính |
|-------|-------|----------------|
| B1 | Grab/Shopee API | active_days, rating_avg, cancel_rate, order_frequency |
| B2 | VietQR / Bank | avg_monthly_income, income_cv, cashflow_drop_30d |
| B3 | EVN / VNPT | bill_on_time_ratio, utility_payment_delay |
| B4 | Neo4j Graph | fraud_ring_flag, shared_device_count, trust_score |
| B5 | Auto-calc | data_confidence = f(B1, B2, B3, B4 weights + months) |

`data_confidence` tự tính theo trọng số: Platform 30% + Bank 30% + Utility 20% + Graph 20%.

---

## Scoring Weights (FlexiScore 0–1000)

| Nhóm | Trọng số | Features chính |
|------|----------|----------------|
| Income Stability | 35% | income_cv, active_days_per_week, cashflow_drop_30d |
| Graph Safety | 25% | fraud_ring_flag, shared_device_count, bad_neighbor_count, trust_score |
| Digital Transactions | 20% | bill_on_time_ratio, wallet_activity_consistency, monthly_surplus_ratio |
| Financial Commitment | 15% | late_payment_count_6m, avg_payment_delay_days, data_confidence |
| Platform Behavior | 5% | rating_avg, cancel_rate, completion_rate |

---

## Roadmap Production

Khi có dữ liệu thật và consent của khách hàng:

1. **Data ingestion** — Kafka pipeline với consent gateway (OAuth2 + PDPA compliance)
2. **Feature engineering** — Chuẩn hóa từng nguồn, xử lý missing, drift detection
3. **Model retraining** — Train trên real default labels, Platt scaling calibration
4. **Threshold calibration** — Điều chỉnh AUTO_APPROVE/HUMAN_REVIEW/CREDIT_COACH thresholds theo risk appetite
5. **Backtest** — Vintage analysis, KS test, PSI monitoring, Gini coefficient
6. **Shadow mode** — Chạy song song scorecard hiện tại 30–90 ngày, so sánh hit rate
7. **Champion/Challenger** — A/B test 10% traffic trước khi full deploy
8. **Production monitoring** — Data drift alert, model decay detection, bias audit định kỳ

---

## Yêu cầu hệ thống

```
Python >= 3.10
pandas, numpy, scikit-learn, lightgbm
shap, networkx, streamlit, plotly
joblib, scipy
```

Xem `requirements.txt` để biết phiên bản cụ thể.
