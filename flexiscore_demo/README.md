# FlexiScore — Alternative Credit Scoring Demo

> **Prototype minh họa** — không phải hệ thống vận hành tín dụng.
> Dữ liệu 100% giả lập. Mọi quyết định tính từ giá trị tham số, không hard-code theo tên use-case.

---

## FlexiScore là gì?

Mô hình chấm điểm tín dụng thay thế cho **khách hàng hồ sơ mỏng** (thin-file): tài xế công nghệ, seller online, freelancer, tiểu thương — những người không có lịch sử tín dụng ngân hàng.

Thay vì bảng lương tĩnh, FlexiScore phân tích 4 nguồn dữ liệu số:

| Nguồn | Thông tin |
|-------|-----------|
| 📱 Grab/Shopee API | Doanh thu, ngày hoạt động, rating |
| 🏦 VietQR / Ngân hàng | Thu nhập, biến động dòng tiền |
| ⚡ EVN / VNPT | Kỷ luật thanh toán hóa đơn |
| 🕸 Neo4j Graph DB | Phát hiện gian lận, mạng lưới rủi ro |

---

## Cách chạy

```bash
pip install -r requirements.txt
streamlit run app.py        # Dashboard
python test_pipeline.py     # Kiểm thử tự động
```

Lần đầu chạy tự sinh dataset + train model (~30–60 giây). Từ lần 2 load từ cache.

---

## Pipeline 6 stage

```
Input → [F0 Hard Rules] → [F1 Stress Test] → [FlexiScore] → [PD Model] → [EP Engine] → Quyết định
```

| Stage | Mô tả |
|-------|-------|
| F0 Hard Rules | Chặn ngay nếu: fraud_ring, device≥4, bad_neighbor≥3, income_cv>0.75, data_conf<40% |
| F1 Stress Test | DTI với thu nhập giảm 30%; income_cv>0.55 → FAIL |
| FlexiScore | Thang 0–1000, 5 nhóm tiêu chí (Income 35%, Graph 25%, Digital 20%, Finance 15%, Platform 5%) |
| PD Model | LightGBM — xác suất vỡ nợ |
| EP Engine | Expected Profit = Revenue − EL(PD×LGD×EAD) − CoF − OpEx |
| Quyết định | 4 nhánh thuần giá trị |

---

## 4 Quyết định đầu ra

```
F0 fail?  ──YES──→  AUTO_REJECT
    │
   NO
    ▼
F1 fail HOẶC data_conf < 60%?  ──YES──→  CREDIT_COACH
    │
   NO
    ▼
FlexiScore≥700 VÀ PD<10% VÀ data_conf≥70% VÀ EP>0?  ──YES──→  AUTO_APPROVE
    │
   NO
    ▼
HUMAN_REVIEW
```

| Quyết định | Điều kiện trigger |
|------------|------------------|
| **AUTO_REJECT** | Bất kỳ vi phạm F0 nào |
| **CREDIT_COACH** | F1 fail (income_cv>0.55 hoặc DTI>35%) hoặc data_confidence<60% |
| **AUTO_APPROVE** | FlexiScore≥700 · PD<10% · conf≥70% · EP>0 |
| **HUMAN_REVIEW** | Pass gates, nhưng chưa đủ tiêu chí auto-approve |

---

## 4 Demo Cases

Use-case chỉ là preset nạp giá trị ban đầu — không hard-code quyết định theo tên.

| Use-case | Profile | Expected |
|----------|---------|---------|
| Happy Path — Nguyễn Văn A | Gig worker, thu nhập ổn (cv=0.15), graph sạch | AUTO_APPROVE |
| Risk-First — Trần Thị B | Seller, fraud_ring=1, device=4, neighbor=3 | AUTO_REJECT |
| Credit Coach — Lê Văn C | Freelancer, income_cv=0.65 → F1 fail | CREDIT_COACH |
| Vùng xám — Nguyễn Thị D | Tiểu thương, score=693<700, PD=10.9%>10% | HUMAN_REVIEW |

---

## Cấu trúc project

```
flexiscore_demo/
├── app.py               # Streamlit dashboard
├── decision_engine.py   # Pipeline + 4-nhánh decision tree
├── risk_gates.py        # F0 Hard Rules + F1 Stress Test
├── scoring_model.py     # FlexiScore formula + LightGBM PD model
├── graph_risk.py        # Graph risk (NetworkX mock Neo4j)
├── expected_profit.py   # EP Engine, 3 loan offers
├── explainability.py    # Reason codes (Vietnamese)
├── data_generator.py    # 3.000 synthetic records
├── test_pipeline.py     # 10 test scenarios
└── requirements.txt
```

---

## Giới hạn & Roadmap

**Giới hạn prototype:**
- Dữ liệu và phân phối giả lập — chưa reflect thực tế gig workers VN
- Thresholds (700, 10%, 70%...) cần calibrate trên dữ liệu thật
- Graph risk là NetworkX mock, không phải Neo4j production

**Roadmap production:** Data ingestion (Kafka) → Model retraining (real labels) → Threshold calibration → Backtest (vintage, KS, PSI) → Shadow mode → Champion/Challenger → Monitoring
