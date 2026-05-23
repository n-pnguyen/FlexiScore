"""
graph_risk.py — Mô phỏng Graph Risk bằng NetworkX.
"""
import networkx as nx
import numpy as np


def build_graph(customer: dict, seed: int = 42) -> nx.Graph:
    """
    Tạo graph đơn giản minh họa quan hệ customer – device – IP – merchant.
    Trả về networkx Graph để visualize.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    cid = customer.get("customer_id", "CUST_DEMO")
    G.add_node(cid, node_type="customer", label=customer.get("name", cid))

    # Thêm device nodes
    n_devices = int(customer.get("shared_device_count", 0)) + 1
    for i in range(n_devices):
        dev_id = f"DEV_{cid}_{i}"
        G.add_node(dev_id, node_type="device")
        G.add_edge(cid, dev_id, edge_type="USES_DEVICE")

    # Thêm IP nodes
    n_ips = int(customer.get("shared_ip_count", 0)) + 1
    for i in range(min(n_ips, 4)):
        ip_id = f"IP_{cid}_{i}"
        G.add_node(ip_id, node_type="ip")
        G.add_edge(cid, ip_id, edge_type="USES_IP")

    # Thêm bad neighbor links
    n_bad = int(customer.get("bad_neighbor_count", 0))
    for i in range(n_bad):
        bad_id = f"BAD_{i}"
        G.add_node(bad_id, node_type="bad_customer", fraud=True)
        # Kết nối qua thiết bị chung
        dev_id = f"DEV_{cid}_0"
        if G.has_node(dev_id):
            G.add_edge(bad_id, dev_id, edge_type="SHARES_DEVICE")

    # Fraud ring: thêm node fraud cluster
    if customer.get("fraud_ring_flag", 0):
        ring_id = "FRAUD_RING"
        G.add_node(ring_id, node_type="fraud_ring", fraud=True)
        G.add_edge(cid, ring_id, edge_type="IN_FRAUD_RING")

    return G


def compute_graph_risk(customer: dict) -> dict:
    """
    Tính graph_risk_score từ features graph và trả reason codes.
    """
    reasons = []

    fraud_flag    = bool(customer.get("fraud_ring_flag", 0))
    shared_dev    = int(customer.get("shared_device_count", 0))
    bad_neighbors = int(customer.get("bad_neighbor_count", 0))
    circ_ratio    = float(customer.get("circular_transaction_ratio", 0))
    trust         = float(customer.get("trust_score", 0.5))
    g_risk        = float(customer.get("graph_risk_score", 0.3))

    # Recalculate để đảm bảo nhất quán với demo cases
    computed_risk = np.clip(
        0.35 * fraud_flag
        + 0.25 * min(shared_dev / 8, 1)
        + 0.20 * min(bad_neighbors / 6, 1)
        + 0.15 * circ_ratio
        + 0.05 * (1 - trust),
        0, 1
    )
    # Dùng giá trị từ customer nếu có (demo cases đã preset), blend nhẹ
    final_risk = round(0.5 * g_risk + 0.5 * computed_risk, 4)

    if fraud_flag:
        reasons.append("Phát hiện liên kết với vòng gian lận (Fraud Ring)")
    if shared_dev >= 4:
        reasons.append(f"Thiết bị dùng chung với {shared_dev} hồ sơ khác — nghi ngờ gian lận")
    elif shared_dev >= 2:
        reasons.append(f"Thiết bị dùng chung với {shared_dev} hồ sơ")
    if bad_neighbors >= 3:
        reasons.append(f"Có {bad_neighbors} liên kết tới hồ sơ nợ xấu/gian lận")
    elif bad_neighbors >= 1:
        reasons.append(f"Có {bad_neighbors} liên kết tới hồ sơ rủi ro")
    if circ_ratio > 0.20:
        reasons.append(f"Tỷ lệ giao dịch vòng cao ({circ_ratio:.0%}) — dấu hiệu đảo tiền")
    if trust >= 0.80:
        reasons.append(f"Điểm tin cậy mạng lưới cao ({trust:.2f}) — graph an toàn")
    elif trust < 0.50:
        reasons.append(f"Điểm tin cậy mạng lưới thấp ({trust:.2f})")

    if not reasons:
        reasons.append("Graph sạch — không phát hiện rủi ro mạng lưới")

    # Risk label
    if final_risk >= 0.70 or fraud_flag:
        risk_label = "HIGH"
    elif final_risk >= 0.40:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    return {
        "graph_risk_score": final_risk,
        "trust_score":       round(trust, 4),
        "fraud_ring_flag":   fraud_flag,
        "graph_risk_label":  risk_label,
        "graph_reasons":     reasons,
    }
