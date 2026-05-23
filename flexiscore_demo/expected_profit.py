"""
expected_profit.py — Expected Profit Engine + Adaptive Loan Architect (ALA).
"""
from dataclasses import dataclass, field
from typing import Optional


# ── Tham số ngân hàng (có thể override) ──────────────────────────────────────
INTEREST_RATE   = 0.20   # lãi suất năm
COST_OF_FUND    = 0.07   # chi phí vốn năm
LGD             = 0.65   # Loss Given Default
FIXED_OPEX      = 150_000  # VND / khoản vay
TARGET_EP_RATIO = 0.005  # EP tối thiểu = 0.5% của loan amount
DTI_LIMIT       = 0.35   # Debt-to-income tối đa (trên stressed income)


@dataclass
class LoanOffer:
    label: str
    amount: float
    tenor_months: int
    schedule: str          # "monthly" | "bi-weekly"
    interest_rate: float = INTEREST_RATE

    # Computed
    revenue: float = 0.0
    expected_loss: float = 0.0
    cost_of_fund: float = 0.0
    opex: float = FIXED_OPEX
    expected_profit: float = 0.0
    ep_positive: bool = False
    monthly_payment: float = 0.0
    reason: str = ""


def _compute_offer(
    label: str,
    amount: float,
    tenor: int,
    schedule: str,
    pd: float,
    stressed_income: float,
    interest_rate: float = INTEREST_RATE,
) -> LoanOffer:
    offer = LoanOffer(label=label, amount=amount, tenor_months=tenor,
                      schedule=schedule, interest_rate=interest_rate)

    offer.revenue        = amount * interest_rate * tenor / 12
    offer.expected_loss  = pd * LGD * amount
    offer.cost_of_fund   = amount * COST_OF_FUND * tenor / 12
    offer.opex           = FIXED_OPEX
    offer.expected_profit = offer.revenue - offer.expected_loss - offer.cost_of_fund - offer.opex

    payments_per_month = 2 if schedule == "bi-weekly" else 1
    total_payments = tenor * payments_per_month
    offer.monthly_payment = (amount / max(total_payments, 1)) * payments_per_month

    dti = offer.monthly_payment / max(stressed_income, 1)
    offer.ep_positive = (offer.expected_profit > TARGET_EP_RATIO * amount)

    if not offer.ep_positive:
        offer.reason = f"EP âm ({offer.expected_profit:,.0f}đ) — khoản vay chưa sinh lời kỳ vọng"
    elif dti > DTI_LIMIT:
        offer.reason = f"DTI stress cao ({dti:.0%}) — vượt ngưỡng 35%"
        offer.ep_positive = False
    else:
        offer.reason = f"EP dương ({offer.expected_profit:,.0f}đ), DTI stress {dti:.0%}"

    return offer


def compute_offers(customer: dict, pd: float) -> dict:
    """
    Tạo 3 loan offers và chọn offer tối ưu.
    Returns dict với danh sách offers và recommended offer.
    """
    amount  = float(customer.get("requested_amount", 10_000_000))
    tenor   = int(customer.get("requested_tenor_months", 12))
    income  = float(customer.get("avg_monthly_income", 10_000_000))
    stressed = income * 0.70

    offers = [
        _compute_offer("Original Offer",    amount,          tenor,             "monthly",    pd, stressed),
        _compute_offer("Optimized Offer A", amount * 0.75,   max(int(tenor * 0.75), 3), "bi-weekly", pd, stressed),
        _compute_offer("Optimized Offer B", amount * 0.60,   max(int(tenor * 0.50), 3), "bi-weekly", pd, stressed),
    ]

    # Chọn offer tốt nhất: ưu tiên EP dương + amount cao nhất
    viable = [o for o in offers if o.ep_positive]
    if viable:
        recommended = max(viable, key=lambda o: o.expected_profit)
    else:
        recommended = None

    return {
        "offers": offers,
        "recommended_offer": recommended,
        "has_viable_offer": recommended is not None,
    }
