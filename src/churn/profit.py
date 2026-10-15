"""Expected profit of a retention campaign, compared with running no campaign at all.

For every customer we decide: send the retention offer or not.
- Offer sent to a customer who would churn: with probability `success_rate` they stay,
  which is worth `value_saved`. The offer costs `offer_cost` either way.
- Offer sent to a customer who would stay anyway: we only pay `offer_cost`.
- No offer: nothing changes compared with no campaign, so the value is 0.

This is different from the original notebook, which assumed every offered churner stays and counted
both the saved revenue and the avoided acquisition cost for the same customer.
"""
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = np.round(np.arange(0.005, 1.0, 0.005), 3)

# assumptions from the original proposal
OFFER_COST = 35.0       # cost of one retention offer, in USD
LIFETIME_MONTHS = 12    # months of revenue kept when a churner is saved
# new assumption: share of offered churners who actually stay (the notebook shows 10% to 100%)
SUCCESS_RATE = 0.3


def customer_value(y_true, value_saved, offer_cost, success_rate):
    """Value of sending the offer to each customer (one number per customer)."""
    y = np.asarray(y_true)
    return np.where(y == 1, success_rate * value_saved - offer_cost, -offer_cost)


def campaign_profit(y_true, contact, value_saved, offer_cost, success_rate, weights=None):
    """Total profit (vs no campaign) when the customers marked in `contact` get the offer."""
    contact = np.asarray(contact, dtype=bool)
    w = np.ones(len(contact)) if weights is None else np.asarray(weights)
    return float(np.sum(w * contact * customer_value(y_true, value_saved, offer_cost, success_rate)))


def profit_curve(y_true, prob, value_saved, offer_cost, success_rate, weights=None,
                 thresholds=DEFAULT_THRESHOLDS):
    """Profit per 1,000 customers and share of customers contacted, for each threshold."""
    prob = np.asarray(prob)
    w = np.ones(len(prob)) if weights is None else np.asarray(weights)
    rows = []
    for thr in thresholds:
        contact = prob >= thr
        profit = campaign_profit(y_true, contact, value_saved, offer_cost, success_rate, w)
        rows.append({
            "threshold": thr,
            "contacted_share": float(np.sum(w * contact) / np.sum(w)),
            "profit_per_1000": profit / np.sum(w) * 1000,
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, prob, value_saved, offer_cost, success_rate, weights=None,
                   thresholds=DEFAULT_THRESHOLDS):
    """Threshold with the highest profit. If no threshold makes money, nobody should be contacted (returns 1.0)."""
    curve = profit_curve(y_true, prob, value_saved, offer_cost, success_rate, weights, thresholds)
    best = curve.loc[curve["profit_per_1000"].idxmax()]
    if best["profit_per_1000"] <= 0:
        return 1.0
    return float(best["threshold"])


def prevalence_weights(y_true, target_rate):
    """Row weights that make the churn rate look like `target_rate` (the data itself is about 50% churn)."""
    y = np.asarray(y_true)
    rate = y.mean()
    return np.where(y == 1, target_rate / rate, (1 - target_rate) / (1 - rate))
