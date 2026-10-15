import numpy as np

from churn.profit import best_threshold, campaign_profit, prevalence_weights


def test_campaign_profit_by_hand():
    y = np.array([1, 1, 0, 0])
    contact = np.array([True, False, True, False])
    # contacted churner: 0.5 * 100 - 10 = 40; contacted stayer: -10; others: 0
    assert campaign_profit(y, contact, value_saved=100, offer_cost=10, success_rate=0.5) == 30


def test_nobody_is_contacted_when_offers_never_pay():
    y = np.array([1, 0, 0, 0])
    prob = np.array([0.9, 0.2, 0.1, 0.05])
    # 0.01 * 100 - 10 < 0 for every churner, so no threshold makes money
    assert best_threshold(y, prob, value_saved=100, offer_cost=10, success_rate=0.01) == 1.0


def test_best_threshold_contacts_only_the_high_scores():
    y = np.array([1, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    thr = best_threshold(y, prob, value_saved=100, offer_cost=10, success_rate=1.0)
    assert 0.3 < thr <= 0.8


def test_prevalence_weights_change_the_churn_rate():
    y = np.array([1] * 50 + [0] * 50)
    w = prevalence_weights(y, target_rate=0.02)
    assert np.isclose(np.sum(w * y) / np.sum(w), 0.02)
