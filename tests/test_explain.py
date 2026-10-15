import numpy as np
import pandas as pd

from churn.explain import DEFAULT_OFFER, NO_USAGE_NOTE, OFFERS, explain_customers
from churn.models import DeviceAgeRule


def _display():
    return pd.DataFrame({
        "eqpdays": [500, 100, 300],
        "change_mou": [-50.0, 20.0, 0.0],
        "mou_Mean": [300.0, 200.0, 0.0],
        "avg6mou": [280.0, 210.0, 0.0],
    })


def test_drivers_are_sorted_and_only_positive():
    shap_values = np.array([
        [0.30, 0.10, -0.05, 0.00],
        [-0.20, -0.10, -0.30, -0.01],
        [0.05, 0.20, 0.00, 0.00],
    ])
    out = explain_customers(shap_values, _display(), k=2)
    assert list(out.loc[0, ["driver_1", "driver_2"]]) == ["eqpdays", "change_mou"]
    assert out.loc[0, "driver_1_value"] == 500
    # nothing pushes customer 1 towards churn, so no drivers and the default offer
    assert out.loc[1, "driver_1"] == ""
    assert out.loc[1, "suggested_offer"] == DEFAULT_OFFER


def test_offer_follows_top_mapped_driver_and_no_usage_note():
    shap_values = np.array([
        [0.30, 0.10, 0.0, 0.0],
        [0.0, 0.0, 0.40, 0.0],   # mou_Mean has no mapped offer, so the default is used
        [0.05, 0.20, 0.0, 0.0],
    ])
    out = explain_customers(shap_values, _display(), k=2)
    assert out.loc[0, "suggested_offer"] == OFFERS["eqpdays"]
    assert out.loc[1, "suggested_offer"] == DEFAULT_OFFER
    # customer 2 has no usage at all, so the note replaces the offer
    assert out.loc[2, "suggested_offer"] == NO_USAGE_NOTE


def test_device_age_rule():
    X = pd.DataFrame({"eqpdays": [10, 365, 366, 900]})
    rule = DeviceAgeRule(days=365).fit(X)
    assert list(rule.predict(X)) == [0, 0, 1, 1]
    assert rule.predict_proba(X).shape == (4, 2)
