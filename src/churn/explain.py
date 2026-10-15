"""Turning SHAP values into the top reasons and a suggested offer for each customer."""
import numpy as np
import pandas as pd

# same mapping as the original notebook (top SHAP driver -> retention action)
OFFERS = {
    "eqpdays": "Hardware upgrade voucher (zero-down, 24-month contract)",
    "ovrrev_Mean": "Upgrade to unlimited plan to eliminate bill shock",
    "drop_vce_Mean": "Proactive retention call with discretionary billing credit",
    "custcare_Mean": "Flag for specialist retention agent outreach",
    "change_mou": "Re-engagement offer (data bonus / loyalty reward)",
}
# the extra features (churn.preprocessing.FEATURE_GROUPS) get the action of the column they are built from
OFFERS.update({
    "device_age_vs_tenure": OFFERS["eqpdays"],
    "overage_share": OFFERS["ovrrev_Mean"],
    "dropped_voice_share": OFFERS["drop_vce_Mean"],
    "dropped_share": OFFERS["drop_vce_Mean"],
    "care_calls_per_call": OFFERS["custcare_Mean"],
    "usage_change_share": OFFERS["change_mou"],
    "usage_vs_lifetime": OFFERS["change_mou"],
})
DEFAULT_OFFER = "Review account and personalise offer"
NO_USAGE_NOTE = "No usage in the last 6 months: check the line is still in use before any offer"


def explain_customers(shap_values, display_values: pd.DataFrame, k=3) -> pd.DataFrame:
    """Top k drivers per customer (only features that push the churn score up) and a suggested offer.

    shap_values: array (customers x features) from shap.TreeExplainer on the encoded features.
    display_values: the same features before encoding (pipeline.named_steps["prep"].clean(X)),
    so text columns show their real values.
    """
    shap_values = np.asarray(shap_values)
    names = np.array(display_values.columns)
    values = display_values.to_numpy(dtype=object)
    rows = np.arange(len(display_values))
    order = np.argsort(-shap_values, axis=1)[:, :k]

    out = pd.DataFrame(index=display_values.index)
    offer = np.full(len(rows), DEFAULT_OFFER, dtype=object)
    offer_found = np.zeros(len(rows), dtype=bool)
    for j in range(k):
        idx = order[:, j]
        impact = shap_values[rows, idx]
        positive = impact > 0
        feature = np.where(positive, names[idx], "")
        out[f"driver_{j + 1}"] = feature
        out[f"driver_{j + 1}_value"] = np.where(positive, values[rows, idx], None)
        out[f"driver_{j + 1}_shap"] = np.where(positive, np.round(impact, 4), np.nan)
        # first driver (in order of impact) that has a mapped action decides the offer
        mapped = pd.Series(feature).map(OFFERS).to_numpy()
        use = ~offer_found & pd.notna(mapped)
        offer[use] = mapped[use]
        offer_found |= use

    no_usage = (display_values["mou_Mean"] == 0) & (display_values["avg6mou"] == 0)
    offer[no_usage.to_numpy()] = NO_USAGE_NOTE
    out["suggested_offer"] = offer
    return out
