"""Small checks of the cleaning rules on made-up data (the real data is not needed)."""
import numpy as np
import pandas as pd

from churn.preprocessing import ChurnPreprocessor


def make_rows(n_rows, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Customer_ID": np.arange(n_rows),
        "months": rng.integers(6, 40, n_rows),
        "avg3mou": rng.uniform(0, 500, n_rows),
        "avg6mou": rng.uniform(1, 500, n_rows),
        "avg3qty": rng.uniform(0, 100, n_rows),
        "avg6qty": rng.uniform(1, 100, n_rows),
        "avg3rev": rng.uniform(10, 80, n_rows),
        "avg6rev": rng.uniform(10, 80, n_rows),
        "change_mou": rng.normal(0, 50, n_rows),
        "change_rev": rng.normal(0, 5, n_rows),
        "eqpdays": rng.integers(0, 1500, n_rows).astype(float),
        "prizm_social_one": rng.choice(["S", "U", "T"], n_rows),
        "hnd_webcap": rng.choice(["WCMB", "WC"], n_rows),
        "area": rng.choice(["NORTH", "SOUTH"], n_rows),
        "churn": rng.integers(0, 2, n_rows),
    })


def test_medians_come_from_training_data_only():
    train = make_rows(200)
    test = make_rows(50, seed=1)
    test["eqpdays"] = np.nan
    test.loc[0, "eqpdays"] = 10_000.0  # one extreme value in the test rows only

    prep = ChurnPreprocessor().fit(train)
    out = prep.transform(test)

    assert np.allclose(out["eqpdays"].iloc[1:], train["eqpdays"].median())


def test_new_category_gets_minus_one_and_no_missing_values_remain():
    train = make_rows(100)
    test = make_rows(10, seed=2)
    test.loc[0, "area"] = "NEVER_SEEN"
    test.loc[1, "prizm_social_one"] = np.nan

    out = ChurnPreprocessor().fit(train).transform(test)

    assert out.loc[0, "area"] == -1
    assert not out.isna().any().any()


def test_new_accounts_use_the_three_month_average():
    rows = make_rows(20)
    rows.loc[0, ["months", "avg6mou"]] = [3, np.nan]

    out = ChurnPreprocessor().fit(rows).transform(rows)

    assert out.loc[0, "avg6mou"] == rows.loc[0, "avg3mou"]


def test_columns_and_order_are_stable():
    train = make_rows(100)
    prep = ChurnPreprocessor().fit(train)
    out = prep.transform(make_rows(30, seed=3))

    assert list(out.columns) == list(prep.get_feature_names_out())
    assert "Customer_ID" not in out.columns and "churn" not in out.columns
    assert {"usage_momentum_ratio", "revenue_momentum_ratio"} <= set(out.columns)


def test_extra_features_are_added_only_when_asked():
    from churn.preprocessing import FEATURE_GROUPS

    rows = make_rows(50)
    rng = np.random.default_rng(4)
    for col in ["phones", "actvsubs", "uniqsubs", "drop_blk_Mean", "attempt_Mean", "drop_vce_Mean",
                "plcd_vce_Mean", "complete_Mean", "custcare_Mean", "ovrrev_Mean", "rev_Mean", "totmrc_Mean",
                "mou_Mean", "avgmou", "avgrev"]:
        rows[col] = rng.uniform(0, 100, len(rows))
    rows.loc[0, "plcd_vce_Mean"] = np.nan  # missing input -> filled like any other number

    plain = ChurnPreprocessor().fit(rows).transform(rows)
    extra = ChurnPreprocessor(extra_features=tuple(FEATURE_GROUPS)).fit(rows).transform(rows)
    new_cols = set(extra.columns) - set(plain.columns)

    assert "usage_vs_lifetime" in new_cols and "dropped_voice_share" in new_cols
    assert len(new_cols) == sum(len(f(rows)) for f in FEATURE_GROUPS.values())
    assert not extra.isna().any().any()
