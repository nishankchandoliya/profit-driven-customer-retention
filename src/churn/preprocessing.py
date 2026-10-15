"""Cleaning steps as a scikit-learn transformer.

The rules are the same as in the original notebook (notebooks/01). The difference: everything that is
learned from data (medians, category codes, constant columns) is learned in fit() from the training
rows only, so nothing from the test set leaks into training.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EPS = 1e-5
AVG6_FROM_AVG3 = [("avg6mou", "avg3mou"), ("avg6qty", "avg3qty"), ("avg6rev", "avg3rev")]
DROP_COLS = ["Customer_ID", "churn"]


def _device_features(df):
    return {
        # close to 1 = the customer still has the phone they joined with
        "device_age_vs_tenure": df["eqpdays"] / (df["months"] * 30.4 + 1),
        "phones_per_year": df["phones"] / (df["months"] / 12 + EPS),
        "active_subs_share": df["actvsubs"] / (df["uniqsubs"] + EPS),
    }


def _call_quality_features(df):
    return {
        "dropped_share": df["drop_blk_Mean"] / (df["attempt_Mean"] + 1),
        "dropped_voice_share": df["drop_vce_Mean"] / (df["plcd_vce_Mean"] + 1),
        "completed_share": df["complete_Mean"] / (df["attempt_Mean"] + 1),
        "care_calls_per_call": df["custcare_Mean"] / (df["plcd_vce_Mean"] + 1),
    }


def _billing_features(df):
    return {
        "overage_share": df["ovrrev_Mean"] / (df["rev_Mean"].abs() + 1),
        "base_plan_share": df["totmrc_Mean"] / (df["rev_Mean"].abs() + 1),
        "revenue_per_minute": df["rev_Mean"] / (df["mou_Mean"] + 1),
    }


def _usage_trend_features(df):
    return {
        # last months compared with the whole customer lifetime
        "usage_vs_lifetime": df["mou_Mean"] / (df["avgmou"] + 1),
        "revenue_vs_lifetime": df["rev_Mean"] / (df["avgrev"] + 1),
        "usage_change_share": df["change_mou"] / (df["avg3mou"] + 1),
    }


# optional extra features (notebooks/03); none of them are used unless asked for
FEATURE_GROUPS = {
    "device": _device_features,
    "call_quality": _call_quality_features,
    "billing": _billing_features,
    "usage_trend": _usage_trend_features,
}


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering + missing values + label encoding, fitted on training data only.

    encode_categoricals=True turns text columns into integer codes (for XGBoost).
    encode_categoricals=False keeps them as text (for one-hot encoding later, e.g. logistic regression).
    extra_features: names from FEATURE_GROUPS to add, e.g. ("device", "billing"). Empty = original features.
    """

    def __init__(self, encode_categoricals=True, extra_features=()):
        self.encode_categoricals = encode_categoricals
        self.extra_features = extra_features

    def _apply_rules(self, X: pd.DataFrame) -> pd.DataFrame:
        """Steps that do not need anything learned from data."""
        df = X.copy()

        # trend features: recent 3-month average vs 6-month average
        df["usage_momentum_ratio"] = df["avg3mou"] / (df["avg6mou"] + EPS)
        df["revenue_momentum_ratio"] = df["avg3rev"] / (df["avg6rev"] + EPS)
        # getattr: models saved before this option existed have no extra_features
        for group in getattr(self, "extra_features", ()):
            for name, values in FEATURE_GROUPS[group](df).items():
                df[name] = values
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # no earlier period to compare with -> no change
        for col in ["change_mou", "change_rev"]:
            df[col] = df[col].fillna(0.0)

        # accounts younger than 6 months: use the 3-month average as the 6-month average
        for avg6, avg3 in AVG6_FROM_AVG3:
            mask = df[avg6].isna() & (df["months"] < 6)
            df.loc[mask, avg6] = df.loc[mask, avg3]

        df["prizm_social_one"] = df["prizm_social_one"].fillna("Missing")
        df["hnd_webcap"] = df["hnd_webcap"].fillna("Unknown")
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.numeric_cols_] = df[self.numeric_cols_].fillna(self.medians_)
        for col in self.categorical_cols_:
            df[col] = df[col].fillna("Unknown").astype(str)
        return df

    def fit(self, X, y=None):
        df = self._apply_rules(X)
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = df.select_dtypes(include=["object", "string"]).columns.tolist()
        self.medians_ = df[self.numeric_cols_].median()

        df = self._fill_missing(df)
        self.categories_ = {col: sorted(df[col].unique()) for col in self.categorical_cols_}

        # a column with one value in the training data carries no information
        self.constant_cols_ = [col for col in df.columns if df[col].nunique() <= 1]
        self.feature_names_out_ = [col for col in df.columns if col not in self.constant_cols_]
        return self

    def clean(self, X):
        """Cleaned features with text columns still as text (useful for showing values to people)."""
        return self._fill_missing(self._apply_rules(X))[self.feature_names_out_]

    def transform(self, X):
        df = self.clean(X)
        if self.encode_categoricals:
            for col in self.categorical_cols_:
                if col in df.columns:
                    codes = {value: i for i, value in enumerate(self.categories_[col])}
                    # categories never seen in training get -1
                    df[col] = df[col].map(codes).fillna(-1).astype(int)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)
