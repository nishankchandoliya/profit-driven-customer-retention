"""Check whether the extra features in churn.preprocessing.FEATURE_GROUPS help.

Settings: the original features, the original features + one group, and + all groups.
Each setting is scored with 5-fold cross-validation on the training split, with the tuned XGBoost
settings. The folds are the same for every setting, so the scores can be compared fold by fold.
The test split is not used here.

Writes outputs/features/feature_groups_cv.csv (one row per setting, ROC-AUC per fold).
Takes about 10 minutes on 2 CPU cores.

Usage:
    python scripts/feature_check.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from churn.data import load_data, split_train_test  # noqa: E402
from churn.models import SEED, cross_val_oof, load_params, make_xgb_pipeline  # noqa: E402
from churn.preprocessing import FEATURE_GROUPS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=ROOT / "data" / "telecom")
    parser.add_argument("--params", default=ROOT / "configs" / "xgb_tuned_params.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=ROOT / "outputs" / "features" / "feature_groups_cv.csv")
    args = parser.parse_args()

    df = load_data(args.data_dir)
    X_train, _, y_train, _ = split_train_test(df)
    params = load_params(args.params)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    settings = {"original": ()}
    settings.update({f"+ {group}": (group,) for group in FEATURE_GROUPS})
    settings["+ all groups"] = tuple(FEATURE_GROUPS)

    rows = []
    for name, groups in settings.items():
        aucs, _ = cross_val_oof(make_xgb_pipeline(params, extra_features=groups), X_train, y_train, cv)
        rows.append({"setting": name, **{f"fold_{i + 1}": round(a, 5) for i, a in enumerate(aucs)}})
        print(f"{name:20s} mean ROC-AUC {aucs.mean():.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
