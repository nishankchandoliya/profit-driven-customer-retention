"""Train the final churn model and choose who to contact.

Steps:
1. Same 80/20 split as the notebooks. The test split is only used for the final numbers.
2. 5-fold cross-validation on the training split gives out-of-fold churn probabilities
   (each customer is scored by a model that did not see them).
3. The contact threshold with the highest expected profit on those out-of-fold probabilities is chosen.
4. The model is fitted on the whole training split and checked once on the test split.

Saves models/churn_model.joblib and models/model_info.json (settings, threshold, test results).
By default the extra feature groups are used (they helped in scripts/feature_check.py).

Usage:
    python scripts/train.py
    python scripts/train.py --success-rate 0.2 --offer-cost 20 --churn-rate 0.02
    python scripts/train.py --features original      # the model without the extra features
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from churn.data import load_data, split_train_test  # noqa: E402
from churn.models import SEED, load_params, make_xgb_pipeline  # noqa: E402
from churn.preprocessing import FEATURE_GROUPS  # noqa: E402
from churn.profit import (  # noqa: E402
    LIFETIME_MONTHS, OFFER_COST, SUCCESS_RATE, best_threshold, campaign_profit, prevalence_weights,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=ROOT / "data" / "telecom")
    parser.add_argument("--params", default=ROOT / "configs" / "xgb_tuned_params.json",
                        help="tuned settings from scripts/tune.py (falls back to the original settings)")
    parser.add_argument("--success-rate", type=float, default=SUCCESS_RATE,
                        help="share of offered churners who stay (assumption)")
    parser.add_argument("--offer-cost", type=float, default=OFFER_COST)
    parser.add_argument("--months", type=int, default=LIFETIME_MONTHS,
                        help="months of revenue kept when a churner is saved")
    parser.add_argument("--churn-rate", type=float, default=None,
                        help="real churn rate to plan for, e.g. 0.02 (default: the rate in the data, about 50%%)")
    parser.add_argument("--features", choices=["extra", "original"], default="extra",
                        help="extra = original features + all groups in FEATURE_GROUPS")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out-dir", default=ROOT / "models")
    args = parser.parse_args()

    if Path(args.params).exists():
        params = load_params(args.params)
        print(f"Using tuned settings from {args.params}")
    else:
        params = None
        print("No tuned settings found, using the original notebook settings")

    extra = tuple(FEATURE_GROUPS) if args.features == "extra" else ()
    df = load_data(args.data_dir)
    X_train, X_test, y_train, y_test = split_train_test(df)
    value_saved = X_train["rev_Mean"].mean() * args.months
    w_train = None if args.churn_rate is None else prevalence_weights(y_train, args.churn_rate)
    w_test = None if args.churn_rate is None else prevalence_weights(y_test, args.churn_rate)

    # 1) threshold from out-of-fold probabilities on the training split
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_xgb_pipeline(params, extra_features=extra), X_train, y_train, cv=cv,
                            method="predict_proba")[:, 1]
    threshold = best_threshold(y_train, oof, value_saved, args.offer_cost, args.success_rate, w_train)

    # 2) final model on the whole training split, checked once on the test split
    model = make_xgb_pipeline(params, extra_features=extra).fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    def per_1000(contact):
        total = campaign_profit(y_test, contact, value_saved, args.offer_cost, args.success_rate, w_test)
        n = len(y_test) if w_test is None else w_test.sum()
        return round(total / n * 1000, 2)

    contact = prob >= threshold
    results = {
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "accuracy_at_0.5": round(accuracy_score(y_test, (prob >= 0.5).astype(int)), 4),
        "oof_roc_auc": round(roc_auc_score(y_train, oof), 4),
        "contacted_share": round(float(np.average(contact, weights=w_test)), 4),
        "profit_per_1000_customers": per_1000(contact),
        "profit_per_1000_offer_everyone": per_1000(np.ones(len(prob), dtype=bool)),
        "profit_per_1000_threshold_0.5": per_1000(prob >= 0.5),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "churn_model.joblib")
    keep = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
            "colsample_bytree", "reg_lambda", "gamma", "random_state"]
    xgb_params = {k: v for k, v in model.named_steps["model"].get_params().items() if k in keep}
    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "xgb_params": xgb_params,
        "extra_features": list(extra),
        "threshold": threshold,
        "assumptions": {
            "success_rate": args.success_rate,
            "offer_cost": args.offer_cost,
            "months": args.months,
            "value_saved": round(value_saved, 2),
            "planning_churn_rate": args.churn_rate,
        },
        "test_results": results,
        "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__},
    }
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2, default=float) + "\n")

    print(f"Threshold chosen on out-of-fold data: {threshold:.3f}")
    for key, value in results.items():
        print(f"  {key:32s} {value}")
    print(f"Saved {out_dir / 'churn_model.joblib'} and {out_dir / 'model_info.json'}")


if __name__ == "__main__":
    main()
