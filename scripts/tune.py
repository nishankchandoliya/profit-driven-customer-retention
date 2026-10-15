"""Tune the XGBoost settings with Optuna, using cross-validation on the training split only.

The test split is never touched here. The original notebook settings are tried first, so the
result can only match or beat them on cross-validated ROC-AUC.

Writes:
    configs/xgb_tuned_params.json      best settings + their CV score
    outputs/tuning/optuna_trials.csv   every trial

Usage:
    python scripts/tune.py --n-trials 30 --folds 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import optuna  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from churn.data import load_data, split_train_test  # noqa: E402
from churn.models import SEED, V1_XGB_PARAMS, make_xgb_pipeline  # noqa: E402


def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=ROOT / "data" / "telecom")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--out", default=ROOT / "configs" / "xgb_tuned_params.json")
    parser.add_argument("--trials-csv", default=ROOT / "outputs" / "tuning" / "optuna_trials.csv")
    args = parser.parse_args()

    df = load_data(args.data_dir)
    X_train, _, y_train, _ = split_train_test(df)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    def objective(trial):
        model = make_xgb_pipeline(suggest_params(trial))
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # trial 0 = the original notebook settings (XGBoost defaults for the rest)
    study.enqueue_trial({**V1_XGB_PARAMS, "min_child_weight": 1.0, "reg_lambda": 1.0, "gamma": 0.0})

    start = time.time()
    study.optimize(objective, n_trials=args.n_trials,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d}  AUC {t.value:.4f}  best {s.best_value:.4f}")])

    best = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in study.best_params.items()}
    result = {
        "params": best,
        "cv_auc": round(study.best_value, 5),
        "original_settings_cv_auc": round(study.trials[0].value, 5),
        "n_trials": args.n_trials,
        "folds": args.folds,
        "seed": SEED,
        "minutes": round((time.time() - start) / 60, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    Path(args.trials_csv).parent.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params"))
    trials.rename(columns={"value": "cv_auc"}).to_csv(args.trials_csv, index=False)

    print(f"\nBest CV ROC-AUC {study.best_value:.4f} (original settings {study.trials[0].value:.4f})")
    print(f"Saved {args.out} and {args.trials_csv}")


if __name__ == "__main__":
    main()
