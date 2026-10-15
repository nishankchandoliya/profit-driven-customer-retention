"""Score customers with the saved model.

For every customer the output CSV has: churn probability, risk rank (1 = highest), risk group
(1 = top 10% of scores), whether to contact them (threshold from scripts/train.py, or --budget),
the top 3 reasons from SHAP and a suggested offer.

Usage:
    python scripts/score_customers.py
    python scripts/score_customers.py --budget 1000 --explain-top 1000
    python scripts/score_customers.py --data-dir path/to/new_month --budget 5000

The SHAP reasons are the slow part (about 20 minutes for 100,000 customers on 2 CPU cores).
Use --explain-top N to compute them only for the N highest-risk customers.

The input folder needs Client.csv and Record.csv in the course format (a churn column is not needed).
Note: on the course data this also scores customers the model was trained on, so those scores
look better than they would on new customers.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn.data import ID_COL, TARGET, load_data  # noqa: E402
from churn.explain import explain_customers  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=ROOT / "data" / "telecom")
    parser.add_argument("--model", default=ROOT / "models" / "churn_model.joblib")
    parser.add_argument("--info", default=ROOT / "models" / "model_info.json")
    parser.add_argument("--output", default=ROOT / "outputs" / "scores" / "customer_scores.csv")
    parser.add_argument("--budget", type=int, default=None,
                        help="contact only the N highest-risk customers instead of using the threshold")
    parser.add_argument("--top-k", type=int, default=3, help="number of reasons per customer")
    parser.add_argument("--explain-top", type=int, default=None,
                        help="compute reasons only for the N highest-risk customers (default: everyone)")
    args = parser.parse_args()

    model = joblib.load(args.model)
    threshold = json.loads(Path(args.info).read_text())["threshold"]

    df = load_data(args.data_dir)
    X = df.drop(columns=[TARGET], errors="ignore")
    prob = model.predict_proba(X)[:, 1]

    rank = pd.Series(prob).rank(ascending=False, method="first").astype(int).to_numpy()

    # SHAP reasons, in chunks so progress can be shown
    explain = np.arange(len(X)) if args.explain_top is None else np.flatnonzero(rank <= args.explain_top)
    prep, xgb = model.named_steps["prep"], model.named_steps["model"]
    explainer = shap.TreeExplainer(xgb)
    X_explain = X.iloc[explain]
    chunks = []
    for start in range(0, len(X_explain), 10_000):
        part = X_explain.iloc[start:start + 10_000]
        chunks.append(explainer.shap_values(prep.transform(part)))
        print(f"  reasons: {start + len(part):,} / {len(X_explain):,} customers", flush=True)
    reasons = explain_customers(np.vstack(chunks), prep.clean(X_explain), k=args.top_k)
    reasons = reasons.reindex(X.index)
    reasons["suggested_offer"] = reasons["suggested_offer"].fillna("")

    if args.budget is not None:
        contact = rank <= args.budget
    else:
        contact = prob >= threshold

    scores = pd.DataFrame({
        ID_COL: df[ID_COL].to_numpy(),
        "churn_probability": np.round(prob, 4),
        "risk_rank": rank,
        "risk_group": np.ceil(rank / len(rank) * 10).astype(int),
        "contact": contact,
    })
    scores = pd.concat([scores, reasons.reset_index(drop=True)], axis=1).sort_values("risk_rank")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out, index=False)

    rule = f"top {args.budget}" if args.budget is not None else f"probability >= {threshold:.3f}"
    print(f"Scored {len(scores):,} customers; {int(contact.sum()):,} to contact ({rule})")
    print(f"Saved {out}\n")
    print(scores.head(5)[[ID_COL, "churn_probability", "driver_1", "driver_2", "suggested_offer"]].to_string(index=False))


if __name__ == "__main__":
    main()
