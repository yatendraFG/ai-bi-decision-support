"""
Step 4a: Churn prediction - "which customers will leave in the next 90 days?"

Setup (time-based, leakage-free):
  * Snapshots every quarter. At each snapshot, features use only past data and the
    label = 1 if the customer churns within the following 90 days.
  * Train: 2025 snapshots | Validation (model choice + threshold): 2026-03-31
  * Test (reported, untouched): 2026-06-30
  * Score today (2026-09-30): ranked at-risk list with revenue at risk + plain-English reasons.

Models compared: Logistic Regression (baseline) vs HistGradientBoosting.
Business metrics: precision / recall in the top 10% riskiest customers (the list a
retention team can actually call), plus ROC-AUC and PR-AUC.

Outputs:
  models/churn_model.joblib
  data/processed/churn_scores.csv
  reports/churn_metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS, DATA_PROCESSED, REPORTS, RANDOM_STATE
from src.ml.features import load_raw, customer_features

LABEL_DAYS = 90
TRAIN_SNAPSHOTS = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]
VALID_SNAPSHOT = "2026-01-01"
TEST_SNAPSHOT = "2026-06-30"
SCORE_SNAPSHOT = "2026-10-01"   # "today": day after the last data point

CAT = ["segment", "plan", "region"]
NUM = ["recency_days", "orders_90d", "orders_180d", "orders_365d", "revenue_90d",
       "revenue_365d", "orders_prev90d", "orders_lifetime", "revenue_lifetime",
       "avg_order_value", "avg_discount", "n_categories", "tickets_90d", "tickets_365d",
       "neg_tickets_90d", "neg_tickets_365d", "high_prio_tickets_90d", "order_trend",
       "revenue_trend", "neg_ticket_share_365d", "tenure_days"]


def labelled(snapshot, sales, cust, tickets):
    f = customer_features(snapshot, sales, cust, tickets)
    snap = pd.Timestamp(snapshot)
    cd = f["customer_id"].map(cust.set_index("customer_id")["churn_date"])
    f["churned"] = ((cd >= snap) & (cd < snap + pd.Timedelta(days=LABEL_DAYS))).astype(int)
    return f


def build_models():
    pre_lr = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    lr = Pipeline([("pre", pre_lr),
                   ("clf", LogisticRegression(max_iter=2000, C=0.5))])
    pre_gb = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
    gb = Pipeline([("pre", pre_gb),
                   ("clf", HistGradientBoostingClassifier(
                       max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                       min_samples_leaf=40, l2_regularization=1.0,
                       random_state=RANDOM_STATE))])
    return {"logistic_regression": lr, "gradient_boosting": gb}


def evaluate(y, p, top_frac=0.10):
    k = max(1, int(len(y) * top_frac))
    top = np.argsort(-p)[:k]
    base = y.mean()
    prec = y[top].mean()
    return {"roc_auc": round(roc_auc_score(y, p), 3),
            "pr_auc": round(average_precision_score(y, p), 3),
            "churn_rate": round(float(base), 3),
            "mean_predicted_prob": round(float(p.mean()), 3),   # calibration check

            "precision_top10pct": round(float(prec), 3),
            "recall_top10pct": round(float(y[top].sum() / max(y.sum(), 1)), 3),
            "lift_top10pct": round(float(prec / base), 2)}


def reasons(row) -> str:
    """Plain-English risk reasons (used later by the AI agent and dashboard)."""
    r = []
    if row["order_trend"] < 0.6 and row["orders_prev90d"] > 0:
        r.append(f"orders down {100 * (1 - row['order_trend']):.0f}% vs previous quarter")
    if row["recency_days"] > 90:
        r.append(f"no purchase for {int(row['recency_days'])} days")
    if row["neg_tickets_90d"] >= 1:
        r.append(f"{int(row['neg_tickets_90d'])} negative support ticket(s) in last 90 days")
    if row["high_prio_tickets_90d"] >= 1:
        r.append("unresolved high-priority complaint")
    if row["plan"] == "Basic":
        r.append("on Basic plan")
    return "; ".join(r[:3]) or "general decline in engagement"


def main():
    print("[Step 4a] Churn prediction")
    sales, cust, tickets = load_raw()
    train = pd.concat([labelled(s, sales, cust, tickets) for s in TRAIN_SNAPSHOTS])
    valid = labelled(VALID_SNAPSHOT, sales, cust, tickets)
    test = labelled(TEST_SNAPSHOT, sales, cust, tickets)
    print(f"  train rows {len(train):,} (churn {train['churned'].mean():.1%}) | "
          f"valid {len(valid):,} | test {len(test):,} (churn {test['churned'].mean():.1%})")

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(train[NUM + CAT], train["churned"])
        pv = model.predict_proba(valid[NUM + CAT])[:, 1]
        results[name] = {"valid": evaluate(valid["churned"].values, pv)}
        fitted[name] = model
        print(f"    {name:20s} valid ROC-AUC {results[name]['valid']['roc_auc']:.3f} "
              f"| PR-AUC {results[name]['valid']['pr_auc']:.3f}")

    best = max(results, key=lambda n: results[n]["valid"]["pr_auc"])
    print(f"  Selected on validation: {best}")

    # refit best on train + valid, report once on the untouched test snapshot
    final = build_models()[best]
    trval = pd.concat([train, valid])
    final.fit(trval[NUM + CAT], trval["churned"])
    pt = final.predict_proba(test[NUM + CAT])[:, 1]
    test_metrics = evaluate(test["churned"].values, pt)
    for name, m in fitted.items():  # also show both models on test for transparency
        results[name]["test"] = evaluate(test["churned"].values, m.predict_proba(test[NUM + CAT])[:, 1])
    print(f"  TEST ({TEST_SNAPSHOT}): ROC-AUC {test_metrics['roc_auc']:.3f} | PR-AUC "
          f"{test_metrics['pr_auc']:.3f} | top-10% precision {test_metrics['precision_top10pct']:.1%} "
          f"(base rate {test_metrics['churn_rate']:.1%}, lift {test_metrics['lift_top10pct']}x) "
          f"| catches {test_metrics['recall_top10pct']:.0%} of churners")
    print(f"  Calibration: predicted churn {test_metrics['mean_predicted_prob']:.1%} "
          f"vs actual {test_metrics['churn_rate']:.1%}")

    imp = permutation_importance(final, test[NUM + CAT], test["churned"], scoring="average_precision",
                                 n_repeats=5, random_state=RANDOM_STATE)
    importance = (pd.Series(imp.importances_mean, index=NUM + CAT)
                  .sort_values(ascending=False).head(8).round(4).to_dict())

    # refit on everything with known labels, then score today's customers
    all_lab = pd.concat([trval, test])
    final.fit(all_lab[NUM + CAT], all_lab["churned"])
    now = customer_features(SCORE_SNAPSHOT, sales, cust, tickets)
    now["churn_probability"] = final.predict_proba(now[NUM + CAT])[:, 1].round(3)
    # bands by rank: "High" = top 10% (the call list we evaluated above), "Medium" = next 15%
    pct = now["churn_probability"].rank(pct=True, method="first")
    now["risk_band"] = np.select([pct > 0.90, pct > 0.75], ["High", "Medium"], "Low")
    now["revenue_at_risk_365d"] = (now["churn_probability"] * now["revenue_365d"]).round(0)
    now["risk_reasons"] = now.apply(reasons, axis=1)
    out = now[["customer_id", "segment", "plan", "region", "churn_probability", "risk_band",
               "revenue_365d", "revenue_at_risk_365d", "recency_days", "orders_90d",
               "order_trend", "neg_tickets_90d", "risk_reasons"]]
    out = out.round({"order_trend": 2}).sort_values("churn_probability", ascending=False)
    out.to_csv(DATA_PROCESSED / "churn_scores.csv", index=False)
    joblib.dump({"model": final, "num": NUM, "cat": CAT, "model_type": best},
                MODELS / "churn_model.joblib")

    summary = {
        "label": f"churns within {LABEL_DAYS} days of snapshot",
        "selected_model": best,
        "comparison": results,
        "test_snapshot": TEST_SNAPSHOT,
        "test_metrics_selected": test_metrics,
        "top_drivers_permutation_importance": importance,
        "scored_at": SCORE_SNAPSHOT,
        "customers_scored": int(len(out)),
        "high_risk_customers": int((out["risk_band"] == "High").sum()),
        "revenue_at_risk_inr": float(out["revenue_at_risk_365d"].sum()),
    }
    (REPORTS / "churn_metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Top drivers: {', '.join(list(importance)[:5])}")
    print(f"  Today: {summary['high_risk_customers']} high-risk customers, "
          f"INR {summary['revenue_at_risk_inr']/1e5:,.1f} lakh annual revenue at risk")
    return summary


if __name__ == "__main__":
    main()
