"""
Step 4b: Customer segmentation (RFM + KMeans).

  * R = recency (days since last order), F = orders in last 365 days,
    M = revenue in last 365 days, plus average order value.
  * Log-transform + standardise, pick k (3-6) by silhouette score.
  * Each cluster gets a business-friendly name and an action for the team.
  * Joined with churn scores into one customer_360.csv (used by the agent later).

Outputs:
  models/segmentation.joblib
  data/processed/customer_segments.csv
  data/processed/customer_360.csv
  reports/segment_profile.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import MODELS, DATA_PROCESSED, REPORTS, RANDOM_STATE
from src.ml.features import load_raw, customer_features

SNAPSHOT = "2026-10-01"
RFM = ["recency_days", "orders_365d", "revenue_365d", "avg_order_value"]

ACTIONS = {
    "Champions": "Reward & upsell: loyalty perks, early access, premium plan offers",
    "Big-ticket buyers": "Account management: dedicated contact, bulk pricing, quarterly reviews",
    "Loyal regulars": "Cross-sell new categories, referral programme",
    "Occasional buyers": "Nudge frequency: personalised offers, reminders around festive season",
    "Slipping away": "Win-back campaign now: discount + support follow-up on open complaints",
}


def name_clusters(profile: pd.DataFrame) -> dict:
    """Map cluster id -> business name using the cluster's RFM averages."""
    names, left = {}, profile.copy()
    # 1. one clearly "gone quiet" cluster
    rec = left["recency_days"]
    if rec.max() > 1.5 * rec.median():
        names[rec.idxmax()] = "Slipping away"
        left = left.drop(rec.idxmax())
    # 2. best customers: most revenue among those buying at least as often as typical
    freq_ok = left[left["orders_365d"] >= left["orders_365d"].median()]
    cid = (freq_ok if len(freq_ok) else left)["revenue_365d"].idxmax()
    names[cid] = "Champions"
    left = left.drop(cid)
    # 3. few but large orders
    if len(left):
        cid = left["avg_order_value"].idxmax()
        names[cid] = "Big-ticket buyers"
        left = left.drop(cid)
    # 4. buy often, smaller baskets
    if len(left):
        cid = left["orders_365d"].idxmax()
        names[cid] = "Loyal regulars"
        left = left.drop(cid)
    # 5. everyone else, ordered by value
    for i, cid in enumerate(left.sort_values("revenue_365d", ascending=False).index):
        names[cid] = "Occasional buyers" + ("" if i == 0 else f" (tier {i + 1})")
    return names


def main():
    print("[Step 4b] Customer segmentation")
    sales, cust, tickets = load_raw()
    f = customer_features(SNAPSHOT, sales, cust, tickets)
    f = f[f["orders_lifetime"] > 0].copy()            # need purchase history to segment

    X = f[RFM].copy()
    X["recency_days"] = X["recency_days"].clip(upper=730)
    X = np.log1p(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    scores = {}
    for k in range(3, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        scores[k] = silhouette_score(Xs, km.labels_, sample_size=2000, random_state=RANDOM_STATE)
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
    f["cluster"] = km.labels_
    print("  silhouette by k: " + ", ".join(f"k={k}: {s:.3f}" for k, s in scores.items())
          + f"  -> using k={best_k}")

    profile = f.groupby("cluster")[RFM].mean()
    names = name_clusters(profile)
    f["segment_name"] = f["cluster"].map(names)
    f["recommended_action"] = f["segment_name"].str.replace(r" \(.*\)", "", regex=True).map(ACTIONS)

    summary = (f.groupby("segment_name")
                 .agg(customers=("customer_id", "size"),
                      avg_recency_days=("recency_days", "mean"),
                      avg_orders_365d=("orders_365d", "mean"),
                      avg_revenue_365d=("revenue_365d", "mean"),
                      avg_order_value=("avg_order_value", "mean"),
                      total_revenue_365d=("revenue_365d", "sum"),
                      top_customer_type=("segment", lambda s: s.mode()[0]))
                 .round(1).sort_values("total_revenue_365d", ascending=False))
    summary["revenue_share_pct"] = (100 * summary["total_revenue_365d"]
                                    / summary["total_revenue_365d"].sum()).round(1)
    print(summary[["customers", "avg_orders_365d", "avg_revenue_365d", "avg_recency_days",
                   "revenue_share_pct", "top_customer_type"]].to_string())

    seg_out = f[["customer_id", "segment", "region", "segment_name", "recommended_action"] + RFM]
    seg_out.to_csv(DATA_PROCESSED / "customer_segments.csv", index=False)
    joblib.dump({"scaler": scaler, "kmeans": km, "features": RFM, "names": names},
                MODELS / "segmentation.joblib")

    # customer 360 = segment + churn risk in one row per customer
    churn_path = DATA_PROCESSED / "churn_scores.csv"
    if churn_path.exists():
        churn = pd.read_csv(churn_path)[["customer_id", "churn_probability", "risk_band",
                                         "revenue_at_risk_365d", "risk_reasons"]]
        c360 = seg_out.merge(churn, on="customer_id", how="left")
        c360.to_csv(DATA_PROCESSED / "customer_360.csv", index=False)
        cross = (c360.groupby("segment_name")
                     .agg(high_risk=("risk_band", lambda s: int((s == "High").sum())),
                          revenue_at_risk=("revenue_at_risk_365d", "sum")).round(0))
        print("  High-risk customers by segment:\n" + cross.to_string())

    out = {"k": best_k, "silhouette": {k: round(v, 3) for k, v in scores.items()},
           "segments": summary.reset_index().to_dict(orient="records"), "actions": ACTIONS}
    (REPORTS / "segment_profile.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
