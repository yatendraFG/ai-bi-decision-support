"""Customer-level feature engineering shared by churn prediction and segmentation.

Every feature is computed "as of" a snapshot date using only data BEFORE that date,
so models trained on past snapshots never peek into the future (no leakage).
"""
import numpy as np
import pandas as pd

from src.config import SALES_CSV, CUSTOMERS_CSV, TICKETS_CSV


def load_raw():
    sales = pd.read_csv(SALES_CSV, parse_dates=["order_date"])
    cust = pd.read_csv(CUSTOMERS_CSV, parse_dates=["signup_date", "churn_date"])
    tickets = pd.read_csv(TICKETS_CSV, parse_dates=["created_at"])
    return sales, cust, tickets


def customer_features(snapshot, sales, cust, tickets) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot)
    s = sales[sales["order_date"] < snapshot]
    t = tickets[tickets["created_at"] < snapshot]
    # customers who exist and haven't left yet at the snapshot
    base = cust[(cust["signup_date"] < snapshot - pd.Timedelta(days=60))
                & ~(cust["churn_date"] < snapshot)].copy()

    def window(df, col, days):
        return df[df[col] >= snapshot - pd.Timedelta(days=days)]

    g_all = s.groupby("customer_id")
    f = pd.DataFrame(index=base["customer_id"])
    f["recency_days"] = (snapshot - g_all["order_date"].max()).dt.days
    for d in (90, 180, 365):
        w = window(s, "order_date", d).groupby("customer_id")
        f[f"orders_{d}d"] = w.size()
        f[f"revenue_{d}d"] = w["revenue"].sum()
    prev90 = s[(s["order_date"] < snapshot - pd.Timedelta(days=90))
               & (s["order_date"] >= snapshot - pd.Timedelta(days=180))].groupby("customer_id")
    f["orders_prev90d"] = prev90.size()
    f["revenue_prev90d"] = prev90["revenue"].sum()
    f["orders_lifetime"] = g_all.size()
    f["revenue_lifetime"] = g_all["revenue"].sum()
    f["avg_order_value"] = g_all["revenue"].mean()
    f["avg_discount"] = g_all["discount"].mean()
    f["n_categories"] = g_all["category"].nunique()

    for d in (90, 365):
        w = window(t, "created_at", d)
        f[f"tickets_{d}d"] = w.groupby("customer_id").size()
        f[f"neg_tickets_{d}d"] = w[w["sentiment"] == "negative"].groupby("customer_id").size()
    w90 = window(t, "created_at", 90)
    f["high_prio_tickets_90d"] = w90[w90["priority"] == "High"].groupby("customer_id").size()

    f = f.fillna({c: 0 for c in f.columns if c not in ("recency_days", "avg_order_value", "avg_discount")})
    f["recency_days"] = f["recency_days"].fillna(999)
    f["avg_order_value"] = f["avg_order_value"].fillna(0)
    f["avg_discount"] = f["avg_discount"].fillna(0)

    # trends: is the customer buying less than before?
    f["order_trend"] = (f["orders_90d"] + 1) / (f["orders_prev90d"] + 1)
    f["revenue_trend"] = (f["revenue_90d"] + 1) / (f["revenue_prev90d"] + 1)
    f["neg_ticket_share_365d"] = f["neg_tickets_365d"] / f["tickets_365d"].replace(0, np.nan)
    f["neg_ticket_share_365d"] = f["neg_ticket_share_365d"].fillna(0)

    meta = base.set_index("customer_id")
    f["tenure_days"] = (snapshot - meta["signup_date"]).dt.days
    for col in ("segment", "plan", "region"):
        f[col] = meta[col]
    f["snapshot"] = snapshot
    return f.reset_index()
