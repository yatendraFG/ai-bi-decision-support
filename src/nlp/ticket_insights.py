"""
Step 6c: Turn ticket predictions into actions for support & management.

Uses the Step 6b models to tag every ticket (exactly what would happen in production,
where new tickets arrive without labels), then answers three questions:

  1. Which complaint categories are getting worse?
     negative-share in the last 3 months vs the 9 months before, per category.
  2. What are people actually complaining about?
     most frequent 2-3 word phrases in recent negative tickets, per category.
  3. Who should we call today?
     recent negative tickets from customers the Step 4 churn model marks High risk,
     sorted by the revenue we could lose.

Outputs:
  data/processed/urgent_tickets.csv
  reports/ticket_insights.json
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import TICKETS_CSV, MODELS, DATA_PROCESSED, REPORTS

# only greetings / closings / filler are removed - words like "not", "on", "is" carry the meaning
STOP = {"hi", "hello", "team", "dear", "support", "please", "pls", "thanks", "thank", "regards", "kindly",
        "needful", "asap", "waiting", "reply", "namaste", "hey", "there", "rs"}


EDGE = {"the", "a", "an", "is", "on", "i", "my", "of", "to", "and", "but", "for", "in", "it", "was",
        "me", "this", "that", "with", "at", "so", "ka", "ki", "ke", "hai", "ho"}


def tag_all(t: pd.DataFrame) -> pd.DataFrame:
    for task in ("sentiment", "category"):
        b = joblib.load(MODELS / f"ticket_{task}.joblib")
        proba = b["model"].predict_proba(t)
        t[f"pred_{task}"] = b["model"].classes_[proba.argmax(1)]
        t[f"{task}_confidence"] = proba.max(1).round(3)
    return t


def clean(text: str) -> str:
    text = re.sub(r"#?\d+", " ", text.lower())          # order ids, amounts, days
    return re.sub(r"[^a-z ]", " ", text)


def top_phrases(texts, n=5):
    if len(texts) < 5:
        return []
    cv = CountVectorizer(ngram_range=(3, 4), min_df=3, stop_words=list(STOP), preprocessor=clean,
                         token_pattern=r"(?u)\b\w+\b")
    try:
        X = cv.fit_transform(texts)
    except ValueError:
        return []
    counts = np.asarray(X.sum(0)).ravel()
    vocab = cv.get_feature_names_out()
    # most frequent first; on ties prefer the longer (more complete) phrase
    order = sorted(range(len(vocab)), key=lambda i: (-counts[i], -len(vocab[i].split())))
    out, kept = [], []
    for i in order:
        words = vocab[i].split()
        if words[0] in EDGE or words[-1] in EDGE:          # "the coupon code" -> ugly edge, skip
            continue
        if any(len(set(words) & k) >= 2 for k in kept):    # same complaint, overlapping words
            continue
        kept.append(set(words))
        out.append({"phrase": vocab[i], "tickets": int(counts[i])})
        if len(out) == n:
            break
    return out


def main():
    print("[Step 6c] Ticket insights")
    t = tag_all(pd.read_csv(TICKETS_CSV, parse_dates=["created_at"]))
    end = t["created_at"].max()
    last3 = t[t["created_at"] > end - pd.DateOffset(months=3)]
    prev9 = t[(t["created_at"] <= end - pd.DateOffset(months=3)) & (t["created_at"] > end - pd.DateOffset(months=12))]

    # 1. trend by category
    trend = []
    for cat in sorted(t["pred_category"].unique()):
        a = last3[last3["pred_category"] == cat]
        b = prev9[prev9["pred_category"] == cat]
        na, nb = (a["pred_sentiment"] == "negative").mean(), (b["pred_sentiment"] == "negative").mean()
        trend.append({"category": cat, "tickets_last_3m": int(len(a)),
                      "tickets_per_month_last_3m": round(len(a) / 3, 1), "tickets_per_month_prev_9m": round(len(b) / 9, 1),
                      "negative_share_last_3m": round(float(na), 3), "negative_share_prev_9m": round(float(nb), 3),
                      "change_pp": round(100 * float(na - nb), 1),
                      "status": "WORSENING" if na - nb > 0.05 else "improving" if nb - na > 0.05 else "stable"})
    trend = sorted(trend, key=lambda r: -r["change_pp"])
    print("  Negative share by category (last 3 months vs previous 9):")
    for r in trend:
        print(f"    {r['category']:16s} {r['negative_share_prev_9m']:.0%} -> {r['negative_share_last_3m']:.0%}"
              f"  ({r['change_pp']:+.1f} pp, {r['status']})  ~{r['tickets_per_month_last_3m']:.0f} tickets/month")

    # 2. what people complain about (recent 6 months, negative tickets)
    recent_neg = t[(t["created_at"] > end - pd.DateOffset(months=6)) & (t["pred_sentiment"] == "negative")]
    themes = {cat: top_phrases(g["text"].tolist()) for cat, g in recent_neg.groupby("pred_category")}
    print("  Top complaint themes:")
    for cat, ph in themes.items():
        print(f"    {cat:16s} " + "; ".join(f"{p['phrase']} ({p['tickets']})" for p in ph[:3]))

    # 3. urgent call list
    churn_path = DATA_PROCESSED / "churn_scores.csv"
    urgent_n = 0
    if churn_path.exists():
        churn = pd.read_csv(churn_path)[["customer_id", "segment", "churn_probability", "risk_band",
                                         "revenue_365d", "revenue_at_risk_365d"]]
        seg_path = DATA_PROCESSED / "customer_segments.csv"
        if seg_path.exists():
            churn = churn.merge(pd.read_csv(seg_path)[["customer_id", "segment_name"]], on="customer_id", how="left")
        recent = t[(t["created_at"] > end - pd.Timedelta(days=60)) & (t["pred_sentiment"] == "negative")]
        urgent = recent.merge(churn, on="customer_id", how="inner")
        urgent = urgent[urgent["risk_band"] == "High"].sort_values(
            ["revenue_at_risk_365d", "created_at"], ascending=[False, False])
        urgent = urgent.drop_duplicates("customer_id")
        cols = ["customer_id", "segment_name", "segment", "churn_probability", "revenue_at_risk_365d",
                "ticket_id", "created_at", "pred_category", "priority", "text"]
        urgent[[c for c in cols if c in urgent.columns]].to_csv(DATA_PROCESSED / "urgent_tickets.csv", index=False)
        urgent_n = len(urgent)
        print(f"  Urgent call list: {urgent_n} high-churn-risk customers with a negative ticket in the last 60 days, "
              f"INR {urgent['revenue_at_risk_365d'].sum() / 1e5:,.1f} lakh revenue at risk -> data/processed/urgent_tickets.csv")

    out = {"as_of": str(end.date()), "category_trend": trend, "complaint_themes": themes,
           "urgent_customers": urgent_n,
           "urgent_revenue_at_risk_inr": float(urgent["revenue_at_risk_365d"].sum()) if urgent_n else 0.0}
    (REPORTS / "ticket_insights.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
