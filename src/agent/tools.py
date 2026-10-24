"""
Step 8a: The agent's tools.

Each tool is a plain Python function that reads the outputs of Steps 3-7 and returns
    {"summary": "<one paragraph of facts with numbers>", "data": <small table/dict>, "source": "<file>"}
The agent never invents numbers: every number in a report comes from one of these tools.

TOOL_SPECS describes each tool in JSON-schema form - the same format an LLM (Claude) uses
for tool calling, so the offline planner and the LLM agent share one tool set.
"""
import json

import pandas as pd

from src.config import SALES_CSV, DATA_PROCESSED, REPORTS

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Furniture", "Office Supplies", "Clothing"]
SEGMENTS = ["Champions", "Big-ticket buyers", "Loyal regulars", "Occasional buyers", "Slipping away"]
TICKET_CATS = ["Delivery", "Billing", "Product Quality", "Technical", "Refund"]


def _lakh(x):
    return f"Rs {x / 1e5:,.1f} lakh"


def _json(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _csv(name, **kw):
    p = DATA_PROCESSED / name
    return pd.read_csv(p, **kw) if p.exists() else None


def _reason_label(r: str) -> str:
    """'2 negative support ticket(s) in last 90 days' -> 'recent negative tickets' (so they can be counted)."""
    for key, label in [("negative support", "recent negative tickets"), ("orders down", "orders falling"),
                       ("no purchase", "no recent purchase"), ("high-priority", "unresolved high-priority complaint"),
                       ("Basic plan", "on Basic plan")]:
        if key in r:
            return label
    return r


# ---------------------------------------------------------------- tools
def sales_forecast(region: str = None):
    fc = _json("forecast_metrics.json")
    if not fc:
        return {"summary": "Forecast not available - run the pipeline first.", "data": None, "source": None}
    by = fc["forecast_by_region_inr"]
    lstm = _json("lstm_forecast_metrics.json")
    acc = fc["metrics"]["model"]["monthly_total_wape"]
    if region:
        val = by[region]
        share = val / sum(by.values())
        s = (f"{region} is forecast at {_lakh(val)} for {fc['forecast_window']} "
             f"({share:.0%} of the company total of {_lakh(fc['forecast_total_inr'])}).")
    else:
        s = (f"Next 30 days ({fc['forecast_window']}): {_lakh(fc['forecast_total_inr'])}, "
             f"{fc['forecast_vs_last30_pct']:+.1f}% vs the last 30 days. By region: "
             + ", ".join(f"{r} {_lakh(v)}" for r, v in sorted(by.items(), key=lambda x: -x[1])) + ".")
    s += f" Typical monthly forecast error: {acc:.1%}"
    if lstm:
        s += f" (LSTM model: {lstm['metrics']['lstm']['monthly_total_wape']:.1%})."
    return {"summary": s, "data": by, "source": "reports/forecast_metrics.json"}


def sales_trend(region: str = None, category: str = None):
    """Last completed quarter vs the one before and vs the same quarter last year."""
    s = pd.read_csv(SALES_CSV, parse_dates=["order_date"])
    if region:
        s = s[s["region"] == region]
    if category:
        s = s[s["category"] == category]
    q = s.set_index("order_date")["revenue"].resample("QE").sum()
    last, prev, ly = q.iloc[-1], q.iloc[-2], q.iloc[-5]
    who = " / ".join(x for x in [region, category] if x) or "Company"
    out = (f"{who} revenue in {q.index[-1]:%b %Y} quarter: {_lakh(last)}, {100 * (last / prev - 1):+.1f}% vs previous "
           f"quarter and {100 * (last / ly - 1):+.1f}% vs same quarter last year.")
    # what drove the change? category (or region) mix
    split_col = "category" if not category else "region"
    cur = s[s["order_date"] > q.index[-2]].groupby(split_col)["revenue"].sum()
    old = s[(s["order_date"] > q.index[-3]) & (s["order_date"] <= q.index[-2])].groupby(split_col)["revenue"].sum()
    delta = (cur - old).sort_values()
    drivers = pd.DataFrame({"last_q": cur, "prev_q": old, "change": delta,
                            "change_pct": 100 * (cur / old - 1)}).loc[delta.index].round(1)
    worst, best = delta.index[0], delta.index[-1]
    out += (f" Biggest drop: {worst} ({_lakh(delta[worst])}, {drivers.loc[worst, 'change_pct']:+.1f}%); "
            f"biggest gain: {best} ({_lakh(delta[best])}).")
    orders = s[s["order_date"] > q.index[-2]]
    prev_orders = s[(s["order_date"] > q.index[-3]) & (s["order_date"] <= q.index[-2])]
    out += (f" Orders {len(orders):,} vs {len(prev_orders):,}; active customers "
            f"{orders['customer_id'].nunique():,} vs {prev_orders['customer_id'].nunique():,}.")
    n_now, n_prev = orders["customer_id"].nunique(), prev_orders["customer_id"].nunique()
    return {"summary": out, "data": drivers.reset_index().to_dict(orient="records"), "source": "data/raw/sales.csv",
            "change_pct": round(float(100 * (last / prev - 1)), 1),
            "customers_change_pct": round(100 * (n_now / max(n_prev, 1) - 1), 1),
            "revenue_per_customer_change_pct": round(100 * ((last / max(n_now, 1)) / (prev / max(n_prev, 1)) - 1), 1)}


def churn_risk(region: str = None, segment: str = None, top_n: int = 5):
    c = _csv("customer_360.csv")
    if c is None:
        return {"summary": "Churn scores not available.", "data": None, "source": None}
    company_share = float((c["risk_band"] == "High").mean())
    if region:
        c = c[c["region"] == region]
    if segment:
        c = c[c["segment_name"] == segment]
    hi = c[c["risk_band"] == "High"].sort_values("revenue_at_risk_365d", ascending=False)
    who = " / ".join(x for x in [region, segment] if x) or "All customers"
    s = (f"{who}: {len(hi)} of {len(c)} customers are High churn risk ({len(hi) / max(len(c), 1):.0%}), "
         f"{_lakh(c['revenue_at_risk_365d'].sum())} annual revenue at risk.")
    if len(hi):
        reasons = hi["risk_reasons"].str.split("; ").explode().map(_reason_label).value_counts()
        s += " Most common warning signs: " + ", ".join(f"{r} ({n})" for r, n in reasons.head(3).items()) + "."
    top = hi.head(top_n)[["customer_id", "segment_name", "region", "churn_probability",
                          "revenue_at_risk_365d", "risk_reasons"]]
    return {"summary": s, "data": top.to_dict(orient="records"), "source": "data/processed/customer_360.csv",
            "high_risk_share": round(len(hi) / max(len(c), 1), 3), "company_high_risk_share": round(company_share, 3)}


def customer_segments():
    seg = _json("segment_profile.json")
    if not seg:
        return {"summary": "Segments not available.", "data": None, "source": None}
    s = "; ".join(f"{x['segment_name']}: {x['customers']} customers, {x['revenue_share_pct']}% of revenue"
                  for x in seg["segments"])
    data = [{"segment": x["segment_name"], "customers": x["customers"], "revenue_share_pct": x["revenue_share_pct"],
             "action": seg["actions"].get(x["segment_name"].split(" (")[0], "")} for x in seg["segments"]]
    return {"summary": f"{seg['k']} segments - {s}.", "data": data, "source": "reports/segment_profile.json"}


def ticket_insights(category: str = None, region: str = None):
    ins = _json("ticket_insights.json")
    if not ins:
        return {"summary": "Ticket insights not available.", "data": None, "source": None}
    trend = ins["category_trend"]
    if category:
        trend = [t for t in trend if t["category"] == category]
    s = "Negative share of tickets, last 3 months vs previous 9: " + "; ".join(
        f"{t['category']} {t['negative_share_prev_9m']:.0%}->{t['negative_share_last_3m']:.0%} ({t['status']})"
        for t in trend) + "."
    themes = {k: v for k, v in ins["complaint_themes"].items() if not category or k == category}
    s += " Top complaints: " + "; ".join(f"{k}: {v[0]['phrase']}" for k, v in themes.items() if v) + "."
    extra = {}
    if region:
        t = _csv("ticket_predictions.csv")
        c = _csv("customer_360.csv")
        if t is not None and c is not None:
            t = t.merge(c[["customer_id", "region"]], on="customer_id", how="left")
            neg = t.groupby("region")["pred_sentiment"].apply(lambda x: (x == "negative").mean())
            comp = float((t["pred_sentiment"] == "negative").mean())
            s += f" In {region}, {neg.get(region, float('nan')):.0%} of 2026 tickets are negative (company: {comp:.0%})."
            extra = {"region_negative_share": round(float(neg.get(region, float("nan"))), 3),
                     "company_negative_share": round(comp, 3)}
    return {"summary": s, "data": {"trend": trend, "themes": themes}, "source": "reports/ticket_insights.json", **extra}


def urgent_calls(region: str = None, top_n: int = 5):
    u = _csv("urgent_tickets.csv")
    if u is None:
        return {"summary": "Urgent list not available.", "data": None, "source": None}
    c = _csv("customer_360.csv")
    if region and c is not None:
        u = u.merge(c[["customer_id", "region"]], on="customer_id", how="left")
        u = u[u["region"] == region]
    s = (f"{len(u)} high-churn-risk customers{' in ' + region if region else ''} raised a negative ticket in the last "
         f"60 days ({_lakh(u['revenue_at_risk_365d'].sum())} at risk). Call them within 1 business day (Support SOP 4).")
    cols = [x for x in ["customer_id", "segment_name", "revenue_at_risk_365d", "pred_category", "text"] if x in u.columns]
    return {"summary": s, "data": u.head(top_n)[cols].to_dict(orient="records"), "source": "data/processed/urgent_tickets.csv"}


def order_anomalies(region: str = None, top_n: int = 5):
    a = _csv("order_anomalies.csv")
    if a is None:
        return {"summary": "Anomaly list not available.", "data": None, "source": None}
    if region:
        a = a[a["region"] == region]
    reasons = a["main_reason"].value_counts()
    s = (f"{len(a)} orders{' in ' + region if region else ''} flagged as unusual by the autoencoder "
         f"({_lakh(a['revenue'].sum())} of order value). Main reasons: "
         + ", ".join(f"{r} ({n})" for r, n in reasons.head(3).items()) + ".")
    top = a.head(top_n)[["order_id", "order_date", "customer_id", "category", "quantity", "unit_price", "discount",
                         "revenue", "main_reason"]]
    return {"summary": s, "data": top.to_dict(orient="records"), "source": "data/processed/order_anomalies.csv"}


def customer_lookup(customer_id: str):
    c = _csv("customer_360.csv")
    row = c[c["customer_id"] == customer_id] if c is not None else pd.DataFrame()
    if row.empty:
        return {"summary": f"No customer {customer_id} found.", "data": None, "source": None}
    r = row.iloc[0]
    s = (f"{customer_id}: {r['segment_name']} ({r['segment']}, {r['region']}). Revenue last 12 months "
         f"{_lakh(r['revenue_365d'])}, {int(r['orders_365d'])} orders, last order {int(r['recency_days'])} days ago. "
         f"Churn risk {r['risk_band']} ({r['churn_probability']:.0%})"
         + (f" - {r['risk_reasons']}" if r["risk_band"] != "Low" else "") + f". Suggested action: {r['recommended_action']}.")
    t = _csv("ticket_predictions.csv")
    if t is not None:
        mine = t[t["customer_id"] == customer_id]
        if len(mine):
            last = mine.iloc[-1]
            s += f" {len(mine)} ticket(s) in 2026; latest ({last['created_at']}, {last['pred_category']}, {last['pred_sentiment']}): \"{last['text'][:120]}\""
    return {"summary": s, "data": row.iloc[0].to_dict(), "source": "data/processed/customer_360.csv"}


_RAG = None


def search_documents(question: str):
    global _RAG
    from src.rag.rag_system import RAG
    _RAG = _RAG or RAG()
    r = _RAG.ask(question, use_llm=False)
    return {"summary": r["answer"], "data": {"sources": r["sources"], "found": r["found"]}, "source": "data/docs (RAG)"}


TOOLS = {f.__name__: f for f in [sales_forecast, sales_trend, churn_risk, customer_segments, ticket_insights,
                                 urgent_calls, order_anomalies, customer_lookup, search_documents]}

_region = {"type": "string", "enum": REGIONS}
TOOL_SPECS = [
    {"name": "sales_forecast", "description": "Revenue forecast for the next 30 days, total or for one region, with model accuracy.",
     "input_schema": {"type": "object", "properties": {"region": _region}}},
    {"name": "sales_trend", "description": "Last quarter's revenue vs previous quarter and last year, and which categories (or regions) drove the change. Use for 'why are sales falling/rising'.",
     "input_schema": {"type": "object", "properties": {"region": _region, "category": {"type": "string", "enum": CATEGORIES}}}},
    {"name": "churn_risk", "description": "Customers likely to leave in the next 90 days: count, revenue at risk, warning signs, top customers.",
     "input_schema": {"type": "object", "properties": {"region": _region, "segment": {"type": "string", "enum": SEGMENTS},
                                                       "top_n": {"type": "integer"}}}},
    {"name": "customer_segments", "description": "Customer segments (RFM clusters) with size, revenue share and recommended action.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "ticket_insights", "description": "Support ticket sentiment trends by category, top complaint phrases, optional region comparison.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string", "enum": TICKET_CATS}, "region": _region}}},
    {"name": "urgent_calls", "description": "High churn-risk customers with a recent negative ticket - who to call today.",
     "input_schema": {"type": "object", "properties": {"region": _region, "top_n": {"type": "integer"}}}},
    {"name": "order_anomalies", "description": "Unusual orders flagged by the autoencoder (pricing errors, odd quantities, big discounts).",
     "input_schema": {"type": "object", "properties": {"region": _region, "top_n": {"type": "integer"}}}},
    {"name": "customer_lookup", "description": "Everything about one customer: segment, revenue, churn risk, recent tickets.",
     "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}},
    {"name": "search_documents", "description": "Search company policies, SOPs and the Q3 business review PDFs; returns the relevant section with citation.",
     "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
]
