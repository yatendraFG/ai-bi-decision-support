"""
Step 9b: Streamlit dashboard - everything from Steps 3-8 in one place.

Run:  streamlit run app/dashboard.py        (or double-click run_dashboard.bat)

Pages
  Overview   - KPIs, revenue by region, 30-day forecast, the agent's top actions
  Sales      - monthly revenue by region/category, forecast vs recent actuals, quarter diagnosis
  Customers  - segments, churn-risk list with filters + download
  Support    - complaint trends, themes, urgent call list, classify a new ticket
  Anomalies  - suspicious orders and why they were flagged
  Ask AI     - chat with the agent (business questions) or the documents (policies)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.config import SALES_CSV, DATA_PROCESSED, MODELS
from src.agent import tools as T

st.set_page_config(page_title="Kaveri Retail AI BI", page_icon=":bar_chart:", layout="wide")

# fixed colour per entity (never by rank) - validated categorical palette, slots 1-4
REGION_COLORS = {"North": "#2a78d6", "South": "#eb6834", "East": "#1baf7a", "West": "#eda100"}
CAT_COLORS = {"Electronics": "#2a78d6", "Furniture": "#eb6834", "Office Supplies": "#1baf7a", "Clothing": "#eda100"}
ONE = "#2a78d6"
lakh = lambda x: f"Rs {x / 1e5:,.1f} lakh"


# ---------------------------------------------------------------- data (cached)
@st.cache_data
def sales():
    return pd.read_csv(SALES_CSV, parse_dates=["order_date"])


@st.cache_data
def processed(name, **kw):
    p = DATA_PROCESSED / name
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


@st.cache_data
def churn_table():
    """All customers scored by the churn model (same numbers as the reports), with their segment."""
    ch = processed("churn_scores.csv")
    seg = processed("customer_segments.csv")
    if ch.empty:
        return ch
    if not seg.empty:
        ch = ch.merge(seg[["customer_id", "segment_name"]], on="customer_id", how="left")
    ch["segment_name"] = ch.get("segment_name", pd.Series(index=ch.index, dtype=object)).fillna("No purchase history")
    return ch


@st.cache_data
def tool(name, **kw):
    return T.TOOLS[name](**kw)


@st.cache_resource
def ticket_models():
    return {t: joblib.load(MODELS / f"ticket_{t}.joblib") for t in ("sentiment", "category")}


@st.cache_resource
def rag():
    from src.rag.rag_system import RAG
    return RAG()


@st.cache_data
def overview_report():
    from src.agent.agent import run_planner
    return run_planner("How is the business doing overall and what should we do?", save_report=False)


def line(df, x, y, color, colors, y_title, height=320):
    return (alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=30))
            .encode(x=alt.X(x, title=None), y=alt.Y(y, title=y_title),
                    color=alt.Color(color, scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[x, color, alt.Tooltip(y, format=",.1f")])
            .properties(height=height).interactive(bind_y=False))


def hbar(df, x, y, x_title, color=ONE, fmt=",.1f", height=None):
    return (alt.Chart(df).mark_bar(color=color, cornerRadiusEnd=4, height=18)
            .encode(x=alt.X(x, title=x_title), y=alt.Y(y, sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
                    tooltip=[y, alt.Tooltip(x, format=fmt)])
            .properties(height=height or 40 * len(df) + 30))


def missing():
    st.warning("Outputs not found. Run `run_project.bat` (or `python run_pipeline.py`) first.")
    st.stop()


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("Business overview")
    fc = tool("sales_forecast")
    if fc["data"] is None:
        missing()
    ch, uc, an = tool("churn_risk"), tool("urgent_calls"), tool("order_anomalies")
    c360 = churn_table()
    fcm = T._json("forecast_metrics.json")
    k = st.columns(5)
    k[0].metric("Next 30 days, Rs lakh", f"{fcm['forecast_total_inr'] / 1e5:,.1f}",
                f"{fcm['forecast_vs_last30_pct']:+.1f}% vs last 30 days")
    k[1].metric("High churn-risk customers", f"{(c360['risk_band'] == 'High').sum():,}", f"of {len(c360):,} active", delta_color="off")
    k[2].metric("At risk (12 m), Rs lakh", f"{c360['revenue_at_risk_365d'].sum() / 1e5:,.1f}", delta_color="off")
    k[3].metric("Call today", f"{len(processed('urgent_tickets.csv')):,}", "risky + negative ticket", delta_color="off")
    k[4].metric("Suspicious orders", f"{len(processed('order_anomalies.csv')):,}", delta_color="off")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Monthly revenue by region")
        s = sales()
        start = (s["order_date"].max() - pd.DateOffset(months=24)).to_period("M").to_timestamp() + pd.DateOffset(months=1)
        m = (s[s["order_date"] >= start]
             .groupby([pd.Grouper(key="order_date", freq="MS"), "region"])["revenue"].sum().div(1e5).round(1)
             .rename("revenue_lakh").reset_index().rename(columns={"order_date": "month"}))
        st.altair_chart(line(m, "month:T", "revenue_lakh:Q", "region:N", REGION_COLORS, "Rs lakh"), width="stretch")
    with right:
        st.subheader("Forecast, next 30 days")
        f = pd.DataFrame({"region": list(fc["data"]), "lakh": [v / 1e5 for v in fc["data"].values()]})
        st.altair_chart(hbar(f, "lakh:Q", "region:N", "Rs lakh"), width="stretch")
        st.caption(fc["summary"].split(". ")[-1])

    r = overview_report()
    st.subheader("What the AI agent recommends")
    st.caption("Rule-based suggestions from the numbers above - review before acting.")
    for i, a in enumerate(r["actions"], 1):
        st.markdown(f"{i}. {a}")


def page_sales():
    st.title("Sales")
    s = sales()
    c1, c2 = st.columns([2, 1])
    regions = c1.multiselect("Regions", list(REGION_COLORS), default=list(REGION_COLORS))
    cat = c2.selectbox("Category", ["All"] + list(CAT_COLORS))
    d = s[s["region"].isin(regions)]
    if cat != "All":
        d = d[d["category"] == cat]
    if d.empty:
        st.info("Pick at least one region.")
        return
    m = (d.groupby([pd.Grouper(key="order_date", freq="MS"), "region"])["revenue"].sum().div(1e5).round(1)
         .rename("revenue_lakh").reset_index().rename(columns={"order_date": "month"}))
    st.subheader(f"Monthly revenue{' - ' + cat if cat != 'All' else ''}")
    st.altair_chart(line(m, "month:T", "revenue_lakh:Q", "region:N",
                         {r: REGION_COLORS[r] for r in regions}, "Rs lakh"), width="stretch")

    st.subheader("Last 90 days and the 30-day forecast (7-day average, all categories)")
    fc = processed("sales_forecast.csv", parse_dates=["date"])
    if not fc.empty:
        act = (s[s["order_date"] > s["order_date"].max() - pd.Timedelta(days=96)]
               .groupby(["order_date", "region"])["revenue"].sum().div(1e5).reset_index()
               .rename(columns={"order_date": "date", "revenue": "lakh"}).assign(kind="actual"))
        fl = fc.melt(id_vars="date", value_vars=list(REGION_COLORS), var_name="region", value_name="lakh")
        fl = fl.assign(lakh=fl["lakh"] / 1e5, kind="forecast")
        both = pd.concat([act, fl]).sort_values(["region", "date"])
        both["lakh"] = both.groupby("region")["lakh"].transform(lambda x: x.rolling(7, min_periods=7).mean())
        both = both.dropna(subset=["lakh"])
        # repeat the last actual point as forecast so the dashed line joins the solid one
        last = both[both["kind"] == "actual"].groupby("region").tail(1).assign(kind="forecast")
        both = pd.concat([both, last])
        both = both[both["region"].isin(regions)]
        ch = (alt.Chart(both).mark_line(strokeWidth=2)
              .encode(x=alt.X("date:T", title=None), y=alt.Y("lakh:Q", title="Rs lakh per day (7-day avg)"),
                      color=alt.Color("region:N", scale=alt.Scale(domain=list(REGION_COLORS), range=list(REGION_COLORS.values())),
                                      legend=alt.Legend(orient="top", title=None)),
                      strokeDash=alt.StrokeDash("kind:N", scale=alt.Scale(domain=["actual", "forecast"], range=[[1, 0], [5, 4]]),
                                                legend=alt.Legend(orient="top", title=None)),
                      tooltip=["date:T", "region:N", "kind:N", alt.Tooltip("lakh:Q", format=",.2f")])
              .properties(height=300))
        st.altair_chart(ch, width="stretch")

    st.subheader("Last quarter - what changed?")
    pick = st.selectbox("Diagnose region", ["Company"] + list(REGION_COLORS))
    tr = tool("sales_trend", region=None if pick == "Company" else pick, category=None if cat == "All" else cat)
    st.markdown(tr["summary"])
    drv = pd.DataFrame(tr["data"]).rename(columns={"index": "name"})
    key = drv.columns[0]
    drv["change_lakh"] = drv["change"] / 1e5
    bars = (alt.Chart(drv).mark_bar(cornerRadiusEnd=4, height=18)
            .encode(x=alt.X("change_lakh:Q", title="Change vs previous quarter (Rs lakh)"),
                    y=alt.Y(f"{key}:N", sort="x", title=None),
                    color=alt.condition("datum.change_lakh < 0", alt.value("#e34948"), alt.value("#2a78d6")),
                    tooltip=[f"{key}:N", alt.Tooltip("change_lakh:Q", format="+,.1f"), alt.Tooltip("change_pct:Q", format="+.1f")])
            .properties(height=40 * len(drv) + 30))
    st.altair_chart(bars, width="stretch")


def page_customers():
    st.title("Customers")
    seg = tool("customer_segments")
    c360 = churn_table()
    if seg["data"] is None or c360.empty:
        missing()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Segments - share of revenue")
        sd = pd.DataFrame(seg["data"])
        st.altair_chart(hbar(sd, "revenue_share_pct:Q", "segment:N", "% of revenue"), width="stretch")
    with right:
        st.subheader("What to do with each segment")
        st.dataframe(sd[["segment", "customers", "action"]], hide_index=True, width="stretch")

    st.subheader("Churn risk")
    f1, f2, f3 = st.columns(3)
    reg = f1.multiselect("Region", list(REGION_COLORS), default=list(REGION_COLORS))
    segs = f2.multiselect("Segment", sorted(c360["segment_name"].dropna().unique()))
    band = f3.selectbox("Risk band", ["High", "Medium", "Low", "All"])
    d = c360[c360["region"].isin(reg)]
    if segs:
        d = d[d["segment_name"].isin(segs)]
    if band != "All":
        d = d[d["risk_band"] == band]
    d = d.sort_values("revenue_at_risk_365d", ascending=False)
    m = st.columns(3)
    m[0].metric("Customers", f"{len(d):,}")
    m[1].metric("Revenue at risk", lakh(d["revenue_at_risk_365d"].sum()))
    m[2].metric("Avg churn probability", f"{d['churn_probability'].mean():.0%}" if len(d) else "-")
    show = d[["customer_id", "segment_name", "region", "plan", "churn_probability", "revenue_365d", "revenue_at_risk_365d", "risk_reasons"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"churn_probability": st.column_config.ProgressColumn("churn prob.", format="%.2f", min_value=0, max_value=1),
                                "revenue_365d": st.column_config.NumberColumn("revenue 12m", format="%.0f"),
                                "revenue_at_risk_365d": st.column_config.NumberColumn("at risk", format="%.0f")})
    st.download_button("Download this list (CSV)", show.to_csv(index=False), "churn_list.csv", "text/csv")

    st.subheader("Look up a customer")
    cid = st.text_input("Customer id", placeholder="e.g. C00509")
    if cid:
        st.markdown(tool("customer_lookup", customer_id=cid.strip().upper())["summary"])


def page_support():
    st.title("Support tickets")
    ins = T._json("ticket_insights.json")
    nlp = T._json("nlp_metrics.json")
    tp = processed("ticket_predictions.csv")
    if not ins or tp.empty:
        missing()
    k = st.columns(4)
    k[0].metric("Tickets in 2026", f"{len(tp):,}")
    k[1].metric("Negative (model)", f"{(tp['pred_sentiment'] == 'negative').mean():.0%}")
    k[2].metric("Sent to human review", f"{tp['needs_review'].mean():.0%}")
    k[3].metric("Routing accuracy (test)", f"{nlp['test']['category']['accuracy']:.0%}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Negative share by category")
        tr = pd.DataFrame(ins["category_trend"])
        long = pd.concat([tr.assign(period="previous 9 months", share=tr["negative_share_prev_9m"]),
                          tr.assign(period="last 3 months", share=tr["negative_share_last_3m"])])
        ch = (alt.Chart(long).mark_bar(cornerRadiusEnd=4, height=12)
              .encode(y=alt.Y("category:N", title=None), yOffset="period:N",
                      x=alt.X("share:Q", title="share of tickets that are negative", axis=alt.Axis(format="%")),
                      color=alt.Color("period:N", scale=alt.Scale(domain=["previous 9 months", "last 3 months"],
                                                                 range=["#2a78d6", "#eb6834"]),
                                      legend=alt.Legend(orient="top", title=None)),
                      tooltip=["category", "period", alt.Tooltip("share:Q", format=".0%")])
              .properties(height=300))
        st.altair_chart(ch, width="stretch")
    with right:
        st.subheader("What customers complain about")
        for cat, ph in ins["complaint_themes"].items():
            st.markdown(f"**{cat}** - " + "; ".join(f"{p['phrase']} ({p['tickets']})" for p in ph[:3]))

    st.subheader("Call today: high churn risk + negative ticket")
    u = processed("urgent_tickets.csv")
    st.dataframe(u, hide_index=True, width="stretch")
    st.download_button("Download call list (CSV)", u.to_csv(index=False), "urgent_calls.csv", "text/csv")

    st.subheader("Classify a new ticket")
    with st.form("classify"):
        text = st.text_area("Ticket text", "Refund abhi tak nahi aaya, 3 hafte ho gaye!!")
        prio = st.selectbox("Priority chosen by customer", ["Low", "Medium", "High"], index=2)
        go = st.form_submit_button("Classify")
    if go:
        if len(text.strip()) < 3:
            st.error("Enter the ticket text first.")
        else:
            X = pd.DataFrame([{"text": text, "priority": prio}])
            cols, review = st.columns(3), False
            for i, (task, b) in enumerate(ticket_models().items()):
                p = b["model"].predict_proba(X)[0]
                j = int(p.argmax())
                cols[i].metric(task.capitalize(), str(b["model"].classes_[j]), f"confidence {p[j]:.0%}", delta_color="off")
                review |= p[j] < b["review_threshold"]
            cols[2].metric("Route", "Human review" if review else "Auto", delta_color="off")


def page_anomalies():
    st.title("Suspicious orders")
    a = processed("order_anomalies.csv")
    if a.empty:
        missing()
    reg = st.multiselect("Region", list(REGION_COLORS), default=list(REGION_COLORS))
    a = a[a["region"].isin(reg)]
    k = st.columns(2)
    k[0].metric("Flagged orders", f"{len(a):,}")
    k[1].metric("Order value", lakh(a["revenue"].sum()))
    labels = {"qty_vs_own_history": "quantity unusual for this customer", "price_vs_category": "price unusual for category",
              "discount": "unusual discount", "qty_vs_segment": "quantity unusual for segment",
              "log_quantity": "very large quantity", "log_unit_price": "unusual unit price", "log_revenue": "unusual order value"}
    r = a["main_reason"].map(lambda x: labels.get(x, x.replace("cat_", "category: ").replace("seg_", "segment: ")))
    rc = r.value_counts().head(8).rename_axis("reason").reset_index(name="orders")
    st.subheader("Why they were flagged")
    st.altair_chart(hbar(rc, "orders:Q", "reason:N", "orders", fmt=","), width="stretch")
    st.subheader("Most unusual first")
    st.dataframe(a.assign(reason=r).drop(columns=["main_reason"]).head(200), hide_index=True, width="stretch")


def page_ask():
    st.title("Ask AI")
    mode = st.radio("Ask", ["Business question (agent)", "Company documents (policies, SOP, Q3 review)"], horizontal=True)
    key = "chat_agent" if mode.startswith("Business") else "chat_docs"
    st.session_state.setdefault(key, [])
    examples = (["West region ki sales kyun gir rahi hai aur kya karein?", "Which customers are likely to leave in North?",
                 "Tell me about customer C01658", "How is the business doing overall?"]
                if key == "chat_agent" else
                ["refund card mein kitne din mein aayega?", "Who approves a refund of Rs 25,000?",
                 "What is the warranty on furniture?", "How many customers are at high churn risk?"])
    st.caption("Try: " + " | ".join(f"*{e}*" for e in examples))
    for role, msg in st.session_state[key]:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Type your question (English or Hinglish)")
    if q:
        st.session_state[key].append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if key == "chat_agent":
                    from src.agent.agent import ask
                    ans = ask(q)["report"].replace("<sub>", "*").replace("</sub>", "*")
                else:
                    r = rag().ask(q)
                    ans = r["answer"] + (("\n\n*Also see: " + " | ".join(r["sources"][1:]) + "*") if r["found"] and len(r["sources"]) > 1 else "")
            st.markdown(ans)
        st.session_state[key].append(("assistant", ans))


PAGES = {"Overview": page_overview, "Sales": page_sales, "Customers": page_customers,
         "Support": page_support, "Anomalies": page_anomalies, "Ask AI": page_ask}
with st.sidebar:
    st.header("Kaveri Retail AI BI")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.caption("Fictional company. Data and models from `run_pipeline.py` (Steps 2-8).")
    if st.button("Reload data"):
        st.cache_data.clear()
        st.cache_resource.clear()
PAGES[page]()
