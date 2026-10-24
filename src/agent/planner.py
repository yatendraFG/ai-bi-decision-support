"""
Step 8b: Offline planner - decides WHICH tools to call, with WHICH arguments.

No LLM needed: it works like a transparent, rule-based "router".
  1. normalise  - lowercase, translate common Hinglish words ("kyun" -> "why", "gir" -> "falling")
  2. entities   - region, product category, segment, ticket category, customer id, top-N
  3. intents    - keyword scores for each tool (forecast, trend, churn, ...)
  4. recipes    - some questions need several tools together:
                  "why are West sales falling?"  -> trend + churn + tickets + urgent calls (a diagnosis)
                  "how is the business doing?"   -> forecast + trend + churn + tickets (an overview)
  5. fallback   - nothing matched -> search the documents

Every decision is recorded in plan["reasons"], so you can see why each tool was chosen.
The Claude agent (agent.py, optional) replaces this file with the LLM's own tool choice.
"""
import re

from src.agent.tools import REGIONS, CATEGORIES, SEGMENTS, TICKET_CATS

HINGLISH = {"kyun": "why", "kyon": "why", "gir": "falling", "girna": "falling", "ghat": "falling",
            "kam ho": "falling", "badh": "growing", "agle": "next", "mahine": "month", "kitni": "how much",
            "kitna": "how much", "grahak": "customer", "chhod": "churn leave", "kaun": "who", "kis": "which",
            "shikayat": "complaint", "kya karein": "what should we do", "kya kare": "what should we do",
            "gadbad": "anomaly", "batao": "tell", "dikhao": "show", "kaisa": "how is", "kaise": "how"}

INTENTS = {
    "sales_forecast": ["forecast", "next month", "next 30", "predict", "expected sales", "outlook", "projection",
                       "will sales", "how much sales", "upcoming"],
    "sales_trend": ["why", "falling", "drop", "decline", "down", "growth", "trend", "increase", "compare",
                    "last quarter", "performance", "performing", "growing", "revenue change", "sales change"],
    "churn_risk": ["churn", "leave", "leaving", "at risk", "retain", "retention", "lose customer", "losing customer",
                   "risky customer", "high risk"],
    "customer_segments": ["segment", "best customer", "valuable", "cluster", "type of customer", "types of customer",
                          "rfm", "customer group"],
    "ticket_insights": ["complain", "ticket", "support issue", "sentiment", "negative", "angry", "unhappy",
                        "customer issue", "problems customers"],
    "urgent_calls": ["call", "contact today", "urgent", "who should we", "reach out", "priority customer"],
    "order_anomalies": ["anomal", "fraud", "unusual", "suspicious", "odd order", "wrong price", "strange order",
                        "pricing error", "outlier"],
    "search_documents": ["policy", "sop", "rule", "approve", "allowed", "how many days", "how long", "warranty",
                         "return window", "shipping charge", "delivery charge", "plan cost", "premium plan",
                         "emi", "gst", "coupon", "procedure", "process for", "sla", "escalat", "can i", "can a",
                         "can we", "kitne din", "refund to", "refund take", "extended warranty", "price match"],
}
DIAGNOSE = ["why", "reason", "explain", "root cause"]
ACTION = ["what should we do", "recommend", "action", "suggest", "fix", "improve", "next steps"]
OVERVIEW = ["how is the business", "how is business", "overall", "summary", "overview", "dashboard",
            "business health", "state of the business", "how are we doing"]


def normalise(q: str) -> str:
    q = " " + q.lower() + " "
    for hi, en in HINGLISH.items():
        q = re.sub(rf"\b{hi}\b", f" {en} ", q)
    return re.sub(r"\s+", " ", q)


def entities(q: str) -> dict:
    e = {}
    for r in REGIONS:
        if re.search(rf"\b{r.lower()}\b", q):
            e["region"] = r
    for c in CATEGORIES:
        if c.lower() in q or (c == "Clothing" and "cloth" in q) or (c == "Office Supplies" and "office" in q):
            e["category"] = c
    for s in SEGMENTS:
        if s.lower() in q or (s == "Champions" and "champion" in q):
            e["segment"] = s
    for t in TICKET_CATS:
        if t.lower() in q or (t == "Product Quality" and "quality" in q):
            e["ticket_category"] = t
    m = re.search(r"\bc\d{5}\b", q)
    if m:
        e["customer_id"] = m.group(0).upper()
    m = re.search(r"\btop (\d{1,2})\b", q)
    if m:
        e["top_n"] = int(m.group(1))
    return e


def plan(question: str) -> dict:
    q = normalise(question)
    e = entities(q)
    scores = {tool: sum(k in q for k in kws) for tool, kws in INTENTS.items()}
    reasons, calls = [], []

    def add(tool, why, **kw):
        if tool not in [c["tool"] for c in calls]:
            calls.append({"tool": tool, "args": {k: v for k, v in kw.items() if v is not None}})
            reasons.append(f"{tool}: {why}")

    region, top_n = e.get("region"), e.get("top_n")
    if "customer_id" in e:
        add("customer_lookup", f"customer id {e['customer_id']} in the question", customer_id=e["customer_id"])

    analytic = {t: s for t, s in scores.items() if t != "search_documents" and s > 0}
    doc_like = scores["search_documents"] > 0 and (scores["search_documents"] >= max(analytic.values(), default=0))

    if any(k in q for k in OVERVIEW):
        add("sales_forecast", "overview question", region=region)
        add("sales_trend", "overview question", region=region)
        add("churn_risk", "overview question", region=region, top_n=top_n)
        add("ticket_insights", "overview question", region=region)
    if any(k in q for k in DIAGNOSE) and (scores["sales_trend"] or region or e.get("category")) and not doc_like:
        add("sales_trend", "asks WHY sales changed", region=region, category=e.get("category"))
        add("churn_risk", "diagnosis: are customers leaving?", region=region, top_n=top_n)
        add("ticket_insights", "diagnosis: are customers unhappy?", region=region)
    if any(k in q for k in ACTION) and not doc_like:
        add("churn_risk", "asks for actions -> who is at risk", region=region, segment=e.get("segment"), top_n=top_n)
        add("urgent_calls", "asks for actions -> who to call first", region=region, top_n=top_n)

    # a policy question ("SLA for a medium priority ticket") should not also trigger analytics tools
    # just because a word like "ticket" appears (doc_like = policy words at least as strong as analytics words)
    for tool, s in sorted(analytic.items(), key=lambda x: -x[1]):
        if doc_like:
            break
        kw = {"region": region}
        if tool == "sales_trend":
            kw["category"] = e.get("category")
        if tool == "churn_risk":
            kw.update(segment=e.get("segment"), top_n=top_n)
        if tool == "ticket_insights":
            kw = {"category": e.get("ticket_category"), "region": region}
        if tool in ("urgent_calls", "order_anomalies"):
            kw["top_n"] = top_n
        if tool == "customer_segments":
            kw = {}
        add(tool, f"keywords matched ({s})", **kw)

    if doc_like or not calls:
        add("search_documents", "policy-style question" if doc_like else "no analytics intent found -> search documents",
            question=question)
    return {"question": question, "normalised": q.strip(), "entities": e, "calls": calls, "reasons": reasons}
