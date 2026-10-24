"""
Step 8b: The AI agent - question in, business report out.

    question -> plan (which tools?) -> run tools -> report (findings + actions + sources)

Two brains, same tools (src/agent/tools.py):
  * "planner" (default, offline) - the rule-based router in planner.py. Transparent: the report
    shows which tools ran and why. Actions are rule-based suggestions built from the tool numbers.
  * "claude" (optional) - Claude chooses the tools itself in a tool-use loop and writes the report.
    Enable with:  pip install anthropic  +  set ANTHROPIC_API_KEY. The loop is tested with a fake
    client (tests the plumbing), but was not run against the real API in this project.

Every report is saved to reports/agent/ as Markdown.

CLI:  python -m src.agent.agent "West region ki sales kyun gir rahi hai aur kya karein?"
"""
import json
import os
import re
import sys
from datetime import datetime

from src.config import REPORTS
from src.agent.planner import plan as make_plan
from src.agent.tools import TOOLS, TOOL_SPECS

OUT = REPORTS / "agent"


# ---------------------------------------------------------------- rule-based recommendations
def recommend(results: dict) -> list:
    acts = []
    tr = results.get("sales_trend")
    if tr and tr.get("change_pct") is not None:
        drivers = tr["data"]
        worst = drivers[0] if drivers else None
        key = "category" if worst and "category" in worst else "region"
        if tr["change_pct"] <= -3 and worst:
            acts.append(f"Sales fell {abs(tr['change_pct']):.1f}% last quarter, mostly in **{worst[key]}** "
                        f"({worst['change_pct']:+.1f}%). Review {worst[key]} pricing, stock and competitor offers "
                        f"with the category manager this week.")
        elif tr["change_pct"] >= 3:
            acts.append(f"Sales grew {tr['change_pct']:.1f}% last quarter - protect the growth: check stock for the "
                        f"top categories before the festive peak.")
    ch = results.get("churn_risk")
    if ch and ch.get("data"):
        top = ch["data"][0]
        above = ch.get("high_risk_share", 0) > 1.2 * ch.get("company_high_risk_share", 1)
        acts.append(("Priority: " if above else "Routine: ") +
                    f"retention calls for High-risk customers, beginning with the biggest revenue at risk "
                    f"(e.g. {top['customer_id']}, {top['segment_name']}). SOP allows up to 10% discount or "
                    f"2 months Premium free.")
    uc = results.get("urgent_calls")
    if uc and uc.get("data"):
        acts.append(f"{uc['summary'].split(' raised')[0]} raised a negative ticket recently - call them within "
                    f"1 business day (Support SOP section 4).")
    ti = results.get("ticket_insights")
    if ti and ti.get("data"):
        trend = ti["data"]["trend"]
        worse = [t for t in trend if t["change_pp"] > 0]
        if worse:
            w = max(worse, key=lambda t: t["change_pp"])
            theme = (ti["data"]["themes"].get(w["category"]) or [{"phrase": "-"}])[0]["phrase"]
            acts.append(f"Negative tickets are rising in **{w['category']}** ({w['negative_share_prev_9m']:.0%} -> "
                        f"{w['negative_share_last_3m']:.0%}); top complaint: \"{theme}\". Assign an owner to fix it.")
    fc = results.get("sales_forecast")
    if fc and fc.get("data") and "+" in fc["summary"].split("vs the last")[0][-12:]:
        top_region = max(fc["data"], key=fc["data"].get)
        acts.append(f"Demand is forecast to rise; prioritise inventory and delivery capacity in {top_region}.")
    cl = results.get("customer_lookup")
    if cl and cl.get("data") and cl["data"].get("risk_band") == "High":
        d = cl["data"]
        acts.append(f"{d['customer_id']} is High churn risk ({d['churn_probability']:.0%}) with "
                    f"Rs {d['revenue_at_risk_365d'] / 1e5:,.1f} lakh at risk: call within 1 business day, resolve the "
                    f"latest ticket first, then offer the retention benefit allowed by the SOP.")
    an = results.get("order_anomalies")
    if an and an.get("data"):
        acts.append("Review the top flagged orders before invoicing/shipping - most are unusual quantities or "
                    "prices vs the customer's history.")
    return acts


def interpret(results: dict) -> list:
    """Turn comparisons into conclusions - including what is NOT the cause."""
    out = []
    tr, ch, ti = results.get("sales_trend"), results.get("churn_risk"), results.get("ticket_insights")
    if tr and tr.get("customers_change_pct") is not None and tr["change_pct"] < -3:
        c, rpc = tr["customers_change_pct"], tr["revenue_per_customer_change_pct"]
        if abs(rpc) > abs(c):
            out.append(f"The drop is mainly **lower spend per customer** ({rpc:+.1f}%), not fewer customers "
                       f"({c:+.1f}% active customers) - look at basket size, prices and big-ticket orders.")
        else:
            out.append(f"The drop is mainly **fewer active customers** ({c:+.1f}%), not smaller baskets "
                       f"({rpc:+.1f}% per customer) - look at acquisition and retention.")
    if ch and ch.get("high_risk_share") is not None and ch["high_risk_share"] != ch["company_high_risk_share"]:
        a, b = ch["high_risk_share"], ch["company_high_risk_share"]
        if a > b * 1.2:
            out.append(f"Churn risk here is **above** the company average ({a:.0%} vs {b:.0%}) - retention is a real lever.")
        elif a < b * 0.8:
            out.append(f"Churn risk here is **below** the company average ({a:.0%} vs {b:.0%}).")
        else:
            out.append(f"Churn risk is in line with the company average ({a:.0%} vs {b:.0%}) - probably not the main cause.")
    if ti and ti.get("region_negative_share") is not None:
        a, b = ti["region_negative_share"], ti["company_negative_share"]
        if a < b:
            out.append(f"Support complaints are **not** the driver: negative tickets are lower here than overall ({a:.0%} vs {b:.0%}).")
        else:
            out.append(f"Customers here complain more than average ({a:.0%} vs {b:.0%} negative tickets) - service may be hurting sales.")
    return out


def write_report(question, plan, results, actions, brain="planner"):
    lines = [f"# {question}", "", f"*Generated {datetime.now():%d %b %Y %H:%M} by the AI BI agent ({brain}).*", ""]
    if plan:
        lines += ["**How I answered:** " + "; ".join(f"`{c['tool']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'question')})`"
                                                    for c in plan["calls"]), ""]
    lines += ["## Findings", ""]
    for tool, r in results.items():
        lines += [f"**{tool.replace('_', ' ').capitalize()}** - {r['summary']}  ", f"<sub>source: {r['source']}</sub>", ""]
    insights = interpret(results)
    if insights:
        lines += ["## What this means", "", *[f"- {i}" for i in insights], ""]
    if actions:
        lines += ["## Recommended actions", "", *[f"{i}. {a}" for i, a in enumerate(actions, 1)], "",
                  "<sub>Actions are rule-based suggestions from the numbers above; review before acting.</sub>"]
    return "\n".join(lines)


def save(question, text):
    OUT.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower())[:40].strip("_")
    path = OUT / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.md"
    path.write_text(text, encoding="utf-8")
    return path


HELP = """**I could not match this to an analysis or a document.** I can answer questions like:
- Sales: *"What is the forecast for next month?"*, *"Why did West sales fall?"*
- Customers: *"Which customers are likely to leave in North?"*, *"Which segment is most valuable?"*, *"Tell me about C01658"*
- Support: *"What are customers complaining about?"*, *"Who should we call today?"*
- Risk: *"Show suspicious orders"*
- Policies: *"Who approves a refund of Rs 25,000?"*, *"refund card mein kitne din mein aayega?"*
- Everything: *"How is the business doing overall?"*"""


# ---------------------------------------------------------------- planner brain
def run_planner(question: str, save_report=True) -> dict:
    p = make_plan(question)
    results = {}
    for call in p["calls"]:
        try:
            results[call["tool"]] = TOOLS[call["tool"]](**call["args"])
        except Exception as e:                              # one broken tool must not kill the report
            results[call["tool"]] = {"summary": f"(tool failed: {type(e).__name__}: {e})", "data": None, "source": None}
    actions = recommend(results)
    text = write_report(question, p, results, actions)
    only_docs = list(results) == ["search_documents"]
    if only_docs and not (results["search_documents"]["data"] or {}).get("found", True):
        text += "\n\n" + HELP
    path = save(question, text) if save_report else None
    return {"plan": p, "results": results, "actions": actions, "report": text, "path": path}


# ---------------------------------------------------------------- Claude brain (optional)
SYSTEM = ("You are a business-intelligence analyst for Kaveri Retail. Answer the user's question by calling the "
          "tools - never invent numbers; every number must come from a tool result. Call several tools when a "
          "question needs diagnosis (e.g. why sales fell -> trend, churn, tickets). Then write a short report: "
          "'## Findings' (bullets with numbers and the tool used) and '## Recommended actions' (numbered, concrete, "
          "owner + timeframe). Reply in the user's language (English or Hinglish).")


def run_claude(question: str, client=None, model=None, max_turns=6, save_report=True) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
    messages, used = [{"role": "user", "content": question}], []
    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=1500, system=SYSTEM, tools=TOOL_SPECS, messages=messages)
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            report = f"# {question}\n\n*Generated by the AI BI agent (claude). Tools used: {', '.join(used) or 'none'}.*\n\n{text}"
            return {"report": report, "tools_used": used, "path": save(question, report) if save_report else None}
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                used.append(b.name)
                try:
                    out = TOOLS[b.name](**b.input)
                    content = json.dumps({"summary": out["summary"], "data": out["data"]}, default=str)[:6000]
                except Exception as e:
                    content = json.dumps({"error": f"{type(e).__name__}: {e}"})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("agent did not finish within max_turns")


def ask(question: str, brain: str = None) -> dict:
    brain = brain or ("claude" if os.environ.get("ANTHROPIC_API_KEY") else "planner")
    if brain == "claude":
        try:
            return run_claude(question)
        except Exception as e:
            print(f"  (Claude unavailable: {type(e).__name__}; using the offline planner)")
    return run_planner(question)


if __name__ == "__main__":
    qs = sys.argv[1:] or ["West region ki sales kyun gir rahi hai aur kya karein?"]
    for q in qs:
        r = ask(q)
        print(r["report"])
        print(f"\n[saved to {r['path']}]\n")
