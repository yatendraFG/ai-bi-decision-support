"""
Step 8c: Evaluate the agent's planner.

A planner can "look smart" on the examples it was built with. So the test set below was
written separately and NOT used while writing the keyword lists (dev examples live in the
notebook). For each question we write which tools MUST be called and which entities
(region, customer id, ...) must be picked up.

Metrics
  tool_recall      - share of required tools that were called
  tool_precision   - share of called tools that were required or acceptable
  exact            - all required tools called and nothing unrelated
  entity_accuracy  - region / category / customer id / top_n extracted correctly
  runs_ok          - full report generated without a tool error

Output: reports/agent_metrics.json
"""
import json

import pandas as pd

from src.config import REPORTS
from src.agent.agent import run_planner
from src.agent.planner import plan

DIAG = {"sales_trend", "churn_risk", "ticket_insights"}
OVERVIEW = {"sales_forecast", "sales_trend", "churn_risk", "ticket_insights"}
# (question, required tools, acceptable extra tools, expected entities)
TEST = [
    ("What revenue do we expect over the coming month?", {"sales_forecast"}, set(), {}),
    ("Give me the sales projection for the East region", {"sales_forecast"}, set(), {"region": "East"}),
    ("agle mahine North mein kitni sales hogi?", {"sales_forecast"}, set(), {"region": "North"}),
    ("Why did South revenue decline last quarter?", DIAG, {"urgent_calls"}, {"region": "South"}),
    ("Explain the drop in Electronics sales", {"sales_trend"}, DIAG, {"category": "Electronics"}),
    ("How did Furniture perform compared to last quarter?", {"sales_trend"}, set(), {"category": "Furniture"}),
    ("East ki sales kyun badh rahi hai?", {"sales_trend"}, DIAG, {"region": "East"}),
    ("Which customers might stop buying from us?", {"churn_risk"}, {"urgent_calls"}, {}),
    ("Show the top 10 customers at risk of churn in West", {"churn_risk"}, {"urgent_calls"}, {"region": "West", "top_n": 10}),
    ("How many Champions are likely to leave?", {"churn_risk"}, set(), {"segment": "Champions"}),
    ("kaun se grahak chhod sakte hain?", {"churn_risk"}, {"urgent_calls"}, {}),
    ("What kinds of customers do we have?", {"customer_segments"}, set(), {}),
    ("Which customer group brings the most money?", {"customer_segments"}, set(), {}),
    ("What are people unhappy about in support tickets?", {"ticket_insights"}, set(), {}),
    ("Are Billing complaints getting worse?", {"ticket_insights"}, set(), {"ticket_category": "Billing"}),
    ("Delivery ki shikayat kya hai?", {"ticket_insights"}, set(), {"ticket_category": "Delivery"}),
    ("Who should the retention team call first today?", {"urgent_calls"}, {"churn_risk"}, {}),
    ("List 5 urgent customers to contact in North", {"urgent_calls"}, {"churn_risk"}, {"region": "North"}),
    ("Are there any suspicious or fraudulent orders?", {"order_anomalies"}, set(), {}),
    ("Show strange orders with wrong prices in South", {"order_anomalies"}, set(), {"region": "South"}),
    ("Give me the full profile of C00509", {"customer_lookup"}, set(), {"customer_id": "C00509"}),
    ("Is customer C02222 likely to churn?", {"customer_lookup"}, {"churn_risk"}, {"customer_id": "C02222"}),
    ("What is the return window for clothing?", {"search_documents"}, set(), {}),
    ("Who can approve a 15% discount?", {"search_documents"}, set(), {}),
    ("What is the SLA for a medium priority ticket?", {"search_documents"}, set(), {}),
    ("warranty kitne din ki hai furniture par?", {"search_documents"}, set(), {}),
    ("How is the business doing overall?", OVERVIEW, set(), {}),
    ("Give me a quick summary of the business for the board", OVERVIEW, set(), {}),
    ("Why are West sales down and what should we do about it?", DIAG | {"urgent_calls"}, set(), {"region": "West"}),
    ("North mein sales kyun gir rahi hai, kya karein?", DIAG | {"urgent_calls"}, set(), {"region": "North"}),
]


def main():
    print("[Step 8] Evaluating the agent planner")
    rows = []
    for q, req, ok_extra, ents in TEST:
        p = plan(q)
        called = {c["tool"] for c in p["calls"]}
        allowed = req | ok_extra
        recall = len(called & req) / len(req)
        precision = len(called & allowed) / max(len(called), 1)
        ent_ok = all(p["entities"].get(k) == v for k, v in ents.items())
        rows.append({"question": q, "required": sorted(req), "called": sorted(called), "recall": recall,
                     "precision": precision, "exact": recall == 1 and precision == 1, "entities_ok": ent_ok,
                     "hinglish": any(w in q.lower() for w in ["kyun", "kitni", "kaun", "ki ", "kya", "kitne", "mein"])})
    df = pd.DataFrame(rows)

    # end-to-end: every plan must produce a report without tool errors
    errors = 0
    for q, *_ in TEST:
        r = run_planner(q, save_report=False)
        errors += any("tool failed" in v["summary"] for v in r["results"].values())

    m = {"questions": len(df), "tool_recall": round(df.recall.mean(), 3), "tool_precision": round(df.precision.mean(), 3),
         "exact_plan": round(df.exact.mean(), 3), "entity_accuracy": round(df.entities_ok.mean(), 3),
         "exact_plan_english": round(df[~df.hinglish].exact.mean(), 3),
         "exact_plan_hinglish": round(df[df.hinglish].exact.mean(), 3),
         "runs_ok": len(TEST) - errors,
         "misses": df[~(df.exact & df.entities_ok)][["question", "required", "called", "entities_ok"]].to_dict(orient="records")}
    (REPORTS / "agent_metrics.json").write_text(json.dumps(m, indent=2))
    print(f"  {m['questions']} unseen questions | exact plan {m['exact_plan']:.0%} (English {m['exact_plan_english']:.0%}, "
          f"Hinglish {m['exact_plan_hinglish']:.0%}) | tool recall {m['tool_recall']:.0%}, precision "
          f"{m['tool_precision']:.0%} | entities {m['entity_accuracy']:.0%} | reports without errors {m['runs_ok']}/{len(TEST)}")
    for x in m["misses"]:
        print(f"    MISS {x['question']}\n         need {x['required']} got {x['called']} entities_ok={x['entities_ok']}")
    return m


if __name__ == "__main__":
    main()
