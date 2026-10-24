"""
Step 7b: Evaluate the RAG system - you can't improve what you don't measure.

Test set: 37 questions with a known correct document + section + key fact, of 4 kinds
   direct      - uses the document's own words        ("return window for electronics")
   paraphrase  - different words, same meaning        ("parcel came broken, deadline to tell you?")
   hinglish    - how staff actually type               ("refund card mein kitne din mein aayega")
   report      - numbers from the Q3 review, which change every time the pipeline runs
plus 6 out-of-scope questions whose answer is NOT in any document (must abstain).

Metrics
   hit@1 / hit@3 - was the right section ranked 1st / in the top 3?
   MRR           - mean of 1/rank of the right section (1.0 = always first)
   answer_ok     - does the final answer contain the key fact?
   abstain       - out-of-scope questions correctly refused, in-scope ones wrongly refused

Outputs: models/rag_config.json (best method + abstain threshold), reports/rag_metrics.json
"""
import json
import math

import numpy as np
import pandas as pd

from src.config import MODELS, REPORTS
from src.rag.rag_system import RAG, RETRIEVERS, BM25Retriever, expand_query, ingest, extractive_answer

R, S, W, C, P, Q = ("Returns_and_Refund_Policy", "Shipping_and_Delivery_Policy", "Warranty_Policy",
                    "Customer_Support_SOP", "Pricing_Discounts_and_Plans", "Q3_2026_Business_Review")
# (question, kind, doc, section number, key facts - any one must appear in the answer)
QUESTIONS = [
    ("What is the return window for electronics?", "direct", R, "2", ["7 days"]),
    ("Can I return clothes 12 days after delivery?", "direct", R, "2", ["15 days"]),
    ("How long does a refund to a credit card take?", "direct", R, "4", ["5 to 7 business days"]),
    ("How fast is a refund to the Kaveri Wallet?", "direct", R, "4", ["24 hours"]),
    ("Are shipping charges refunded when only part of an order is returned?", "direct", R, "5", ["not refunded"]),
    ("What happens if a refund is delayed beyond 15 business days?", "direct", R, "7", ["Rs 200", "Finance Operations"]),
    ("When will I get my money back if I paid with UPI?", "paraphrase", R, "4", ["2 to 3 business days"]),
    ("refund card mein kitne din mein aayega", "hinglish", R, "4", ["5 to 7"]),
    ("What is the shipping charge for orders below Rs 499?", "direct", S, "3", ["Rs 49"]),
    ("How long does delivery take in metro cities?", "direct", S, "2", ["2 to 3 business days"]),
    ("Do Premium customers pay for express delivery?", "direct", S, "3", ["twice per month"]),
    ("Within how many hours must damage in transit be reported?", "direct", S, "5", ["48 hours"]),
    ("My parcel came broken, what is the deadline to tell you?", "paraphrase", S, "5", ["48 hours"]),
    ("Can the delivery address be changed after the order is shipped?", "direct", S, "6", ["same city"]),
    ("order 7 din se zyada late hai toh kya milega", "hinglish", S, "4", ["5 percent"]),
    ("How long is the warranty on electronics?", "direct", W, "1", ["1 year"]),
    ("What is the warranty period for furniture?", "direct", W, "1", ["6 month"]),
    ("How much does Kaveri Care extended warranty cost?", "direct", W, "2", ["8 percent"]),
    ("Is water damage covered under warranty?", "direct", W, "3", ["water damage"]),
    ("My laptop broke the same way three times, what happens now?", "paraphrase", W, "5", ["replaced with a new unit"]),
    ("warranty claim kaise karu", "hinglish", W, "4", ["Product Quality", "invoice"]),
    ("What is the first response time for High priority tickets?", "direct", C, "1", ["2 hours"]),
    ("Who must approve a refund of Rs 25,000?", "direct", C, "5", ["Support Manager"]),
    ("When should a ticket be escalated to Level 2?", "direct", C, "3", ["three or more tickets"]),
    ("How quickly must customers on the urgent call list be called?", "direct", C, "4", ["1 business day"]),
    ("What can we offer a customer who is about to leave us?", "paraphrase", C, "4", ["10 percent", "Premium plan free"]),
    ("Biggest refund a level one agent can sign off?", "paraphrase", C, "5", ["Rs 2,000"]),
    ("How much does the Premium plan cost?", "direct", P, "1", ["Rs 2,499"]),
    ("What discount can a regional sales manager approve?", "direct", P, "2", ["12 percent"]),
    ("Can I use two coupons on one order?", "direct", P, "3", ["one coupon"]),
    ("Can a GSTIN be added after the invoice is generated?", "direct", P, "4", ["cannot be added"]),
    ("Is no-cost EMI available on a Rs 8,000 order?", "direct", P, "5", ["above Rs 10,000"]),
    ("premium plan ka price kitna hai", "hinglish", P, "1", ["2,499"]),
    ("What is the sales forecast for the next 30 days?", "report", Q, "2", ["{forecast}"]),
    ("How many customers are at high churn risk?", "report", Q, "3", ["{high_risk}"]),
    ("Which region has the highest forecast revenue?", "report", Q, "2", ["North"]),
    ("Which customer segment brings the most revenue?", "report", Q, "4", ["Champions"]),
]
OUT_OF_SCOPE = ["What is the CEO's salary?", "How do I apply for a job at Kaveri Retail?",
                "What is the weather in Mumbai today?", "Does Kaveri Retail sell groceries?",
                "What is the company share price?", "Who won the cricket world cup?"]


def report_facts():
    out = {}
    try:
        fc = json.loads((REPORTS / "forecast_metrics.json").read_text())
        out["forecast"] = f"{fc['forecast_total_inr'] / 1e5:,.1f} lakh"
        out["high_risk"] = str(json.loads((REPORTS / "churn_metrics.json").read_text())["high_risk_customers"])
    except FileNotFoundError:
        pass
    return out


def rank_of_gold(scores, chunks, doc, sec):
    order = np.argsort(-scores)
    for r, i in enumerate(order, start=1):
        if chunks[i].doc == doc and chunks[i].section.split(".")[0] == sec:
            return r
    return 999


def main():
    print("[Step 7] Evaluating RAG")
    chunks = ingest()
    texts = [f"{c.section} {c.text}" for c in chunks]
    facts = report_facts()
    qs = [(q, k, d, s, [f.format(**facts) for f in fs]) for q, k, d, s, fs in QUESTIONS
          if all("{" not in f or f.strip("{}") in facts for f in fs)]
    print(f"  {len(chunks)} chunks from {len({c.doc for c in chunks})} documents | "
          f"{len(qs)} test questions + {len(OUT_OF_SCOPE)} out-of-scope")

    # ---- retrieval: every method, with and without Hinglish query expansion
    rows = []
    for name, make in RETRIEVERS.items():
        ret = make().fit(texts)
        for expand in (False, True):
            for q, kind, doc, sec, keys in qs:
                qq = expand_query(q) if expand else q
                r = rank_of_gold(ret.scores(qq), chunks, doc, sec)
                rows.append({"method": name, "expand": expand, "kind": kind, "rank": r})
    df = pd.DataFrame(rows)
    df["hit1"], df["hit3"], df["rr"] = df["rank"] == 1, df["rank"] <= 3, 1 / df["rank"]
    table = df.groupby(["method", "expand"])[["hit1", "hit3", "rr"]].mean().round(3)
    by_kind = df[df["expand"]].pivot_table(index="method", columns="kind", values="hit3").round(2)
    print("  Retrieval (all questions):\n" + table.rename(columns={"rr": "MRR"}).to_string())
    print("  hit@3 by question type (with expansion):\n" + by_kind.to_string())

    # ---- abstain threshold from BM25 relevance (in-scope vs out-of-scope)
    gate = BM25Retriever().fit(texts)
    rel_in = np.array([gate.scores(expand_query(q)).max() for q, *_ in qs])
    rel_out = np.array([gate.scores(expand_query(q)).max() for q in OUT_OF_SCOPE])
    # threshold = the one that gets the most questions right (answer the in-scope ones, refuse the rest);
    # on ties prefer the lower one (refusing a real question is worse than a weak answer with a citation)
    cands = np.unique(np.concatenate([rel_in, rel_out, [0.0]]))
    correct = [((rel_in >= t).sum() + (rel_out < t).sum(), -t) for t in cands]
    thr = float(cands[int(np.argmax([c[0] * 1000 + c[1] for c in correct]))])
    thr = math.floor(thr * 100) / 100          # round DOWN, so the question sitting exactly on it is still answered
    print(f"  relevance: in-scope min {rel_in.min():.2f}, out-of-scope max {rel_out.max():.2f} -> threshold {thr:.2f} "
          f"({(rel_in >= thr).sum()}/{len(rel_in)} in-scope answered, {(rel_out < thr).sum()}/{len(rel_out)} refused)")

    # ---- end-to-end answers (no LLM). Pick retriever + expansion by answer accuracy,
    # ties -> higher MRR -> simpler method
    def run(mode, method, expand):
        rag = RAG(method=method, expand=expand, min_score=thr, mode=mode)
        out = []
        for q, kind, doc, sec, keys in qs:
            a = rag.ask(q, use_llm=False)
            ok = a["found"] and any(k.lower() in a["answer"].lower() for k in keys)
            out.append({"question": q, "kind": kind, "found": a["found"], "answer_ok": ok, "answer": a["answer"]})
        for q in OUT_OF_SCOPE:
            a = rag.ask(q, use_llm=False)
            out.append({"question": q, "kind": "out_of_scope", "found": a["found"], "answer_ok": not a["found"],
                        "answer": a["answer"]})
        return pd.DataFrame(out)

    simplicity = ["tfidf", "bm25", "char", "lsa", "hybrid"]
    inscope_acc = lambda d: float(d[d.kind != "out_of_scope"]["answer_ok"].mean())
    grid = {(m, e): inscope_acc(run("section", m, e)) for m in simplicity for e in (False, True)}
    best_method, best_expand = max(grid, key=lambda me: (round(grid[me], 3), table["rr"][me], -simplicity.index(me[0])))
    print("  End-to-end answer accuracy (section mode): " +
          ", ".join(f"{m}{'+exp' if e else ''} {v:.0%}" for (m, e), v in grid.items()))
    print(f"  selected: {best_method} (query expansion: {best_expand})")
    modes = {m: run(m, best_method, best_expand) for m in ("sentences", "section")}
    mode_acc = {m: round(float(d[d.kind != "out_of_scope"]["answer_ok"].mean()), 3) for m, d in modes.items()}
    best_mode = max(mode_acc, key=mode_acc.get)
    print(f"  Answer mode accuracy (in-scope): {mode_acc} -> using '{best_mode}'")
    MODELS.mkdir(exist_ok=True)
    (MODELS / "rag_config.json").write_text(json.dumps(
        {"method": best_method, "query_expansion": best_expand, "min_bm25_score": round(thr, 2),
         "answer_mode": best_mode}, indent=2))
    ans = modes[best_mode]
    acc = ans.groupby("kind")["answer_ok"].mean().round(2).to_dict()
    inscope = ans[ans.kind != "out_of_scope"]
    summary = {"chunks": len(chunks), "questions": len(qs), "out_of_scope": len(OUT_OF_SCOPE),
               "retrieval": {f"{m}{'+expand' if e else ''}": v for (m, e), v in
                             table.rename(columns={"rr": "mrr"}).to_dict(orient="index").items()},
               "hit3_by_kind_with_expansion": by_kind.to_dict(orient="index"),
               "selected": {"method": best_method, "query_expansion": best_expand, "abstain_threshold": round(thr, 2),
                            "answer_mode": best_mode},
               "answer_mode_accuracy": mode_acc,
               "end_to_end_by_method": {f"{m}{'+expand' if e else ''}": round(v, 3) for (m, e), v in grid.items()},
               "answer_accuracy_in_scope": round(float(inscope["answer_ok"].mean()), 3),
               "answer_accuracy_by_kind": acc,
               "wrongly_refused_in_scope": int((~inscope["found"]).sum()),
               "correctly_refused_out_of_scope": int(ans[ans.kind == "out_of_scope"]["answer_ok"].sum()),
               "failures": ans[~ans["answer_ok"]][["question", "kind", "answer"]].to_dict(orient="records")}
    (REPORTS / "rag_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"  Answers: {summary['answer_accuracy_in_scope']:.0%} of in-scope questions answered with the right fact "
          f"| by type {acc}")
    print(f"  Out-of-scope refused: {summary['correctly_refused_out_of_scope']}/{len(OUT_OF_SCOPE)} | "
          f"in-scope wrongly refused: {summary['wrongly_refused_in_scope']}")
    for f in summary["failures"]:
        print(f"    MISS [{f['kind']}] {f['question']}\n         -> {f['answer'][:160]}")
    return summary


if __name__ == "__main__":
    main()
