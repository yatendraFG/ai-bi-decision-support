"""
Runs Steps 2-8 end to end, then checks the Step 9 API and writes a business summary to reports/summary.md.

    python run_pipeline.py                  # generate data + train everything
    python run_pipeline.py --skip-generate  # use your own CSVs in data/raw/
    python run_pipeline.py --skip-dl        # no TensorFlow: skips Step 5 and the Keras NLP model
    python run_pipeline.py --skip-nlp       # skip Step 6
    python run_pipeline.py --skip-rag       # skip Step 7
    python run_pipeline.py --skip-agent     # skip Step 8

Note: if you regenerate data, delete reports/lstm_folds/ so the LSTM folds retrain.
"""
import argparse
import time

from src.config import REPORTS
from src.data import generate_data
from src.ml import churn, forecasting, segmentation


def inr_lakh(x):
    return f"INR {x / 1e5:,.1f} lakh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-dl", action="store_true", help="skip Step 5 (TensorFlow)")
    ap.add_argument("--skip-nlp", action="store_true", help="skip Step 6 (ticket NLP)")
    ap.add_argument("--skip-rag", action="store_true", help="skip Step 7 (document Q&A)")
    ap.add_argument("--skip-agent", action="store_true", help="skip Step 8 (AI agent)")
    ap.add_argument("--skip-api", action="store_true", help="skip the Step 9 API check")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_generate:
        generate_data.main()
    fc = forecasting.main()
    ch = churn.main()
    seg = segmentation.main()
    lstm = anom = None
    if not args.skip_dl:
        from src.dl import anomaly, lstm_forecast   # imported here so Steps 2-4 work without TensorFlow
        lstm = lstm_forecast.main()
        anom = anomaly.main()
    nlp = ins = None
    if not args.skip_nlp:
        from src.nlp import ticket_classifier, ticket_insights
        nlp = ticket_classifier.main(with_keras=not args.skip_dl)
        ins = ticket_insights.main()
    rag = None
    if not args.skip_rag:
        from src.rag import evaluate_rag
        try:
            from src.data import make_company_docs   # refreshes the sample PDFs, incl. the Q3 review numbers
            make_company_docs.main()
        except ImportError:
            print("[Step 7a] reportlab not installed -> using the PDFs already in data/docs")
        rag = evaluate_rag.main()
    agent_m = agent_report = None
    if not args.skip_agent:
        from src.agent import evaluate_agent
        from src.agent.agent import run_planner
        agent_m = evaluate_agent.main()
        agent_report = run_planner("How is the business doing overall and what should we do?")
        (REPORTS / "agent" / "latest_business_overview.md").write_text(agent_report["report"], encoding="utf-8")

    m, nv = fc["metrics"]["model"], fc["metrics"]["naive_last30"]
    t = ch["test_metrics_selected"]
    top_seg = seg["segments"][0]
    lines = [
        "# Business summary (auto-generated)",
        "",
        "## Sales forecast",
        f"- Next 30 days ({fc['forecast_window']}): **{inr_lakh(fc['forecast_total_inr'])}**, "
        f"{fc['forecast_vs_last30_pct']:+.1f}% vs last 30 days.",
        "- By region: " + ", ".join(f"{r} {inr_lakh(v)}" for r, v in
                                    sorted(fc["forecast_by_region_inr"].items(), key=lambda x: -x[1])),
        f"- Accuracy ({fc['backtest']}): monthly error (WAPE) {m['monthly_total_wape']:.1%} "
        f"vs {nv['monthly_total_wape']:.1%} for a naive 'last 30 days' guess.",
        "",
        "## Churn risk",
        f"- Model: {ch['selected_model']}; on unseen data the top-10% risk list was "
        f"{t['precision_top10pct']:.0%} accurate ({t['lift_top10pct']}x better than random) "
        f"and caught {t['recall_top10pct']:.0%} of customers who left.",
        f"- Today: **{ch['high_risk_customers']} high-risk customers**, "
        f"{inr_lakh(ch['revenue_at_risk_inr'])} of annual revenue at risk.",
        "- Strongest warning signs: " + ", ".join(list(ch["top_drivers_permutation_importance"])[:4]),
        "",
        "## Customer segments",
        f"- {seg['k']} segments. Largest revenue share: **{top_seg['segment_name']}** "
        f"({top_seg['customers']} customers, {top_seg['revenue_share_pct']}% of revenue).",
    ]
    for s in seg["segments"]:
        lines.append(f"  - {s['segment_name']}: {s['customers']} customers, "
                     f"{s['revenue_share_pct']}% revenue -> {seg['actions'].get(s['segment_name'].split(' (')[0], '')}")
    if lstm:
        lm = lstm["metrics"]
        best = min(lm, key=lambda k: lm[k]["monthly_total_wape"])
        lines += ["", "## Deep learning (Step 5)",
                  "- Forecast models on the same 12 folds (monthly WAPE): " + ", ".join(
                      f"{k} {v['monthly_total_wape']:.1%}" for k, v in
                      sorted(lm.items(), key=lambda kv: kv[1]["monthly_total_wape"])),
                  f"- Best: **{best}**."]
    if anom:
        a, i = anom["results"]["autoencoder"], anom["results"]["isolation_forest"]
        lines += [f"- Order anomalies: autoencoder flagged-list precision {a['precision_in_flagged']:.0%} "
                  f"vs Isolation Forest {i['precision_in_flagged']:.0%} (on injected test anomalies); "
                  f"**{anom['flagged_real_orders']} real orders** flagged for review."]
    if nlp:
        s_, c_ = nlp["test"]["sentiment"], nlp["test"]["category"]
        hl = s_["human_in_the_loop"]
        lines += ["", "## Support tickets (Step 6)",
                  f"- Category routing: {c_['accuracy']:.0%} correct ({c_['model']}). "
                  f"Sentiment: {s_['accuracy']:.0%} correct, macro-F1 {s_['macro_f1']:.2f} ({s_['model']}).",
                  f"- Human-in-the-loop: {hl['auto_handled_share']:.0%} of tickets auto-tagged at "
                  f"{hl['accuracy_auto_handled']:.0%} accuracy; the rest go to review."]
    if ins:
        worst = ins["category_trend"][0]
        lines += [f"- Complaint trend: **{worst['category']}** negative share {worst['negative_share_prev_9m']:.0%} -> "
                  f"{worst['negative_share_last_3m']:.0%} ({worst['status']}).",
                  f"- **Call today: {ins['urgent_customers']} high-churn-risk customers** with a recent negative ticket "
                  f"({inr_lakh(ins['urgent_revenue_at_risk_inr'])} at risk) -> data/processed/urgent_tickets.csv"]
    if rag:
        sel = rag["selected"]
        lines += ["", "## Document Q&A (Step 7)",
                  f"- {rag['chunks']} chunks from the PDFs in data/docs; retriever {sel['method']}"
                  f"{' + Hinglish expansion' if sel['query_expansion'] else ''}, answers as '{sel['answer_mode']}'.",
                  f"- Test set: {rag['answer_accuracy_in_scope']:.0%} of {rag['questions']} questions answered with the "
                  f"right fact; {rag['correctly_refused_out_of_scope']}/{rag['out_of_scope']} out-of-scope questions "
                  f"refused. Ask your own: ask_documents.bat or `python -m src.rag.ask`."]
    if agent_m:
        lines += ["", "## AI agent (Step 8)",
                  f"- Planner picks the right tools for {agent_m['exact_plan']:.0%} of {agent_m['questions']} unseen test "
                  f"questions (Hinglish {agent_m['exact_plan_hinglish']:.0%}); entities {agent_m['entity_accuracy']:.0%}.",
                  "- Latest overview report: reports/agent/latest_business_overview.md. "
                  "Ask your own: ask_agent.bat or `python -m src.agent.ask_agent`."]
        if agent_report and agent_report["actions"]:
            lines += ["- Top actions: " + " | ".join(a.replace("**", "") for a in agent_report["actions"][:3])]
    if not args.skip_api:
        try:
            from src.api import test_api
            print("[Step 9] API check")
            test_api.main()
            lines += ["", "## Dashboard + API (Step 9)",
                      "- All API endpoint checks passed. Dashboard: run_dashboard.bat (or `streamlit run app/dashboard.py`); "
                      "API: run_api.bat, docs at http://127.0.0.1:8000/docs."]
        except ImportError as e:
            print(f"[Step 9] skipped - install requirements first ({e.name} missing)")
    lines += ["", "Files: data/processed/ (sales_forecast, churn_scores, customer_segments, customer_360, "
                  "order_anomalies, ticket_predictions, urgent_tickets)"]
    (REPORTS / "summary.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
