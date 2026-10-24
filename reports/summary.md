# Business summary (auto-generated)

## Sales forecast
- Next 30 days (2026-10-01 -> 2026-10-30): **INR 689.3 lakh**, +28.1% vs last 30 days.
- By region: North INR 233.2 lakh, South INR 197.4 lakh, West INR 157.8 lakh, East INR 100.9 lakh
- Accuracy (12 rolling 30-day windows, 2025-10-06 -> 2026-09-30): monthly error (WAPE) 7.9% vs 8.6% for a naive 'last 30 days' guess.

## Churn risk
- Model: logistic_regression; on unseen data the top-10% risk list was 48% accurate (7.26x better than random) and caught 72% of customers who left.
- Today: **213 high-risk customers**, INR 86.1 lakh of annual revenue at risk.
- Strongest warning signs: neg_tickets_90d, tickets_90d, tenure_days, orders_90d

## Customer segments
- 5 segments. Largest revenue share: **Champions** (491 customers, 47.8% of revenue).
  - Champions: 491 customers, 47.8% revenue -> Reward & upsell: loyalty perks, early access, premium plan offers
  - Big-ticket buyers: 285 customers, 32.4% revenue -> Account management: dedicated contact, bulk pricing, quarterly reviews
  - Occasional buyers: 692 customers, 12.4% revenue -> Nudge frequency: personalised offers, reminders around festive season
  - Loyal regulars: 415 customers, 6.6% revenue -> Cross-sell new categories, referral programme
  - Slipping away: 237 customers, 0.8% revenue -> Win-back campaign now: discount + support follow-up on open complaints

## Deep learning (Step 5)
- Forecast models on the same 12 folds (monthly WAPE): lstm 6.9%, ensemble 7.4%, gradient_boosting 7.9%, naive_last30 8.6%, seasonal_naive 8.7%
- Best: **lstm**.
- Order anomalies: autoencoder flagged-list precision 78% vs Isolation Forest 43% (on injected test anomalies); **694 real orders** flagged for review.

## Support tickets (Step 6)
- Category routing: 94% correct (tfidf_lr). Sentiment: 85% correct, macro-F1 0.82 (tfidf_lr_priority).
- Human-in-the-loop: 79% of tickets auto-tagged at 94% accuracy; the rest go to review.
- Complaint trend: **Product Quality** negative share 36% -> 39% (stable).
- **Call today: 87 high-churn-risk customers** with a recent negative ticket (INR 29.2 lakh at risk) -> data/processed/urgent_tickets.csv

## Document Q&A (Step 7)
- 42 chunks from the PDFs in data/docs; retriever tfidf + Hinglish expansion, answers as 'section'.
- Test set: 89% of 37 questions answered with the right fact; 5/6 out-of-scope questions refused. Ask your own: ask_documents.bat or `python -m src.rag.ask`.

## AI agent (Step 8)
- Planner picks the right tools for 90% of 30 unseen test questions (Hinglish 100%); entities 100%.
- Latest overview report: reports/agent/latest_business_overview.md. Ask your own: ask_agent.bat or `python -m src.agent.ask_agent`.
- Top actions: Sales grew 3.4% last quarter - protect the growth: check stock for the top categories before the festive peak. | Routine: retention calls for High-risk customers, beginning with the biggest revenue at risk (e.g. C01658, Champions). SOP allows up to 10% discount or 2 months Premium free. | 87 high-churn-risk customers raised a negative ticket recently - call them within 1 business day (Support SOP section 4).

## Dashboard + API (Step 9)
- All API endpoint checks passed. Dashboard: run_dashboard.bat (or `streamlit run app/dashboard.py`); API: run_api.bat, docs at http://127.0.0.1:8000/docs.

Files: data/processed/ (sales_forecast, churn_scores, customer_segments, customer_360, order_anomalies, ticket_predictions, urgent_tickets)