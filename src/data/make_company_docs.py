"""
Step 7a: Sample company documents for RAG (fictional company "Kaveri Retail").

Real projects put their own PDFs in data/docs/. These samples exist so the RAG
system has something realistic to read: policies with exact numbers, an SOP,
and a quarterly business review whose numbers come from our own model outputs
(Steps 3-6) - so management can ask questions about the latest analysis too.

Each document uses numbered section headings ("3. Refund Timelines"); the RAG
chunker uses them to keep each answer's context together.

Run:  python -m src.data.make_company_docs     (needs reportlab: pip install reportlab)
"""
import json
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.config import ROOT, REPORTS

DOCS = ROOT / "data" / "docs"
COMPANY = "Kaveri Retail Pvt. Ltd. (fictional company used for this project)"

POLICIES = {
"Returns_and_Refund_Policy": ("Returns and Refund Policy", "RET-01, version 3.2, effective 1 April 2026", [
 ("1. Purpose", "This policy explains when customers can return products, how refunds are paid, and how long "
  "each step takes. It applies to all orders placed on the Kaveri Retail website and app."),
 ("2. Return Window by Category",
  "Electronics can be returned within 7 days of delivery. Clothing can be returned within 15 days of delivery. "
  "Furniture can be returned within 10 days of delivery, only if it is unassembled and in original packaging. "
  "Office Supplies can be returned within 30 days of delivery. Items marked 'Final Sale' cannot be returned. "
  "The return window is counted from the delivery date shown in the tracking history."),
 ("3. Condition of Returned Items",
  "Products must be unused, with all tags, accessories, manuals and the original box. Electronics must be "
  "factory reset and must not have physical damage. If an item fails the quality check at the warehouse, it is "
  "sent back to the customer and no refund is issued."),
 ("4. Refund Timelines",
  "After the returned item passes the quality check, the refund is started within 48 hours. Refunds to the "
  "Kaveri Wallet are credited within 24 hours of approval. Refunds to UPI are credited within 2 to 3 business days. "
  "Refunds to credit or debit cards take 5 to 7 business days, depending on the bank. Cash on Delivery orders are "
  "refunded to the Kaveri Wallet or to a bank account given by the customer."),
 ("5. Partial Refunds",
  "If only part of an order is returned, only the returned items are refunded. Shipping charges are not refunded "
  "unless the item was damaged, defective or wrong. Coupon discounts are adjusted proportionally across items."),
 ("6. Cancellations",
  "Orders can be cancelled free of charge before they are shipped. After shipping, the customer must refuse the "
  "delivery or raise a return. Cancelled prepaid orders are refunded using the same timelines as section 4."),
 ("7. Refund Delays and Escalation",
  "If a refund is not received within 10 business days after approval, the customer should raise a ticket with "
  "category Refund. Support must treat such tickets as High priority. A refund delayed beyond 15 business days must "
  "be escalated to the Finance Operations team, and the customer receives a goodwill credit of Rs 200 in the wallet."),
]),

"Shipping_and_Delivery_Policy": ("Shipping and Delivery Policy", "SHP-02, version 2.4, effective 1 January 2026", [
 ("1. Delivery Coverage", "Kaveri Retail delivers to over 19,000 pin codes across India through three courier "
  "partners. Remote areas in the North-East and Ladakh may take additional time."),
 ("2. Delivery Timelines",
  "Metro cities: 2 to 3 business days. Tier-2 cities: 3 to 5 business days. Other areas: 5 to 8 business days. "
  "Furniture and large items: 7 to 12 business days because they ship from regional warehouses."),
 ("3. Shipping Charges",
  "Orders above Rs 499 ship free for all customers. Orders below Rs 499 have a shipping charge of Rs 49. "
  "Customers on the Premium plan get free shipping on every order with no minimum value. Express delivery "
  "(next day in metro cities) costs Rs 99 and is free for Premium customers twice per month."),
 ("4. Late Deliveries",
  "If an order is not delivered within 3 days after the promised date, the customer can request a full refund of the "
  "shipping charge. If an order is more than 7 days late, support may offer a 5 percent discount coupon on the next "
  "order. The courier partner is penalised under the partner agreement."),
 ("5. Damaged or Missing Items",
  "Damage in transit must be reported within 48 hours of delivery with photos of the package and the product. "
  "Missing items must be reported within 72 hours. Kaveri Retail will send a free replacement or a full refund, "
  "as the customer prefers."),
 ("6. Address Changes",
  "The delivery address can be changed only before the order is shipped. After shipping, the customer must contact "
  "support, and a change is possible only within the same city."),
]),

"Warranty_Policy": ("Warranty Policy", "WAR-01, version 1.9, effective 1 April 2026", [
 ("1. Warranty Periods",
  "Electronics carry a 1 year manufacturer warranty from the date of delivery. Furniture carries a 6 month warranty "
  "against manufacturing defects. Office Supplies carry a 3 month warranty on mechanical items such as staplers and "
  "printers' accessories. Clothing is not covered by warranty but can be returned under the Returns Policy."),
 ("2. Extended Warranty",
  "Customers can buy Kaveri Care extended warranty for electronics within 30 days of delivery. It adds 1 or 2 extra "
  "years and costs 8 percent of the product price per extra year. Premium plan customers get a 10 percent discount "
  "on Kaveri Care."),
 ("3. What Is Not Covered",
  "Physical damage, water damage, damage from incorrect voltage, normal wear and tear, cosmetic scratches that do not "
  "affect function, and products repaired by unauthorised service centres are not covered."),
 ("4. How to Claim",
  "Raise a ticket with category Product Quality, attach the invoice and photos or a short video of the defect. "
  "A technician visit or pickup is scheduled within 3 business days. Repairs are completed within 10 business days; "
  "if a product cannot be repaired, it is replaced, and if a replacement is not available, a full refund is issued."),
 ("5. Repeated Defects",
  "If the same defect occurs three times within the warranty period, the product is replaced with a new unit "
  "without further inspection."),
]),

"Customer_Support_SOP": ("Customer Support Standard Operating Procedure", "SOP-CS-05, version 5.1, effective 1 July 2026", [
 ("1. Ticket Priorities and SLAs",
  "High priority tickets must receive a first response within 2 hours and be resolved within 24 hours. "
  "Medium priority: first response within 8 hours, resolution within 3 business days. "
  "Low priority: first response within 24 hours, resolution within 5 business days. "
  "SLA time counts only business hours, 9 am to 9 pm, Monday to Saturday."),
 ("2. Automatic Tagging",
  "Every new ticket is tagged with a category and sentiment by the AI ticket classifier. Tickets where the model "
  "confidence is low are marked needs_review and must be checked by a Level 1 agent before routing. Agents should "
  "correct wrong tags because corrected tags are used to retrain the model every quarter."),
 ("3. Escalation Matrix",
  "Level 1 agents handle all tickets first. Escalate to a Level 2 specialist if the ticket is not resolved within the "
  "SLA, if the customer has raised three or more tickets in 30 days, or if the sentiment is negative and the "
  "customer is in the High churn-risk band. Escalate to the Support Manager if a refund above Rs 10,000 is requested "
  "or if the customer mentions legal action or social media."),
 ("4. Churn-Risk Call Protocol",
  "Every morning the urgent call list (high churn-risk customers with a negative ticket in the last 60 days) is shared "
  "with the retention team. Each customer must be called within 1 business day. The agent may offer up to a 10 percent "
  "discount or 2 months of Premium plan free. Offers above that need Support Manager approval."),
 ("5. Refund Approval Limits",
  "Level 1 agents can approve refunds up to Rs 2,000. Level 2 specialists can approve up to Rs 10,000. Refunds above "
  "Rs 10,000 need Support Manager approval, and above Rs 50,000 need Finance Head approval."),
 ("6. Tone and Language",
  "Reply in the language the customer used, including Hinglish. Apologise once, explain the next step and give a "
  "specific time. Never promise a timeline shorter than the policy timeline."),
]),

"Pricing_Discounts_and_Plans": ("Pricing, Discounts and Membership Plans", "PRC-03, version 4.0, effective 1 April 2026", [
 ("1. Membership Plans",
  "Basic plan is free. Standard plan costs Rs 999 per year and gives 2 percent cashback on every order. "
  "Premium plan costs Rs 2,499 per year and gives 5 percent cashback, free shipping on every order, two free express "
  "deliveries per month and priority support."),
 ("2. Discount Approval Limits",
  "Sales executives can give up to 5 percent discount on any order. Regional sales managers can approve up to 12 percent. "
  "Discounts above 12 percent, and any discount above 20 percent on Electronics, need approval from the Head of Sales. "
  "Corporate bulk orders above Rs 5 lakh may receive negotiated pricing approved by the Head of Sales."),
 ("3. Coupons",
  "Only one coupon can be used per order. Coupons cannot be combined with bank offers unless the coupon says so. "
  "Coupons expire 30 days after they are issued unless another date is printed. Coupons cannot be exchanged for cash."),
 ("4. GST Invoices",
  "A GST invoice is generated automatically for every order and is available in the Orders section within 24 hours of "
  "delivery. Business customers must enter their GSTIN before placing the order; a GSTIN cannot be added after the "
  "invoice is generated."),
 ("5. EMI Options",
  "No-cost EMI is available on orders above Rs 10,000 on select credit cards for 3 and 6 months. Standard EMI is "
  "available on orders above Rs 3,000 for 3 to 24 months."),
 ("6. Price Match",
  "Kaveri Retail does not offer price matching with other websites. If the price of an item drops within 7 days of "
  "purchase, Premium customers can request the difference as wallet credit."),
]),
}


def business_review() -> tuple:
    """Q3 review built from the pipeline's own reports (falls back to text if reports are missing)."""
    def load(name):
        p = REPORTS / name
        return json.loads(p.read_text()) if p.exists() else None
    fc, ch, seg, ins, nlp = (load(n) for n in ["forecast_metrics.json", "churn_metrics.json",
                                               "segment_profile.json", "ticket_insights.json", "nlp_metrics.json"])
    lakh = lambda x: f"Rs {x / 1e5:,.1f} lakh"
    sec = [("1. Summary",
            "This review covers July to September 2026 and the outlook for October 2026. It is prepared by the "
            "Business Intelligence team using the AI decision support system.")]
    if fc:
        regions = ", ".join(f"{r} {lakh(v)}" for r, v in sorted(fc["forecast_by_region_inr"].items(), key=lambda x: -x[1]))
        sec.append(("2. Sales Outlook",
                    f"Revenue for the next 30 days ({fc['forecast_window']}) is forecast at {lakh(fc['forecast_total_inr'])}, "
                    f"which is {fc['forecast_vs_last30_pct']:+.1f} percent compared with the last 30 days, driven by the "
                    f"festive season. By region: {regions}. The forecast model's average monthly error over the last 12 "
                    f"months was {fc['metrics']['model']['monthly_total_wape']:.1%}."))
    if ch:
        sec.append(("3. Customer Churn",
                    f"The churn model flags {ch['high_risk_customers']} customers as High risk, with "
                    f"{lakh(ch['revenue_at_risk_inr'])} of annual revenue at risk. The strongest warning signs are recent "
                    f"negative support tickets, the number of tickets, short tenure and falling order counts."))
    if seg:
        top = seg["segments"][0]
        sec.append(("4. Customer Segments",
                    f"Customers fall into {seg['k']} segments. {top['segment_name']} is the most valuable segment with "
                    f"{top['customers']} customers contributing {top['revenue_share_pct']} percent of revenue. " +
                    " ".join(f"{s['segment_name']}: {s['customers']} customers, {s['revenue_share_pct']} percent of revenue."
                             for s in seg["segments"][1:])))
    if ins and nlp:
        w = ins["category_trend"][0]
        sec.append(("5. Support Tickets",
                    f"The AI classifier routes tickets to the right team with {nlp['test']['category']['accuracy']:.0%} "
                    f"accuracy. {w['category']} has the highest rise in negative tickets, from "
                    f"{w['negative_share_prev_9m']:.0%} to {w['negative_share_last_3m']:.0%}. "
                    f"{ins['urgent_customers']} high churn-risk customers with recent negative tickets are on the urgent call "
                    f"list, representing {lakh(ins['urgent_revenue_at_risk_inr'])} of revenue at risk."))
    sec.append(("6. Actions for Q4",
                "Retention team to call every customer on the urgent call list within 1 business day. Operations to review "
                "the top complaint themes with courier partners before Diwali. Sales to focus account management on the "
                "Big-ticket buyers segment. Support to clear the needs_review queue daily."))
    return "Q3 2026 Business Review", f"BI-Q3-2026, prepared {date.today():%d %B %Y}", sec


def write_pdf(fname, title, meta, sections):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(DOCS / f"{fname}.pdf"), pagesize=A4, title=title, author="Kaveri Retail (fictional)",
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    story = [Paragraph(title, styles["Title"]), Paragraph(COMPANY, styles["Italic"]),
             Paragraph(meta, styles["Italic"]), Spacer(1, 14)]
    for head, body in sections:
        story += [Paragraph(head, styles["Heading2"]), Paragraph(body.replace("&", "&amp;"), styles["BodyText"]),
                  Spacer(1, 6)]
    doc.build(story)


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    for fname, (title, meta, sections) in POLICIES.items():
        write_pdf(fname, title, meta, sections)
    write_pdf("Q3_2026_Business_Review", *business_review())
    print(f"[Step 7a] wrote {len(POLICIES) + 1} PDFs to {DOCS}")


if __name__ == "__main__":
    main()
