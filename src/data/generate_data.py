"""
Step 2: Realistic synthetic business data generator.

Creates three linked tables (same schema you can later replace with real company data):

customers.csv
    customer_id, name, region, segment, plan, signup_date, churn_date (empty if active)
sales.csv
    order_id, order_date, customer_id, region, category, quantity, unit_price, discount, revenue
tickets.csv
    ticket_id, created_at, customer_id, category, priority, text, sentiment

Built-in patterns (so the models have real signal to find):
  * Sales: upward trend, weekly cycle (weekend dip), festive peak (Oct-Nov, Diwali season),
    year-end bump, region-specific levels.
  * Churn: churning customers order less and less in the ~4 months before they leave,
    raise more tickets and the tickets turn negative.
  * Segments: Corporate customers buy big & rarely, Consumers small & often, SMB in between.
"""
import numpy as np
import pandas as pd

from src.config import CUSTOMERS_CSV, SALES_CSV, TICKETS_CSV, RANDOM_STATE
from src.data.ticket_text import make_ticket_texts

START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2026-09-30")
N_CUSTOMERS = 3000

REGIONS = ["North", "South", "East", "West"]
REGION_W = [0.32, 0.28, 0.18, 0.22]
SEGMENTS = ["Consumer", "SMB", "Corporate"]
SEGMENT_W = [0.6, 0.28, 0.12]
PLANS = ["Basic", "Standard", "Premium"]

CATEGORIES = {
    # category: (mean unit price INR, price spread)
    "Electronics": (12000, 0.5),
    "Furniture": (8000, 0.4),
    "Office Supplies": (600, 0.6),
    "Clothing": (1500, 0.5),
}
SEGMENT_PROFILE = {
    # segment: (orders per year, quantity mean, category preference)
    "Consumer": (6, 1.6, [0.30, 0.10, 0.20, 0.40]),
    "SMB": (10, 4.0, [0.30, 0.25, 0.35, 0.10]),
    "Corporate": (8, 12.0, [0.35, 0.30, 0.30, 0.05]),
}

TICKET_TEMPLATES = {
    "Delivery": {
        "neg": ["My order is delayed by {d} days and nobody is responding.",
                "Package arrived damaged, very disappointed with the delivery.",
                "Still waiting for my delivery, this is the {n} time I'm complaining."],
        "neu": ["Can you share the tracking number for my order?",
                "I want to change the delivery address for my order."],
        "pos": ["Delivery was quick, thanks! Just need the invoice copy."],
    },
    "Billing": {
        "neg": ["I was charged twice for the same order, please fix this immediately.",
                "Wrong amount billed on my invoice, this keeps happening."],
        "neu": ["Please send GST invoice for my last purchase.",
                "How can I update my billing details?"],
        "pos": ["Thanks for the quick refund on the billing issue."],
    },
    "Product Quality": {
        "neg": ["The product stopped working within a week, terrible quality.",
                "Item is not as described, very poor quality. I want to return it."],
        "neu": ["Is there a warranty on this product?",
                "What is the material used in this item?"],
        "pos": ["Great product, just want to know about the extended warranty."],
    },
    "Technical": {
        "neg": ["Your app keeps crashing when I try to checkout, so frustrating.",
                "Unable to login for 3 days, worst experience."],
        "neu": ["How do I reset my password?",
                "Where can I download my order history?"],
        "pos": ["The new app update is nice, small question about notifications."],
    },
    "Refund": {
        "neg": ["Refund not received even after {d} days, I am cancelling my account.",
                "You promised a refund weeks ago, this is unacceptable."],
        "neu": ["What is the refund timeline for returned items?"],
        "pos": ["Refund received, thank you for the support."],
    },
}


def _seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily demand multiplier: trend x yearly x weekly."""
    t = (dates - START).days.values / 365.25
    trend = 1 + 0.12 * t                                   # ~12% growth / year
    doy = dates.dayofyear.values
    festive = 0.45 * np.exp(-0.5 * ((doy - 305) / 18) ** 2)  # peak ~ 1 Nov
    year_end = 0.15 * np.exp(-0.5 * ((doy - 355) / 8) ** 2)
    summer_dip = -0.10 * np.exp(-0.5 * ((doy - 165) / 25) ** 2)
    weekly = np.where(dates.dayofweek.values >= 5, 0.8, 1.05)
    return trend * (1 + festive + year_end + summer_dip) * weekly


def make_customers(rng: np.random.Generator) -> pd.DataFrame:
    n = N_CUSTOMERS
    # 40% are existing customers (joined 2021-2022), the rest join steadily until the end
    existing = rng.random(n) < 0.4
    old_start = pd.Timestamp("2021-01-01")
    old = old_start + pd.to_timedelta(rng.integers(0, (START - old_start).days, n), unit="D")
    new = START + pd.to_timedelta(rng.integers(0, (END - START).days - 60, n), unit="D")
    signup = pd.DatetimeIndex(np.where(existing, old, new))
    seg = rng.choice(SEGMENTS, n, p=SEGMENT_W)
    plan = rng.choice(PLANS, n, p=[0.5, 0.35, 0.15])

    # latent churn risk -> who churns and when
    risk = (-1.4
            + np.where(plan == "Basic", 0.6, np.where(plan == "Premium", -0.7, 0))
            + np.where(seg == "Consumer", 0.4, np.where(seg == "Corporate", -0.6, 0))
            + rng.normal(0, 0.9, n))
    churns = rng.random(n) < 1 / (1 + np.exp(-risk))
    # churn happens at least 120 days after signup (and inside the data window), before END
    earliest = signup + pd.Timedelta(days=120)
    floor = START + pd.Timedelta(days=60)
    earliest = earliest.where(earliest >= floor, floor)   # works on pandas 2.x and 3.x
    span = np.clip((END - earliest).days.values, 1, None)
    churn_date = earliest + pd.to_timedelta((rng.random(n) * span).astype(int), unit="D")
    churn_date = pd.Series(churn_date).where(churns & (earliest < END))

    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "name": [f"Customer {i}" for i in range(1, n + 1)],
        "region": rng.choice(REGIONS, n, p=REGION_W),
        "segment": seg,
        "plan": plan,
        "signup_date": signup.normalize(),
        "churn_date": churn_date.dt.normalize(),
        "_activity": rng.lognormal(0, 0.35, n),  # personal buying intensity (hidden)
    })


def make_sales(cust: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    season = _seasonality(dates)
    seg_rate = cust["segment"].map({s: p[0] for s, p in SEGMENT_PROFILE.items()}).values / 365
    base_rate = seg_rate * cust["_activity"].values

    signup = cust["signup_date"].values
    churn = cust["churn_date"].fillna(END + pd.Timedelta(days=999)).values
    # region-level demand differences
    region_mult = cust["region"].map({"North": 1.1, "South": 1.0, "East": 0.85, "West": 1.05}).values
    base_rate = base_rate * region_mult

    # 1) decide WHO orders on WHICH day (Poisson per customer-day)
    order_day, order_cust = [], []
    for d, s in zip(dates.values, season):
        active = (signup <= d) & (churn > d)
        # disengagement: order rate fades during the 120 days before churn
        days_to_churn = (churn - d).astype("timedelta64[D]").astype(int)
        fade = np.clip(days_to_churn / 120, 0.15, 1.0)
        n_orders = rng.poisson(base_rate * s * fade * active)
        idx = np.repeat(np.nonzero(n_orders)[0], n_orders[n_orders > 0])
        order_day.append(np.full(len(idx), d))
        order_cust.append(idx)
    order_day = np.concatenate(order_day)
    order_cust = np.concatenate(order_cust)
    n = len(order_cust)

    # 2) decide WHAT they bought (vectorised)
    cat_names = list(CATEGORIES)
    segs = cust["segment"].values[order_cust]
    cats = np.empty(n, dtype=object)
    qty = np.empty(n, dtype=int)
    for seg, (_, q_mean, cat_p) in SEGMENT_PROFILE.items():
        m = segs == seg
        cats[m] = rng.choice(cat_names, size=m.sum(), p=cat_p)
        qty[m] = np.maximum(1, rng.poisson(q_mean, m.sum()))
    price_mu = np.array([CATEGORIES[c][0] for c in cats])
    spread = np.array([CATEGORIES[c][1] for c in cats])
    price = np.round(price_mu * rng.lognormal(0, 1, n) ** spread, 2)
    disc = rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], n)

    sales = pd.DataFrame({
        "order_date": order_day,
        "customer_id": cust["customer_id"].values[order_cust],
        "region": cust["region"].values[order_cust],
        "category": cats, "quantity": qty, "unit_price": price, "discount": disc,
    })
    sales["revenue"] = (sales["quantity"] * sales["unit_price"] * (1 - sales["discount"])).round(2)
    sales.insert(0, "order_id", [f"O{i:07d}" for i in range(1, len(sales) + 1)])
    return sales


def make_tickets(cust: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for c in cust.itertuples():
        life_end = c.churn_date if pd.notna(c.churn_date) else END
        life_days = max((life_end - c.signup_date).days, 1)
        n = rng.poisson(life_days / 365 * 1.2)                  # normal: ~1 ticket / year
        dates = [c.signup_date + pd.Timedelta(days=int(rng.integers(0, life_days))) for _ in range(n)]
        if pd.notna(c.churn_date):                               # churners: extra tickets before leaving
            for _ in range(rng.poisson(2.2)):
                dates.append(c.churn_date - pd.Timedelta(days=int(rng.integers(5, 110))))
        for dt in dates:
            if dt < START or dt > END:
                continue
            near_churn = pd.notna(c.churn_date) and 0 <= (c.churn_date - dt).days <= 120
            p_neg = 0.75 if near_churn else 0.25
            sentiment = rng.choice(["negative", "neutral", "positive"],
                                   p=[p_neg, (1 - p_neg) * 0.7, (1 - p_neg) * 0.3])
            cat = rng.choice(list(TICKET_TEMPLATES), p=[0.3, 0.18, 0.22, 0.18, 0.12])
            key = {"negative": "neg", "neutral": "neu", "positive": "pos"}[sentiment]
            text = rng.choice(TICKET_TEMPLATES[cat][key]).format(
                d=int(rng.integers(3, 20)), n=rng.choice(["second", "third", "fourth"]))
            prio = ("High" if sentiment == "negative" and rng.random() < 0.6
                    else rng.choice(["Low", "Medium"], p=[0.5, 0.5]))
            rows.append((dt, c.customer_id, cat, prio, text, sentiment))

    t = pd.DataFrame(rows, columns=["created_at", "customer_id", "category", "priority", "text", "sentiment"])
    t = t.sort_values("created_at").reset_index(drop=True)
    t.insert(0, "ticket_id", [f"T{i:06d}" for i in range(1, len(t) + 1)])
    return t


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    print("[Step 2] Generating customers ...")
    cust = make_customers(rng)
    print("[Step 2] Generating sales orders (takes ~1 min) ...")
    sales = make_sales(cust, rng)
    print("[Step 2] Generating support tickets ...")
    tickets = make_tickets(cust, rng)
    # Step 6: realistic free-text (own random stream, so all other data stays identical)
    tickets["text"] = make_ticket_texts(tickets, np.random.default_rng(RANDOM_STATE + 6))

    cust.drop(columns="_activity").to_csv(CUSTOMERS_CSV, index=False)
    sales.to_csv(SALES_CSV, index=False)
    tickets.to_csv(TICKETS_CSV, index=False)
    print(f"  customers: {len(cust):,}  (churned: {cust['churn_date'].notna().sum():,})")
    print(f"  orders:    {len(sales):,}  revenue: INR {sales['revenue'].sum()/1e7:,.1f} Cr")
    print(f"  tickets:   {len(tickets):,}")


if __name__ == "__main__":
    main()
