"""
Step 5b: Anomaly detection on orders with an Autoencoder (TensorFlow / Keras).

Business question: "Which orders look wrong?" - pricing errors, fat-finger quantities,
unusual discounts, or a customer suddenly buying in a way they never do (fraud / data issue).

Idea of an autoencoder:
    order features (15 numbers) -> squeeze to 4 numbers -> rebuild the 15 numbers
    It is trained only on normal orders, so it learns what "normal" looks like.
    A weird order can't be rebuilt well -> high reconstruction error = anomaly.

How we prove it works (our data has no "anomaly" labels, which is normal in real life):
    * Train on orders before 2026-07-01.
    * Take orders from 2026-07-01 onward and INJECT ~1% known-bad orders
      (x10 price, x20 quantity, 60-90% discount, consumer bulk-buying furniture...).
    * Check how many injected orders land in the flagged list.
    * Compare with Isolation Forest (classic ML baseline) on exactly the same data.

Outputs:
    models/anomaly_autoencoder.keras + models/anomaly_scaler.joblib
    data/processed/order_anomalies.csv     (top suspicious real orders + reason)
    reports/anomaly_metrics.json
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.config import SALES_CSV, CUSTOMERS_CSV, MODELS, DATA_PROCESSED, REPORTS, RANDOM_STATE

SPLIT_DATE = "2026-07-01"
INJECT_RATE = 0.01
FLAG_RATE = 0.01        # flag the top 1% most unusual orders for review
CATS = ["Electronics", "Furniture", "Office Supplies", "Clothing"]
SEGS = ["Consumer", "SMB", "Corporate"]


# ---------------------------------------------------------------- features
def load_orders():
    s = pd.read_csv(SALES_CSV, parse_dates=["order_date"])
    c = pd.read_csv(CUSTOMERS_CSV)[["customer_id", "segment"]]
    return s.merge(c, on="customer_id", how="left")


def reference_stats(train: pd.DataFrame) -> dict:
    """Typical values learned from TRAINING orders only (re-used for test / scoring)."""
    return {
        "cat_price_median": train.groupby("category")["unit_price"].median().to_dict(),
        "seg_qty_median": train.groupby("segment")["quantity"].median().to_dict(),
        "cust_qty_median": train.groupby("customer_id")["quantity"].median().to_dict(),
    }


def featurize(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["log_revenue"] = np.log1p(df["revenue"])
    f["log_quantity"] = np.log1p(df["quantity"])
    f["log_unit_price"] = np.log1p(df["unit_price"])
    f["discount"] = df["discount"]
    # price vs the usual price of that category
    f["price_vs_category"] = np.log(df["unit_price"] / df["category"].map(ref["cat_price_median"]))
    # quantity vs what this kind of customer usually buys / what THIS customer usually buys
    f["qty_vs_segment"] = np.log(df["quantity"] / df["segment"].map(ref["seg_qty_median"]))
    own = df["customer_id"].map(ref["cust_qty_median"]).fillna(df["segment"].map(ref["seg_qty_median"]))
    f["qty_vs_own_history"] = np.log(df["quantity"] / own)
    for c in CATS:
        f[f"cat_{c}"] = (df["category"] == c).astype(float)
    for s in SEGS:
        f[f"seg_{s}"] = (df["segment"] == s).astype(float)
    return f


def inject_anomalies(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Copy some real orders and corrupt them in realistic ways. Label = 1."""
    n = int(len(df) * INJECT_RATE)
    bad = df.sample(n, random_state=RANDOM_STATE).copy()
    kind = rng.choice(["price_x10", "qty_x20", "huge_discount", "consumer_bulk_furniture"], n)
    bad["anomaly_type"] = kind
    m = kind == "price_x10"
    bad.loc[m, "unit_price"] *= rng.choice([10, 0.1], m.sum())
    m = kind == "qty_x20"
    bad.loc[m, "quantity"] = (bad.loc[m, "quantity"] * 20).astype(int)
    m = kind == "huge_discount"
    bad.loc[m, "discount"] = rng.uniform(0.6, 0.9, m.sum()).round(2)
    m = kind == "consumer_bulk_furniture"
    bad.loc[m, ["segment", "category", "quantity"]] = ["Consumer", "Furniture", 40]
    bad.loc[m, "unit_price"] = 8000.0
    bad["revenue"] = (bad["quantity"] * bad["unit_price"] * (1 - bad["discount"])).round(2)
    bad["is_injected"] = 1
    good = df.copy()
    good["is_injected"], good["anomaly_type"] = 0, ""
    return pd.concat([good, bad], ignore_index=True)


# ---------------------------------------------------------------- model
def build_autoencoder(n_features: int) -> keras.Model:
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(16, activation="relu")(inp)
    x = layers.Dense(8, activation="relu")(x)
    code = layers.Dense(4, activation="relu", name="bottleneck")(x)   # squeeze
    x = layers.Dense(8, activation="relu")(code)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(n_features)(x)                                  # rebuild
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def recon_error(model, X):
    rec = model.predict(X, verbose=0, batch_size=1024)
    return ((X - rec) ** 2), ((X - rec) ** 2).mean(axis=1)


def evaluate(y, score):
    k = int(len(score) * FLAG_RATE)
    top = np.argsort(-score)[:k]
    return {"roc_auc": round(roc_auc_score(y, score), 3),
            "pr_auc": round(average_precision_score(y, score), 3),
            "precision_in_flagged": round(float(y[top].mean()), 3),
            "recall_in_flagged": round(float(y[top].sum() / y.sum()), 3)}


def main():
    print("[Step 5b] Order anomaly detection (autoencoder vs Isolation Forest)")
    rng = np.random.default_rng(RANDOM_STATE)
    orders = load_orders()
    train = orders[orders["order_date"] < SPLIT_DATE]
    test = inject_anomalies(orders[orders["order_date"] >= SPLIT_DATE], rng)
    ref = reference_stats(train)
    feat_names = list(featurize(train.head(), ref).columns)

    scaler = StandardScaler().fit(featurize(train, ref))
    Xtr = scaler.transform(featurize(train, ref)).astype("float32")
    Xte = scaler.transform(featurize(test, ref)).astype("float32")
    y = test["is_injected"].values
    print(f"  train orders {len(Xtr):,} | test orders {len(Xte):,} "
          f"(incl. {y.sum()} injected anomalies)")

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    ae = build_autoencoder(Xtr.shape[1])
    ae.fit(Xtr, Xtr, epochs=30, batch_size=256, validation_split=0.1, verbose=0,
           callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    per_feat, ae_score = recon_error(ae, Xte)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE).fit(Xtr)
    iso_score = -iso.score_samples(Xte)          # higher = more unusual

    results = {"autoencoder": evaluate(y, ae_score), "isolation_forest": evaluate(y, iso_score)}
    for name, m in results.items():
        print(f"    {name:17s} ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | "
              f"top-1% flagged: {m['precision_in_flagged']:.0%} are real anomalies, "
              f"catches {m['recall_in_flagged']:.0%} of them")

    # which kinds of anomalies does each method catch?
    k = int(len(ae_score) * FLAG_RATE)
    test["ae_flag"] = np.isin(np.arange(len(test)), np.argsort(-ae_score)[:k])
    test["iso_flag"] = np.isin(np.arange(len(test)), np.argsort(-iso_score)[:k])
    by_type = (test[test["is_injected"] == 1].groupby("anomaly_type")[["ae_flag", "iso_flag"]]
               .mean().round(2))
    print("  Caught by type:\n" + by_type.to_string())

    # final: score every real order with the trained model and explain the top ones
    Xall = scaler.transform(featurize(orders, ref)).astype("float32")
    per_feat_all, score_all = recon_error(ae, Xall)
    thr = float(np.quantile(score_all, 1 - FLAG_RATE))
    worst_feat = np.array(feat_names)[per_feat_all.argmax(axis=1)]
    flagged = orders.assign(anomaly_score=score_all.round(3), main_reason=worst_feat)
    flagged = flagged[flagged["anomaly_score"] >= thr].sort_values("anomaly_score", ascending=False)
    flagged.to_csv(DATA_PROCESSED / "order_anomalies.csv", index=False)

    ae.save(MODELS / "anomaly_autoencoder.keras")
    joblib.dump({"scaler": scaler, "ref": ref, "features": feat_names, "threshold": thr},
                MODELS / "anomaly_scaler.joblib")
    summary = {"test_setup": f"orders from {SPLIT_DATE} + {INJECT_RATE:.0%} injected anomalies",
               "results": results, "caught_by_type": by_type.to_dict(),
               "flagged_real_orders": int(len(flagged)),
               "flagged_reasons": flagged["main_reason"].value_counts().head(5).to_dict()}
    (REPORTS / "anomaly_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"  {len(flagged)} real orders flagged for review -> data/processed/order_anomalies.csv")
    return summary


if __name__ == "__main__":
    main()
