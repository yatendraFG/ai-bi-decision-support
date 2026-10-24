"""
Step 6b: Support-ticket NLP - sentiment (negative / neutral / positive) and
category (Delivery / Billing / Product Quality / Technical / Refund).

Business use: route every new ticket to the right team automatically, push angry
tickets to the top of the queue, and send only the unclear ones to a human.

Approaches, compared on the same time-based split:
  1. keyword rules          - human-written word lists (the baseline every model must beat)
  2. tfidf_lr               - word 1-2 grams + character 2-5 grams -> LogisticRegression.
                              Character n-grams handle typos ("refnd") and Hinglish spellings.
  3. tfidf_lr_priority      - same + the ticket's priority (chosen by the customer when they
                              raise it). Text alone can't tell an emotionless "refund not
                              received" apart as neutral or negative; priority helps.
  4. keras_bigru            - TextVectorization -> Embedding -> Bidirectional GRU with two
                              outputs (sentiment + category). Uses TensorFlow from Step 5.

Split (like a real deployment, no leakage):
  train : tickets before 2025-07-01
  valid : 2025-07-01 .. 2025-12-31  -> pick best approach + confidence threshold
  test  : 2026-01-01 onwards        -> reported once
Metric: macro-F1 (every class counts equally; accuracy alone hides a weak "positive" class).

Human-in-the-loop: if the model's top probability is below a threshold (chosen on
validation so that auto-handled tickets are >= 95% correct), the ticket is flagged
needs_review instead of being auto-routed.

Outputs:
  models/ticket_sentiment.joblib, models/ticket_category.joblib
  data/processed/ticket_predictions.csv
  reports/nlp_metrics.json
"""
import json
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import TICKETS_CSV, MODELS, DATA_PROCESSED, REPORTS, RANDOM_STATE

VALID_START, TEST_START = "2025-07-01", "2026-01-01"
TASKS = {"sentiment": ["negative", "neutral", "positive"],
         "category": ["Billing", "Delivery", "Product Quality", "Refund", "Technical"]}
TARGET_AUTO_ACCURACY = 0.95
HINGLISH = re.compile(r"\b(?:nahi|hai|ho|kar|karu|mera|mere|kya|abhi|bahut|paise|gaya|gaye|aaya|batao|kitne|din)\b",
                      re.I)


# ---------------------------------------------------------------- 1. keyword baseline
NEG_WORDS = ["frustrat", "disappoint", "worst", "unacceptable", "fed up", "useless", "never",
             "cancel my", "complain", "bura", "bakwas", "barbaad", "not happy", "nobody", "!!!"]
POS_WORDS = ["thank you, aap", "love", "great", "awesome", "happy", "appreciate", "accha", "best", "mast"]
CAT_WORDS = {
    "Refund": ["refund", "return"],
    "Billing": ["charge", "invoice", "gst", "bill", "debit", "discount", "coupon", "emi", "payment", "paise"],
    "Technical": ["app", "login", "log in", "otp", "password", "website", "error", "crash", "download",
                  "notification", "search"],
    "Delivery": ["deliver", "courier", "tracking", "shipment", "package", "parcel", "address", "missing",
                 "wrong item"],
    "Product Quality": ["quality", "warranty", "stopped working", "described", "material", "scratch", "noise",
                        "colour", "battery", "kharab"],
}


def keyword_sentiment(text: str) -> str:
    t = text.lower()
    neg = sum(w in t for w in NEG_WORDS)
    pos = sum(w in t for w in POS_WORDS)
    return "negative" if neg > pos else "positive" if pos > neg else "neutral"


def keyword_category(text: str) -> str:
    t = text.lower()
    scores = {c: sum(w in t for w in ws) for c, ws in CAT_WORDS.items()}
    return max(scores, key=scores.get) if max(scores.values()) else "Delivery"  # most common class


# ---------------------------------------------------------------- 2/3. TF-IDF + LR (+ priority)
def tfidf_model(use_priority: bool = False) -> Pipeline:
    """Takes a DataFrame with columns `text` (and `priority`)."""
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text"),
             ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), "text")]
    if use_priority:
        parts.append(("priority", OneHotEncoder(handle_unknown="ignore"), ["priority"]))
    return Pipeline([("features", ColumnTransformer(parts)),
                     ("clf", LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))])


# ---------------------------------------------------------------- 4. Keras BiGRU (multi-task)
def _as_tensor(text):
    import tensorflow as tf
    return tf.constant(np.array(list(text), dtype=str)[:, None])


def train_keras(train_text, train_y: dict, valid_text=None, valid_y=None, epochs=15):
    """valid_text=None -> train for exactly `epochs` (used for the final refit)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    vec = layers.TextVectorization(max_tokens=8000, output_sequence_length=40,
                                   standardize="lower_and_strip_punctuation")
    vec.adapt(np.array(list(train_text), dtype=str))
    inp = keras.Input(shape=(1,), dtype="string", name="text")
    x = layers.Embedding(8000, 64, mask_zero=True)(vec(inp))
    x = layers.Bidirectional(layers.GRU(32))(x)
    x = layers.Dropout(0.3)(x)
    outs = [layers.Dense(len(TASKS[t]), activation="softmax", name=t)(x) for t in TASKS]
    model = keras.Model(inp, outs)
    model.compile(optimizer="adam", loss={t: "sparse_categorical_crossentropy" for t in TASKS},
                  metrics={t: ["accuracy"] for t in TASKS})

    enc = lambda y: {t: np.array([TASKS[t].index(v) for v in y[t]]) for t in TASKS}
    kw = {}
    if valid_text is not None:
        kw = {"validation_data": (_as_tensor(valid_text), enc(valid_y)),
              "callbacks": [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]}
    hist = model.fit(_as_tensor(train_text), enc(train_y), epochs=epochs, batch_size=64, verbose=0, **kw)
    return model, hist


def keras_proba(model, text) -> dict:
    return dict(zip(TASKS, model.predict(_as_tensor(text), verbose=0)))


# ---------------------------------------------------------------- helpers
def scores(y_true, y_pred, labels):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro"), 3)}


def pick_threshold(y_true, y_pred, conf, target=TARGET_AUTO_ACCURACY):
    """Lowest confidence cut-off where tickets above it are >= target accurate."""
    ok = (np.asarray(y_true) == np.asarray(y_pred))
    for thr in np.arange(0.30, 0.99, 0.01):
        m = conf >= thr
        if m.sum() and ok[m].mean() >= target:
            return round(float(thr), 2)
    return 0.99


def auto_vs_review(y_true, y_pred, conf, thr):
    ok = (np.asarray(y_true) == np.asarray(y_pred))
    m = conf >= thr
    return {"threshold": thr, "auto_handled_share": round(float(m.mean()), 3),
            "accuracy_auto_handled": round(float(ok[m].mean()), 3) if m.any() else None,
            "accuracy_sent_to_review": round(float(ok[~m].mean()), 3) if (~m).any() else None}


def main(with_keras: bool = True):
    print("[Step 6] Ticket NLP: sentiment + category")
    t = pd.read_csv(TICKETS_CSV, parse_dates=["created_at"])
    t["is_hinglish"] = t["text"].str.contains(HINGLISH)
    train = t[t["created_at"] < VALID_START]
    valid = t[(t["created_at"] >= VALID_START) & (t["created_at"] < TEST_START)]
    test = t[t["created_at"] >= TEST_START].copy()
    print(f"  train {len(train):,} | valid {len(valid):,} | test {len(test):,} tickets "
          f"({t['is_hinglish'].mean():.0%} Hinglish)")

    # ---- validation: compare approaches (store predictions + confidence)
    pv, conf_v = {}, {}
    pv["keyword"] = {"sentiment": valid["text"].map(keyword_sentiment).values,
                     "category": valid["text"].map(keyword_category).values}
    for name, prio in [("tfidf_lr", False), ("tfidf_lr_priority", True)]:
        pv[name], conf_v[name] = {}, {}
        for task in TASKS:
            m = tfidf_model(prio).fit(train, train[task])
            proba = m.predict_proba(valid)
            pv[name][task] = m.classes_[proba.argmax(1)]
            conf_v[name][task] = proba.max(1)
    use_keras = with_keras
    if use_keras:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            print("  TensorFlow not installed -> skipping the Keras model")
            use_keras = False
    if use_keras:
        print("  training Keras BiGRU ...")
        km, khist = train_keras(train["text"], {k: train[k] for k in TASKS},
                                valid["text"], {k: valid[k] for k in TASKS})
        kprob = keras_proba(km, valid["text"])
        pv["keras_bigru"] = {k: np.array(TASKS[k])[p.argmax(1)] for k, p in kprob.items()}
        conf_v["keras_bigru"] = {k: p.max(1) for k, p in kprob.items()}

    comparison = {task: {n: scores(valid[task], p[task], TASKS[task]) for n, p in pv.items()} for task in TASKS}
    for task, res in comparison.items():
        print(f"  VALID {task:9s} " + " | ".join(f"{n} {r['macro_f1']:.3f}" for n, r in res.items()))
    # pick the best model, but prefer a simpler one if it is within 0.005 F1 (differences that small are noise)
    simplicity = [n for n in ["tfidf_lr", "tfidf_lr_priority", "keras_bigru"] if n in pv]
    best = {}
    for task, res in comparison.items():
        top = max(res[n]["macro_f1"] for n in simplicity)
        best[task] = next(n for n in simplicity if res[n]["macro_f1"] >= top - 0.005)
    thresholds = {task: pick_threshold(valid[task], pv[best[task]][task], conf_v[best[task]][task])
                  for task in TASKS}
    print(f"  selected: {best} | review thresholds: {thresholds}")

    # ---- refit winners on train+valid, report once on test
    trval = pd.concat([train, valid])
    test_report = {}
    if "keras_bigru" in best.values():
        best_epochs = int(np.argmin(khist.history["val_loss"])) + 1
        km, _ = train_keras(trval["text"], {k: trval[k] for k in TASKS}, epochs=best_epochs)
        ktest = keras_proba(km, test["text"])
        km.save(MODELS / "ticket_nlp_bigru.keras")
    for task in TASKS:
        if best[task] == "keras_bigru":
            proba = ktest[task]
            test[f"pred_{task}"] = np.array(TASKS[task])[proba.argmax(1)]
        else:
            m = tfidf_model(best[task] == "tfidf_lr_priority").fit(trval, trval[task])
            proba = m.predict_proba(test)
            test[f"pred_{task}"] = m.classes_[proba.argmax(1)]
            joblib.dump({"model": m, "uses_priority": best[task] == "tfidf_lr_priority",
                         "review_threshold": thresholds[task]}, MODELS / f"ticket_{task}.joblib")
        test[f"{task}_confidence"] = proba.max(1).round(3)
        labels = TASKS[task]
        y, yhat = test[task], test[f"pred_{task}"]
        rep = classification_report(y, yhat, labels=labels, output_dict=True, zero_division=0)
        h = test["is_hinglish"]
        test_report[task] = {
            "model": best[task], **scores(y, yhat, labels),
            "f1_by_class": {c: round(rep[c]["f1-score"], 3) for c in labels},
            "macro_f1_english": round(f1_score(y[~h], yhat[~h], average="macro"), 3),
            "macro_f1_hinglish": round(f1_score(y[h], yhat[h], average="macro"), 3),
            "human_in_the_loop": auto_vs_review(y, yhat, test[f"{task}_confidence"].values, thresholds[task]),
            "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y, yhat, labels=labels).tolist()},
        }
        r, hl = test_report[task], test_report[task]["human_in_the_loop"]
        print(f"  TEST  {task:9s} {r['model']}: macro-F1 {r['macro_f1']:.3f}, accuracy {r['accuracy']:.1%} "
              f"| English {r['macro_f1_english']:.3f} / Hinglish {r['macro_f1_hinglish']:.3f}")
        print(f"        auto-handle {hl['auto_handled_share']:.0%} of tickets at {hl['accuracy_auto_handled']:.1%} "
              f"accuracy, rest ({1 - hl['auto_handled_share']:.0%}) to human review")

    # if a Keras model won, still keep fast sklearn models for the agent / API (no TensorFlow needed)
    for task in TASKS:
        if best[task] == "keras_bigru":
            m = tfidf_model(True).fit(trval, trval[task])
            joblib.dump({"model": m, "uses_priority": True, "review_threshold": thresholds[task]},
                        MODELS / f"ticket_{task}.joblib")

    test["needs_review"] = ((test["sentiment_confidence"] < thresholds["sentiment"])
                            | (test["category_confidence"] < thresholds["category"]))
    wrong = test[test["sentiment"] != test["pred_sentiment"]]
    examples = wrong.sample(min(8, len(wrong)), random_state=RANDOM_STATE)[
        ["text", "priority", "sentiment", "pred_sentiment", "sentiment_confidence"]]
    test[["ticket_id", "created_at", "customer_id", "text", "priority", "category", "pred_category",
          "category_confidence", "sentiment", "pred_sentiment", "sentiment_confidence", "needs_review",
          "is_hinglish"]].to_csv(DATA_PROCESSED / "ticket_predictions.csv", index=False)

    out = {"split": {"train_until": VALID_START, "valid_until": TEST_START, "test_from": TEST_START},
           "validation_comparison": comparison, "selected": best, "review_thresholds": thresholds,
           "test": test_report, "needs_review_share": round(float(test["needs_review"].mean()), 3),
           "sentiment_error_examples": examples.to_dict(orient="records")}
    (REPORTS / "nlp_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
