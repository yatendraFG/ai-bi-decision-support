"""
Step 5a: LSTM sales forecasting (TensorFlow / Keras).

Question we want to answer honestly: does a neural network beat the Step 3
gradient-boosting model on the SAME test windows?

Model (one model for all 4 regions):
    input 1  - last 91 days of revenue, divided by their own mean ("level")
               + day-of-year sin/cos for each day          -> LSTM(48)
    input 2  - same 30 days last year (scaled by last year's level)  -> seasonality hint
               + region one-hot
    output   - next 30 days as multiples of the level  (x level = revenue in INR)

Why scale by level?  Revenue grows over time; a network trained on 2024 numbers would
never have "seen" 2026 values. Dividing by the recent level makes every window look
alike, so the network learns shape (weekly cycle, festive bump), not absolute size.

Evaluation: identical 12 rolling 30-day folds as Step 3 + an ensemble (average of
gradient boosting and LSTM), all scored with the same function.

Outputs:
    models/lstm_forecaster.keras
    reports/lstm_forecast_metrics.json
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow info logs

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import MODELS, REPORTS, FORECAST_HORIZON_DAYS as H, RANDOM_STATE
from src.ml.forecasting import (load_daily, backtest_predictions, score_backtest,
                                fold_cutoffs)

LOOKBACK = 91          # days of history the LSTM reads
LY_OFFSET = 364        # "same weekday last year"
EPOCHS = 40
BATCH = 64
ORIGIN_STRIDE = 2   # use every 2nd day as a training window (halves training time)
LOSS = os.environ.get("LSTM_LOSS", "mse")  # mse -> predicts the mean (MAE predicts the median = under-forecast)
FOLD_CACHE = REPORTS / f"lstm_folds_{LOSS}"  # finished folds are saved -> a stopped run resumes


# ---------------------------------------------------------------- data -> windows
def make_windows(daily: pd.DataFrame, origins):
    """For each region and each forecast origin t (first day to predict) build:
       X_seq [LOOKBACK, 3], X_aux [H + n_regions], y [H], level."""
    regions = list(daily.columns)
    doy = daily.index.dayofyear.values
    sin, cos = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    pos = {d: i for i, d in enumerate(daily.index)}
    Xs, Xa, Y, L, meta = [], [], [], [], []
    for r_i, r in enumerate(regions):
        v = daily[r].values
        for t in origins:
            i = pos[t]
            if i - LOOKBACK - LY_OFFSET < 0:
                continue
            hist = v[i - LOOKBACK:i]
            level = hist.mean()
            if level <= 0:
                continue
            seq = np.stack([hist / level, sin[i - LOOKBACK:i], cos[i - LOOKBACK:i]], axis=1)
            ly_level = v[i - LOOKBACK - LY_OFFSET:i - LY_OFFSET].mean()
            ly_next = v[i - LY_OFFSET:i - LY_OFFSET + H] / max(ly_level, 1e-9)
            onehot = np.eye(len(regions))[r_i]
            y = v[i:i + H] / level if i + H <= len(v) else np.full(H, np.nan)
            Xs.append(seq); Xa.append(np.concatenate([ly_next, onehot]))
            Y.append(y); L.append(level); meta.append((r, t))
    return (np.array(Xs, dtype="float32"), np.array(Xa, dtype="float32"),
            np.array(Y, dtype="float32"), np.array(L), meta)


# ---------------------------------------------------------------- model
def build_model(n_aux: int) -> keras.Model:
    seq_in = keras.Input(shape=(LOOKBACK, 3), name="history")
    aux_in = keras.Input(shape=(n_aux,), name="last_year_and_region")
    x = layers.LSTM(48)(seq_in)
    x = layers.Concatenate()([x, aux_in])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(H, activation="softplus", name="next_30_days")(x)  # always >= 0
    model = keras.Model([seq_in, aux_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=LOSS)
    return model


def train(daily: pd.DataFrame, cutoff, verbose=0):
    """Train on every origin whose 30 target days end before `cutoff` (no leakage)."""
    last_origin = cutoff - pd.Timedelta(days=H)
    origins = [d for d in daily.index if d <= last_origin][::-1][::ORIGIN_STRIDE]
    Xs, Xa, Y, _, _ = make_windows(daily, origins)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = build_model(Xa.shape[1])
    stop = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    hist = model.fit([Xs, Xa], Y, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH,
                     callbacks=[stop], verbose=verbose)
    return model, hist


def predict_from(model, daily: pd.DataFrame, origin) -> pd.DataFrame:
    Xs, Xa, _, L, meta = make_windows(daily, [origin])
    ratio = model.predict([Xs, Xa], verbose=0)
    rows = []
    for (r, t), lv, rr in zip(meta, L, ratio):
        dates = pd.date_range(t, periods=H)
        rows += [{"date": d, "region": r, "pred_lstm": float(p * lv)} for d, p in zip(dates, rr)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main(n_folds: int = 12):
    print("[Step 5a] LSTM forecasting (Keras)")
    daily = load_daily()
    print("  running Step 3 gradient boosting + baselines on the same folds ...")
    res = backtest_predictions(daily, n_folds)

    FOLD_CACHE.mkdir(exist_ok=True)
    lstm_preds = []
    for cutoff in fold_cutoffs(daily, n_folds):
        cache = FOLD_CACHE / f"{cutoff.date()}.csv"
        if cache.exists():
            lstm_preds.append(pd.read_csv(cache, parse_dates=["date"]))
            print(f"    fold {cutoff.date()} loaded from cache")
            continue
        model, hist = train(daily, cutoff)      # train() only uses windows ending before cutoff
        p = predict_from(model, daily, cutoff)  # inputs = the 91 days before cutoff
        p.to_csv(cache, index=False)
        lstm_preds.append(p)
        print(f"    fold {cutoff.date()} trained ({len(hist.history['loss'])} epochs)")
    lstm = pd.concat(lstm_preds)
    res = res.merge(lstm, on=["date", "region"], how="left")
    res["pred_ensemble"] = (res["pred_model"] + res["pred_lstm"]) / 2

    cols = ["pred_model", "pred_lstm", "pred_ensemble", "pred_naive_last30", "pred_seasonal_naive"]
    scored = score_backtest(res, cols, n_folds)
    for part in ("metrics", "monthly_error_pct_by_fold"):
        scored[part]["gradient_boosting"] = scored[part].pop("model")
    print("  Same 12 folds, lower is better:")
    for name, m in scored["metrics"].items():
        print(f"    {name:18s} monthly WAPE {m['monthly_total_wape']:.1%} | weekly total "
              f"{m['weekly_total_wape']:.1%} | weekly by region {m['weekly_region_wape']:.1%}"
              f" | daily {m['daily_wape']:.1%}")

    # final model on all data, saved for the dashboard / agent
    final, _ = train(daily, daily.index[-1] + pd.Timedelta(days=1))
    final.save(MODELS / "lstm_forecaster.keras")
    (REPORTS / "lstm_forecast_metrics.json").write_text(json.dumps(scored, indent=2))
    return scored


if __name__ == "__main__":
    main()
