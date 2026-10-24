"""
Step 3: Sales forecasting (daily revenue per region, next 30 days).

Approach  - "direct" multi-horizon forecasting with gradient boosting:
  * Every feature for day t uses only data up to t-30, so one model can predict
    all 30 future days at once with no recursive error build-up.
  * Features: lagged revenue (30, 35, 364 days), rolling means, last-year same-week
    level, calendar (day of week, day of year, month), region.
  * Target is log1p(revenue) to tame large corporate orders.

Evaluation - time-based holdout (last 90 days), compared against two baselines:
  * naive_last30: average of the last 30 known days
  * seasonal_naive: same day last year x growth factor
Metric: WAPE (weighted absolute % error) at daily and weekly level.

Outputs:
  models/sales_forecaster.joblib
  data/processed/sales_forecast.csv     (next 30 days, per region + total)
  reports/forecast_metrics.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import (SALES_CSV, MODELS, DATA_PROCESSED, REPORTS,
                        FORECAST_HORIZON_DAYS as H, RANDOM_STATE)

FEATURES = ["region_code", "dow", "doy", "month", "is_weekend",
            "lag_30", "lag_35", "lag_364", "roll7_lag30", "roll28_lag30",
            "roll91_lag30", "ly_week_mean", "ly_month_mean", "ly_ratio"]
LEVEL = "roll91_lag30"   # the model predicts y / LEVEL (a seasonal multiplier)
REGION_CODE = {"East": 0, "North": 1, "South": 2, "West": 3}


def wape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def load_daily(path=SALES_CSV) -> pd.DataFrame:
    s = pd.read_csv(path, parse_dates=["order_date"])
    daily = (s.groupby(["region", "order_date"])["revenue"].sum()
               .unstack("region").asfreq("D").fillna(0.0))
    return daily  # index=date, columns=regions


def build_features(daily: pd.DataFrame, extend_days: int = 0) -> pd.DataFrame:
    """Long table (date, region) with features. extend_days adds future rows (y = NaN)."""
    if extend_days:
        future = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=extend_days)
        daily = pd.concat([daily, pd.DataFrame(index=future, columns=daily.columns, dtype=float)])
    frames = []
    for region in daily.columns:
        y = daily[region]
        f = pd.DataFrame({"date": daily.index, "region": region, "y": y.values})
        f["lag_30"] = y.shift(30).values
        f["lag_35"] = y.shift(35).values
        f["lag_364"] = y.shift(364).values
        f["roll7_lag30"] = y.shift(30).rolling(7).mean().values
        f["roll28_lag30"] = y.shift(30).rolling(28).mean().values
        f["roll91_lag30"] = y.shift(30).rolling(91).mean().values
        f["ly_week_mean"] = y.shift(364 - 3).rolling(7).mean().values   # same week last year
        f["ly_month_mean"] = y.shift(364 - 15).rolling(30).mean().values
        # how much last year's same month stood above/below last year's base level
        f["ly_ratio"] = f["ly_month_mean"] / y.shift(364 + 30).rolling(91).mean().values
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    d = pd.to_datetime(df["date"])
    df["region_code"] = df["region"].map(REGION_CODE)
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["month"] = d.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def make_model():
    # Poisson loss keeps predictions positive and unbiased on average (a log-target
    # model predicts the median and under-forecasts revenue -> tried, rejected).
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.04, min_samples_leaf=50,
        l2_regularization=1.0, categorical_features=[0], random_state=RANDOM_STATE)


def fit(df: pd.DataFrame):
    """Target = revenue / recent 91-day level. Trees can't extrapolate a growing trend,
    but they can learn seasonal multipliers; the level carries the trend."""
    train = df.dropna(subset=FEATURES + ["y"])
    train = train[train[LEVEL] > 0]
    model = make_model()
    model.fit(train[FEATURES], train["y"] / train[LEVEL])
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return (model.predict(df[FEATURES]) * df[LEVEL].values).clip(min=0)


def fold_cutoffs(daily: pd.DataFrame, n_folds: int = 12):
    """Start dates of the n_folds consecutive 30-day test windows at the end of the data."""
    return [daily.index[-k * H] for k in range(n_folds, 0, -1)]


def backtest_predictions(daily: pd.DataFrame, n_folds: int = 12) -> pd.DataFrame:
    """Rolling-origin backtest: n_folds consecutive 30-day windows (covers a full year,
    including the festive season). Each fold trains only on data before its cutoff.
    Returns one row per (date, region) with actual y and every method's prediction."""
    df = build_features(daily)
    preds = []
    for cutoff in fold_cutoffs(daily, n_folds):
        end = cutoff + pd.Timedelta(days=H - 1)
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] <= end)].copy()
        test["pred_model"] = predict(fit(train), test)

        # baselines (only use info available at forecast time: before cutoff)
        hist = daily[daily.index < cutoff]
        test["pred_naive_last30"] = test["region"].map(hist.tail(30).mean())
        growth = hist.tail(365).sum() / hist.iloc[-730:-365].sum()
        test["pred_seasonal_naive"] = test["lag_364"] * test["region"].map(growth)
        test["fold"] = str(cutoff.date())
        preds.append(test)
    return pd.concat(preds)


def backtest(daily: pd.DataFrame, n_folds: int = 12) -> dict:
    res = backtest_predictions(daily, n_folds)
    return score_backtest(res, ["pred_model", "pred_naive_last30", "pred_seasonal_naive"], n_folds)


def score_backtest(res: pd.DataFrame, cols, n_folds: int = 12) -> dict:
    """Same metrics for every method -> fair comparison (also reused by Step 5 LSTM)."""
    metrics, per_fold = {}, {}
    for col in cols:
        name = col.replace("pred_", "")
        weekly = res.groupby([pd.Grouper(key="date", freq="W"), "region"])[["y", col]].sum()
        total_w = weekly.groupby(level=0).sum()
        month = res.groupby("fold")[["y", col]].sum()
        metrics[name] = {
            "daily_wape": round(wape(res["y"], res[col]), 3),
            "weekly_region_wape": round(wape(weekly["y"], weekly[col]), 3),
            "weekly_total_wape": round(wape(total_w["y"], total_w[col]), 3),
            "monthly_total_wape": round(wape(month["y"], month[col]), 3),
        }
        per_fold[name] = (100 * (month[col] / month["y"] - 1)).round(1).to_dict()
    return {"backtest": f"{n_folds} rolling 30-day windows, "
                        f"{res['date'].min().date()} -> {res['date'].max().date()}",
            "metrics": metrics, "monthly_error_pct_by_fold": per_fold}


def forecast_future(daily: pd.DataFrame, horizon: int = H):
    df = build_features(daily, extend_days=horizon)
    model = fit(df[df["date"] <= daily.index[-1]])
    fut = df[df["date"] > daily.index[-1]].copy()
    fut["forecast"] = predict(model, fut)
    out = fut.pivot(index="date", columns="region", values="forecast")
    out["Total"] = out.sum(axis=1)
    return model, out.round(2)


def main():
    print("[Step 3] Sales forecasting")
    daily = load_daily()
    bt = backtest(daily)
    print(f"  Backtest: {bt['backtest']}")
    for name, m in bt["metrics"].items():
        print(f"    {name:16s} monthly WAPE {m['monthly_total_wape']:.1%} | weekly total "
              f"{m['weekly_total_wape']:.1%} | weekly by region {m['weekly_region_wape']:.1%}"
              f" | daily {m['daily_wape']:.1%}")

    model, fc = forecast_future(daily)
    joblib.dump({"model": model, "features": FEATURES, "region_code": REGION_CODE},
                MODELS / "sales_forecaster.joblib")
    fc.to_csv(DATA_PROCESSED / "sales_forecast.csv")

    last30 = daily.tail(30).sum().sum()
    summary = {
        **bt,
        "forecast_window": f"{fc.index[0].date()} -> {fc.index[-1].date()}",
        "forecast_total_inr": round(float(fc["Total"].sum()), 0),
        "last_30_days_actual_inr": round(float(last30), 0),
        "forecast_vs_last30_pct": round(100 * (fc["Total"].sum() / last30 - 1), 1),
        "forecast_by_region_inr": fc.drop(columns="Total").sum().round(0).to_dict(),
    }
    (REPORTS / "forecast_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"  Next {H} days forecast: INR {summary['forecast_total_inr']/1e5:,.1f} lakh "
          f"({summary['forecast_vs_last30_pct']:+.1f}% vs last 30 days)")
    return summary


if __name__ == "__main__":
    main()
