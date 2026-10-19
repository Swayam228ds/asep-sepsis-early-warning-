"""M1: how many patients will be admitted tomorrow? Daily forecast from Hero DMC Heart Institute, Ludhiana.

Input: data/raw/hdhi/HDHI Admission data.csv and HDHI Pollution Data.csv (daily AQI, temperature, humidity)
Output: models/admissions_forecast.txt, models/admissions_profile.json, models/admissions_metrics.json

Features known the evening before: day of week, month, Indian public holidays, admissions over the
past days, and that day's weather and air quality. The last 4 months are the test set.
"""
import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import MODELS, save_json
from hdhi import load_admissions, load_weather

TEST_FROM = "2018-12-01"
FEATURES = ["dow", "month", "holiday", "lag1", "lag7", "lag14", "mean7", "mean28",
            "aqi", "max_temp", "min_temp", "humidity"]


def daily_frame():
    d = load_admissions()
    days = pd.date_range(d.admit_date.min(), d.admit_date.max())
    s = d.groupby("admit_date").size().reindex(days, fill_value=0)
    f = pd.DataFrame({"y": s})
    f["dow"], f["month"] = f.index.dayofweek, f.index.month
    india = holidays.India(years=range(2017, 2020))
    f["holiday"] = [int(x in india) for x in f.index.date]
    for k in (1, 7, 14):
        f[f"lag{k}"] = f.y.shift(k)
    f["mean7"] = f.y.shift(1).rolling(7).mean()
    f["mean28"] = f.y.shift(1).rolling(28).mean()
    f = f.join(load_weather(), how="left")
    return f.dropna(subset=["lag14", "mean28"])


def main():
    f = daily_frame()
    train, test = f[f.index < TEST_FROM], f[f.index >= TEST_FROM]
    fit, val = train.iloc[:-60], train.iloc[-60:]
    params = dict(objective="poisson", learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  feature_fraction=0.9, verbose=-1)
    m = lgb.train(params, lgb.Dataset(fit[FEATURES], fit.y), 3000, valid_sets=[lgb.Dataset(val[FEATURES], val.y)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    model = lgb.train(params, lgb.Dataset(train[FEATURES], train.y), m.best_iteration)

    y = test.y.values
    preds = {"model": model.predict(test[FEATURES]), "same_day_last_week": test.lag7.values,
             "average_of_last_7_days": test.mean7.values}
    mae = {k: float(np.abs(p - y).mean()) for k, p in preds.items()}
    mape = {k: float((np.abs(p - y) / np.maximum(y, 1)).mean()) for k, p in preds.items()}

    # Shape of the week and the year, used by the demo hospital (scaled to its own size).
    mean = train.y.mean()
    profile = {"dow": (train.groupby("dow").y.mean() / mean).round(3).tolist(),
               "month": (f.groupby("month").y.mean() / f.y.mean()).reindex(range(1, 13)).fillna(1).round(3).tolist(),
               "source": "Hero DMC Heart Institute, Ludhiana, Apr 2017 to Mar 2019"}
    gain = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    metrics = {
        "n_train_days": int(len(train)), "n_test_days": int(len(test)), "mean_daily_admissions": float(f.y.mean()),
        "mae": mae, "mape": mape,
        "test_series": [[d.strftime("%Y-%m-%d"), int(a), round(float(p), 1)] for d, a, p in zip(test.index, y, preds["model"])],
        "dow_profile": dict(zip(names, profile["dow"])),
        "top_features": [[k, round(float(v / gain.sum()), 4)] for k, v in gain.items()],
    }
    metrics["summary"] = {
        "Daily error, our model": f"{mae['model']:.1f} admissions",
        "Daily error, average of last 7 days": f"{mae['average_of_last_7_days']:.1f}",
        "Daily error, same day last week": f"{mae['same_day_last_week']:.1f}",
        "Average admissions per day": f"{f.y.mean():.1f}",
        "Busiest / quietest day": f"{names[int(np.argmax(profile['dow']))]} / {names[int(np.argmin(profile['dow']))]}",
    }
    metrics["note"] = "Mean absolute error per day on Dec 2018 to Mar 2019, which the model never saw."
    model.save_model(str(MODELS / "admissions_forecast.txt"))
    save_json(profile, MODELS / "admissions_profile.json")
    save_json(metrics, MODELS / "admissions_metrics.json")
    print("MAE", {k: round(v, 2) for k, v in mae.items()}, "| mean/day", round(f.y.mean(), 1))
    print("dow profile", metrics["dow_profile"]); print("top", metrics["top_features"][:6])


if __name__ == "__main__":
    main()
