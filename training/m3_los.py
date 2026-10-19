"""M3: how many days will this patient occupy the bed? Trained on Hero DMC Heart Institute, Ludhiana.

Input: data/raw/hdhi/HDHI Admission data.csv
Output: models/los_p50.txt, models/los_p90.txt, models/los_meta.json, models/los_metrics.json,
        models/demo_patients.csv (held-out test admissions used to fill the demo hospital)

Two quantile models: P50 is the likely stay, P90 is a stay that 9 in 10 similar patients finish within.
Split by time: the last 4 months (Dec 2018 to Mar 2019) are the test set, as a hospital would use it.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import MODELS, save_json
from hdhi import DIAGNOSES, FEATURES, load_admissions

TEST_FROM = "2018-12-01"
VALID_FROM = "2018-10-01"


def condition_label(row):
    for c in ["CARDIOGENIC SHOCK", "STEMI", "ACS", "HEART FAILURE", "AKI", "CVA INFRACT", "AF", "VT", "CHB",
              "VALVULAR", "STABLE ANGINA", "ATYPICAL CHEST PAIN", "CHEST INFECTION", "SEVERE ANAEMIA", "UTI"]:
        if row.get(c) == 1:
            return c.title().replace("Acs", "Acute coronary syndrome").replace("Stemi", "STEMI").replace("Aki", "Acute kidney injury") \
                .replace("Af", "Atrial fibrillation").replace("Vt", "Ventricular tachycardia").replace("Chb", "Complete heart block") \
                .replace("Cva Infract", "Stroke").replace("Uti", "Urinary tract infection")
    return "Cardiac evaluation"


def main():
    d = load_admissions()
    train = d[d.admit_date < TEST_FROM]
    test = d[d.admit_date >= TEST_FROM]
    fit, val = train[train.admit_date < VALID_FROM], train[train.admit_date >= VALID_FROM]
    print(f"{len(d):,} admissions | train {len(train):,} | test {len(test):,} ({TEST_FROM} onwards)")

    params = dict(objective="quantile", learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    models = {}
    for name, alpha in [("p50", 0.5), ("p90", 0.9)]:
        m = lgb.train({**params, "alpha": alpha}, lgb.Dataset(fit[FEATURES], fit.los), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(val[FEATURES], val.los)], callbacks=[lgb.early_stopping(100, verbose=False)])
        # Refit on all training months with the chosen number of rounds.
        models[name] = lgb.train({**params, "alpha": alpha}, lgb.Dataset(train[FEATURES], train.los), num_boost_round=m.best_iteration)

    y = test.los.values
    p50 = models["p50"].predict(test[FEATURES])
    p90 = np.maximum(models["p90"].predict(test[FEATURES]), p50)
    base = test.emergency.map(train.groupby("emergency").los.median()).values  # median stay by admission type
    base_all = np.full(len(test), train.los.median())

    def summary(pred):
        err = np.abs(pred - y)
        return {"mae_days": float(err.mean()), "within_2_days": float((err <= 2).mean())}

    by_type = {}
    for flag, label in [(1, "emergency"), (0, "opd")]:
        mask = test.emergency.values == flag
        by_type[label] = {"n": int(mask.sum()), "mae_model": float(np.abs(p50[mask] - y[mask]).mean()),
                          "mae_baseline": float(np.abs(base[mask] - y[mask]).mean())}
    gain = pd.Series(models["p50"].feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    # What the dashboard uses: how many patients will leave on each day.
    def daily_discharges(stay):
        out = test.admit_date + pd.to_timedelta(np.round(stay) - 1, unit="D")
        return out.value_counts()
    days = pd.date_range("2018-12-15", "2019-03-31")  # skip the first 2 weeks, whose patients were admitted earlier
    actual = daily_discharges(y).reindex(days, fill_value=0)
    disc_model = np.abs(daily_discharges(p50).reindex(days, fill_value=0) - actual).mean()
    disc_base = np.abs(daily_discharges(base).reindex(days, fill_value=0) - actual).mean()
    metrics = {
        "n_train": int(len(train)), "n_test": int(len(test)), "test_period": [TEST_FROM, str(test.admit_date.max().date())],
        "mean_stay_days": float(d.los.mean()), "median_stay_days": float(d.los.median()),
        "model_p50": summary(p50), "baseline_by_type": summary(base), "baseline_overall_median": summary(base_all),
        "p90_coverage": float((y <= p90).mean()), "by_admission_type": by_type,
        "daily_discharges_mae": {"model": float(disc_model), "baseline_by_type": float(disc_base),
                                 "actual_mean_per_day": float(actual.mean())},
        "top_features": [[k, round(float(v / gain.sum()), 4)] for k, v in gain.head(12).items()],
    }
    metrics["summary"] = {
        "Average error, our model": f"{metrics['model_p50']['mae_days']:.2f} days",
        "Average error, median stay by admission type": f"{metrics['baseline_by_type']['mae_days']:.2f} days",
        "Predicted within 2 days": f"{metrics['model_p50']['within_2_days']:.0%} vs {metrics['baseline_by_type']['within_2_days']:.0%}",
        "Stays finished by the P90 date": f"{metrics['p90_coverage']:.0%}",
        "Daily discharge count error": f"{disc_model:.1f} vs {disc_base:.1f} patients/day (actual ~{actual.mean():.0f}/day)",
        "Test admissions (Dec 2018 to Mar 2019)": f"{len(test):,}",
    }
    metrics["note"] = "Average error in predicted stay on later months the model never saw. The baseline is what a ward would guess today."

    for name, m in models.items():
        m.save_model(str(MODELS / f"los_{name}.txt"))
    save_json({"features": FEATURES, "categories": {}, "target_transform": None}, MODELS / "los_meta.json")
    save_json(metrics, MODELS / "los_metrics.json")

    demo = test.sample(n=min(1500, len(test)), random_state=0).copy()
    demo["condition"] = demo.apply(condition_label, axis=1)
    demo[FEATURES + ["los", "condition"]].to_csv(MODELS / "demo_patients.csv", index=False)

    m = metrics
    print(f"MAE model {m['model_p50']['mae_days']:.2f} d | baseline by type {m['baseline_by_type']['mae_days']:.2f} d "
          f"| overall median {m['baseline_overall_median']['mae_days']:.2f} d | within 2 d {m['model_p50']['within_2_days']:.0%} "
          f"vs {m['baseline_by_type']['within_2_days']:.0%} | P90 coverage {m['p90_coverage']:.0%}")
    print("daily discharges MAE:", m["daily_discharges_mae"])
    print("top:", m["top_features"][:6])


if __name__ == "__main__":
    main()
