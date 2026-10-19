"""M2: will this emergency patient be admitted? Trained on the Yale ED triage dataset.

Input: data/raw/yale_triage_subset.csv.gz (triage-time columns only, see README).
Output: models/triage_admit.txt, models/triage_admit_meta.json, models/triage_admit_metrics.json

Only information a triage nurse has is used: ESI level, age, sex, arrival mode and hour,
triage vitals, chief complaint and counts of past visits. Race, religion, insurance and
other social columns are left out on purpose.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS, RAW, curve_points, save_json

CATEGORICAL = ["gender", "arrivalmode", "arrivalhour_bin", "previousdispo"]
NUMERIC = ["esi", "age", "n_edvisits", "n_admissions", "n_surgeries"]
VITALS = ["triage_vital_hr", "triage_vital_sbp", "triage_vital_dbp", "triage_vital_rr",
          "triage_vital_o2", "triage_vital_o2_device", "triage_vital_temp"]
TOP_COMPLAINTS = 60  # most common chief complaints kept as yes/no flags


def build_features(df, meta):
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC + VITALS:
        X[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in CATEGORICAL:
        cats = meta["categories"][c]
        X[c] = pd.Categorical(df.get(c), categories=cats)
    for c in meta["complaints"]:
        X[c] = df[c].fillna(0).astype(np.int8) if c in df else 0
    return X[meta["features"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW / "yale_triage_subset.csv.gz"))
    args = ap.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    y = (df["disposition"] == "Admit").astype(int).values
    print(f"{len(df):,} visits, admit rate {y.mean():.1%}")

    cc_cols = [c for c in df.columns if c.startswith("cc_") and c != "cc_other"]
    complaints = df[cc_cols].sum().sort_values(ascending=False).index[:TOP_COMPLAINTS].tolist()
    meta = {
        "categories": {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL},
        "complaints": complaints,
    }
    meta["features"] = NUMERIC + VITALS + CATEGORICAL + complaints
    X = build_features(df, meta)

    # Random split for the headline numbers, plus a site split: train on EDs A and B, test on C.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=1)
    model = lgb.train(params, lgb.Dataset(X_fit, y_fit), num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    p = model.predict(X_te, num_iteration=model.best_iteration)

    # Baselines: ESI alone (what triage nurses use) and logistic regression on the same inputs.
    esi_score = -X_te["esi"].fillna(3).values
    num = X_tr.select_dtypes("number").columns
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    med = X_tr[num].median()
    lr.fit(X_tr[num].fillna(med), y_tr)
    p_lr = lr.predict_proba(X_te[num].fillna(med))[:, 1]

    site = df.loc[X_te.index, "dep_name"].values
    site_tr = df["dep_name"].isin(["A", "B"]).values
    m_site = lgb.train(params, lgb.Dataset(X[site_tr], y[site_tr]), num_boost_round=model.best_iteration)
    p_site = m_site.predict(X[~site_tr])

    fpr, tpr, _ = roc_curve(y_te, p)
    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10)
    gain = pd.Series(model.feature_importance("gain"), index=meta["features"]).sort_values(ascending=False)
    metrics = {
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "admit_rate": float(y.mean()),
        "auroc": {"model": roc_auc_score(y_te, p), "esi_only": roc_auc_score(y_te, esi_score),
                  "logistic_regression": roc_auc_score(y_te, p_lr)},
        "auprc": {"model": average_precision_score(y_te, p),
                  "esi_only": average_precision_score(y_te, esi_score),
                  "logistic_regression": average_precision_score(y_te, p_lr)},
        "unseen_site_C": {"n": int((~site_tr).sum()), "auroc": roc_auc_score(y[~site_tr], p_site)},
        "by_site_random_split": {s: roc_auc_score(y_te[site == s], p[site == s]) for s in ["A", "B", "C"]},
        "roc_curve": curve_points(fpr, tpr),
        "calibration": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(mean_pred, frac_pos)],
        "top_features": [[k, round(float(v / gain.sum()), 4)] for k, v in gain.head(15).items()],
        "best_iteration": int(model.best_iteration),
    }
    MODELS.mkdir(exist_ok=True)
    model.save_model(str(MODELS / "triage_admit.txt"), num_iteration=model.best_iteration)
    save_json(meta, MODELS / "triage_admit_meta.json")
    save_json(metrics, MODELS / "triage_admit_metrics.json")
    a = metrics["auroc"]
    print(f"AUROC model {a['model']:.3f} | ESI only {a['esi_only']:.3f} | LR {a['logistic_regression']:.3f} "
          f"| unseen ED C {metrics['unseen_site_C']['auroc']:.3f}")


if __name__ == "__main__":
    main()
