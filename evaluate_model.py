"""Score a patient-hour CSV with sepsis-model.onnx and write dashboard metrics.

Reports discrimination (AUROC/AUPRC), confusion matrices at the README's tier
cutoffs, observed sepsis rate per risk tier, bedside-rule baselines computed
from the same columns, per-facility results, and patient-level alarm lead time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

FEATURES = ["age", "hoursSinceAdmission", "HR", "Temp", "Resp", "SBP", "DBP", "O2Sat", "EtCO2"]
LOW_CUTOFF, HIGH_CUTOFF = 0.3, 0.6
MODEL_PATH = Path(__file__).with_name("sepsis-model.onnx")


def load_session(path=MODEL_PATH):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def score(session, df):
    x = df[FEATURES].to_numpy(dtype=np.float32)  # pandas NaN stays NaN
    return session.run(["riskScore"], {"input": x})[0]


def tier(score_value):
    if score_value >= HIGH_CUTOFF:
        return "High"
    if score_value >= LOW_CUTOFF:
        return "Medium"
    return "Low"


# Bedside rules, vitals only. Missing values contribute nothing, as at the bedside.
def qsofa(df):
    return (df.Resp >= 22).astype(int) + (df.SBP <= 100).astype(int)


def sirs(df):
    return ((df.Temp > 38) | (df.Temp < 36)).astype(int) + (df.HR > 90).astype(int) + (df.Resp > 20).astype(int)


def news2_partial(df):
    def band(x, edges_points):
        out = np.zeros(len(x))
        for cond, pts in edges_points:
            out = np.where(cond(x) & x.notna(), np.maximum(out, pts), out)
        return out
    resp = band(df.Resp, [(lambda r: r <= 8, 3), (lambda r: (r >= 9) & (r <= 11), 1),
                          (lambda r: (r >= 21) & (r <= 24), 2), (lambda r: r >= 25, 3)])
    spo2 = band(df.O2Sat, [(lambda s: s <= 91, 3), (lambda s: (s >= 92) & (s <= 93), 2),
                           (lambda s: (s >= 94) & (s <= 95), 1)])
    sbp = band(df.SBP, [(lambda s: s <= 90, 3), (lambda s: (s >= 91) & (s <= 100), 2),
                        (lambda s: (s >= 101) & (s <= 110), 1), (lambda s: s >= 220, 3)])
    hr = band(df.HR, [(lambda h: h <= 40, 3), (lambda h: (h >= 41) & (h <= 50), 1),
                      (lambda h: (h >= 91) & (h <= 110), 1), (lambda h: (h >= 111) & (h <= 130), 2),
                      (lambda h: h >= 131, 3)])
    temp = band(df.Temp, [(lambda t: t <= 35, 3), (lambda t: (t > 35) & (t <= 36), 1),
                          (lambda t: (t > 38) & (t <= 39), 1), (lambda t: t > 39, 2)])
    return resp + spo2 + sbp + hr + temp


def safe_auc(fn, y, s):
    return float(fn(y, s)) if len(np.unique(y)) == 2 else None


def confusion(y, s, threshold):
    pred = s >= threshold
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else None
    sens, ppv = div(tp, tp + fn), div(tp, tp + fp)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": div(tn, tn + fp), "ppv": ppv, "npv": div(tn, tn + fn),
        "f1": div(2 * sens * ppv, sens + ppv) if sens is not None and ppv is not None else None,
    }


def downsample_curve(xs, ys, thresholds, n=200):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return {"x": xs[idx].round(4).tolist(), "y": ys[idx].round(4).tolist(),
            "thresholds": np.nan_to_num(thresholds[np.minimum(idx, len(thresholds) - 1)], posinf=1.0).round(4).tolist()}


def patient_alarms(df):
    """Septic patients alarmed (score >= high cutoff) before onset, and how early."""
    leads, alarmed, total = [], 0, 0
    for _, p in df[df.onset_hour.notna()].groupby("patient_id"):
        total += 1
        onset = p.onset_hour.iloc[0]
        hits = p[(p.riskScore >= HIGH_CUTOFF) & (p.hour < onset)]
        if len(hits):
            alarmed += 1
            leads.append(float(onset - hits.hour.min()))
    controls = df[df.onset_hour.isna()].groupby("patient_id").riskScore.max()
    return {
        "septic_patients": total, "alarmed_before_onset": alarmed,
        "median_lead_hours": float(np.median(leads)) if leads else None,
        "control_patients": int(len(controls)),
        "controls_ever_alarmed": int((controls >= HIGH_CUTOFF).sum()),
    }


def evaluate(df, session=None):
    session = session or load_session()
    df = df.copy()
    df["riskScore"] = score(session, df)
    y, s = df.SepsisLabel.to_numpy(), df.riskScore.to_numpy()
    base_rate = float(y.mean())

    fpr, tpr, roc_t = roc_curve(y, s)
    prec, rec, pr_t = precision_recall_curve(y, s)
    auprc = safe_auc(average_precision_score, y, s)

    tiers = []
    for name, lo, hi in [("Low", 0, LOW_CUTOFF), ("Medium", LOW_CUTOFF, HIGH_CUTOFF), ("High", HIGH_CUTOFF, 1.01)]:
        m = (s >= lo) & (s < hi)
        tiers.append({"tier": name, "n": int(m.sum()), "share": float(m.mean()),
                      "observed_rate": float(y[m].mean()) if m.any() else None,
                      "mean_score": float(s[m].mean()) if m.any() else None})

    baselines = []
    for name, sc in [("qSOFA (vitals only)", qsofa(df)), ("SIRS (vitals only)", sirs(df)),
                     ("NEWS2 (partial)", news2_partial(df)), ("This model", s)]:
        ap_ = safe_auc(average_precision_score, y, sc)
        baselines.append({"model": name, "auprc": ap_, "auroc": safe_auc(roc_auc_score, y, sc),
                          "lift": ap_ / base_rate if ap_ and base_rate else None})

    facilities = []
    for fac, g in df.groupby("facility"):
        facilities.append({"facility": fac, "patients": int(g.patient_id.nunique()),
                           "septic_patients": int(g.groupby("patient_id").SepsisLabel.max().sum()),
                           "auroc": safe_auc(roc_auc_score, g.SepsisLabel, g.riskScore),
                           "sensitivity": confusion(g.SepsisLabel.to_numpy(), g.riskScore.to_numpy(), HIGH_CUTOFF)["sensitivity"]})

    missing = {f: float(df[f].isna().mean()) for f in FEATURES[2:]}

    return {
        "summary": {"rows": int(len(df)), "patients": int(df.patient_id.nunique()),
                    "septic_patients": int(df.groupby("patient_id").SepsisLabel.max().sum()),
                    "positive_rate": base_rate, "auroc": safe_auc(roc_auc_score, y, s),
                    "auprc": auprc, "lift": auprc / base_rate if auprc and base_rate else None},
        "roc": downsample_curve(fpr, tpr, roc_t),
        "pr": downsample_curve(rec[::-1], prec[::-1], np.append(pr_t, 1.0)[::-1]),
        "confusion": [confusion(y, s, LOW_CUTOFF), confusion(y, s, HIGH_CUTOFF)],
        "tiers": tiers, "baselines": baselines, "facilities": facilities,
        "alarms": patient_alarms(df), "missingness": missing,
        "score_histogram": {
            "bins": np.linspace(0, 1, 21).round(2).tolist(),
            "septic": np.histogram(s[y == 1], bins=20, range=(0, 1))[0].tolist(),
            "non_septic": np.histogram(s[y == 0], bins=20, range=(0, 1))[0].tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/synthetic_clinical_test.csv")
    ap.add_argument("--out", default="data/model_metrics.json")
    args = ap.parse_args()

    metrics = evaluate(pd.read_csv(args.data))
    Path(args.out).write_text(json.dumps(metrics, indent=2))
    sm = metrics["summary"]
    print(f"AUROC {sm['auroc']:.4f}  AUPRC {sm['auprc']:.4f}  lift {sm['lift']:.1f}x  "
          f"base rate {sm['positive_rate']:.2%}")
    for t in metrics["tiers"]:
        rate = f"{t['observed_rate']:.2%}" if t["observed_rate"] is not None else "-"
        print(f"  {t['tier']:<6} n={t['n']:<6} observed sepsis {rate}")
    for b in metrics["baselines"]:
        print(f"  {b['model']:<22} AUPRC {b['auprc']:.4f}  AUROC {b['auroc']:.4f}")
    a = metrics["alarms"]
    print(f"  alarmed {a['alarmed_before_onset']}/{a['septic_patients']} septic patients before onset, "
          f"median lead {a['median_lead_hours']} h; {a['controls_ever_alarmed']}/{a['control_patients']} controls ever alarmed")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
