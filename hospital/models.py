"""Loads the trained models and turns dashboard inputs into predictions.

Every model has a simple fallback so the dashboard still runs before training:
the fallback is clearly flagged in the API output ("source": "fallback").
"""
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = Path(__file__).resolve().parent.parent / "models"

FEATURE_LABELS = {
    "esi": "ESI triage level", "age": "Age", "n_edvisits": "Past ED visits (1 yr)",
    "n_admissions": "Past admissions (1 yr)", "n_surgeries": "Past surgeries",
    "triage_vital_hr": "Heart rate", "triage_vital_sbp": "Systolic BP", "triage_vital_dbp": "Diastolic BP",
    "triage_vital_rr": "Breathing rate", "triage_vital_o2": "Oxygen saturation",
    "triage_vital_o2_device": "On oxygen", "triage_vital_temp": "Temperature",
    "gender": "Sex", "arrivalmode": "Arrival mode", "arrivalhour_bin": "Arrival time",
    "previousdispo": "Last visit outcome",
}


def label(feature):
    if feature.startswith("cc_"):
        return "Complaint: " + feature[3:].replace("-", " ")
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def _load_json(name):
    p = MODELS / name
    return json.loads(p.read_text()) if p.exists() else None


@lru_cache(maxsize=None)
def _booster(name):
    import lightgbm as lgb
    p = MODELS / name
    return lgb.Booster(model_file=str(p)) if p.exists() else None


def _frame(values, meta):
    """One-row DataFrame in the exact column order and dtypes the model was trained on."""
    row = {}
    for f in meta["features"]:
        v = values.get(f)
        if f in meta.get("categories", {}):
            row[f] = pd.Categorical([v if v not in ("", None) else None], categories=meta["categories"][f])
        else:
            try:
                row[f] = [float(v)] if v not in ("", None) else [np.nan]
            except (TypeError, ValueError):
                row[f] = [np.nan]
    return pd.DataFrame(row)[meta["features"]]


def _reasons(booster, X, top=4):
    contrib = booster.predict(X, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(contrib))
    out = []
    for i in order[:top]:
        if abs(contrib[i]) < 1e-3:
            break
        f = X.columns[i]
        v = X.iloc[0, i]
        out.append({"feature": f, "label": label(f), "value": None if pd.isna(v) else (str(v) if not isinstance(v, (int, float, np.floating)) else round(float(v), 1)),
                    "effect": "raises" if contrib[i] > 0 else "lowers", "weight": round(float(contrib[i]), 3)})
    return out


# ---------- M2: triage admission ----------

def triage_meta():
    return _load_json("triage_admit_meta.json")


def predict_admit(values):
    meta, booster = triage_meta(), _booster("triage_admit.txt")
    if booster is None:
        esi = float(values.get("esi") or 3)
        prob = {1: 0.86, 2: 0.55, 3: 0.29, 4: 0.02, 5: 0.004}.get(int(esi), 0.3)
        return {"probability": prob, "reasons": [], "source": "fallback (admit rate by ESI level)"}
    X = _frame(values, meta)
    prob = float(booster.predict(X)[0])
    return {"probability": round(prob, 4), "reasons": _reasons(booster, X), "source": "model"}


def urgency_from_esi(esi):
    esi = int(float(esi or 3))
    return 3 if esi <= 2 else 2 if esi == 3 else 1


# ---------- M3: length of stay ----------

def los_meta():
    return _load_json("los_meta.json")


def predict_los(patient):
    """Return (p50, p90, source, reasons) in days for an admitted patient dict."""
    meta = los_meta()
    b50, b90 = _booster("los_p50.txt"), _booster("los_p90.txt")
    values = {**patient.get("clinical", {}), "AGE": patient.get("age"),
              "male": int(patient.get("gender") == "Male"), "emergency": int(patient.get("source") == "emergency")}
    if meta and b50 is not None:
        X = _frame(values, meta)
        transform = meta.get("target_transform")
        p50 = float(b50.predict(X)[0])
        p90 = float(b90.predict(X)[0]) if b90 is not None else p50 * 1.8
        if transform == "log1p":
            p50, p90 = float(np.expm1(p50)), float(np.expm1(p90))
        p50, p90 = max(p50, 0.5), max(p90, p50)
        return round(p50, 1), round(p90, 1), "model", _reasons(b50, X, top=3)
    base = 6.0 if patient.get("source") == "emergency" else 4.0
    if (patient.get("age") or 0) >= 65:
        base += 1.5
    return base, round(base * 1.8, 1), "fallback (typical stay by admission type)", []


# ---------- M1: admissions forecast ----------

def forecast_profile():
    """Day-of-week and month shape learned from the Ludhiana data (models/admissions_profile.json)."""
    prof = _load_json("admissions_profile.json")
    if prof:
        return prof["dow"], prof["month"], "model"
    return [1.0] * 7, [1.0] * 12, "fallback (flat week)"


def forecast_admissions(recent_daily_mean, start, days):
    dow, month, source = forecast_profile()
    out = []
    for i in range(days):
        d = date.fromordinal(start.toordinal() + i)
        out.append({"day": d.isoformat(), "expected": round(recent_daily_mean * dow[d.weekday()] * month[d.month - 1], 1)})
    return out, source


def all_metrics():
    return {name: _load_json(f"{name}_metrics.json") for name in ("triage_admit", "los", "admissions")}
