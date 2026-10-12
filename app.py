"""Flask API and UI for the asep sepsis early-warning model."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from evaluate_model import FEATURES, HIGH_CUTOFF, LOW_CUTOFF, evaluate, load_session, tier
from generate_test_data import BOUNDS, generate

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synthetic_clinical_test.csv"
METRICS_PATH = ROOT / "data" / "model_metrics.json"

# Bounds for the patient-context columns; vitals reuse the generator's plausibility bounds.
INPUT_BOUNDS = {"age": (18, 120), "hoursSinceAdmission": (0, 10000), **BOUNDS}

# Adult reference ranges used only to flag readings in the UI.
NORMAL_RANGES = {"HR": (60, 100), "Temp": (36.0, 38.0), "Resp": (12, 20), "SBP": (100, 140),
                 "DBP": (60, 90), "O2Sat": (94, 100), "EtCO2": (35, 45), "MAP": (65, 110)}

GUIDANCE = {
    "Low": ["Continue routine monitoring at the usual interval.",
            "Re-score if any vital sign changes or the patient's condition worsens."],
    "Medium": ["Increase observation frequency and re-score within 1-2 hours.",
               "Review for a possible source of infection; consider lactate if available.",
               "Measure any missing vitals to improve the estimate."],
    "High": ["Request prompt clinician review.",
             "Assess for sepsis per local protocol (e.g. lactate, cultures, fluid status).",
             "Consider referral or escalation if the facility cannot manage deterioration."],
}

app = Flask(__name__)
CORS(app)
session = load_session()


def ensure_data():
    if not DATA_PATH.exists() or not METRICS_PATH.exists():
        regenerate(400, 42)


def regenerate(n_patients, seed):
    df = generate(n_patients=n_patients, seed=seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(evaluate(df, session)))


def parse_features(payload):
    """Return (row, errors). Omitted, null or blank fields become NaN, never 0."""
    row, errors = [], {}
    for name in FEATURES:
        raw = payload.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            row.append(math.nan)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = "not a number"
            continue
        lo, hi = INPUT_BOUNDS[name]
        if not (lo <= value <= hi) or math.isnan(value):
            errors[name] = f"outside plausible range {lo}-{hi}"
            continue
        row.append(value)
    return row, errors


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES, normal=NORMAL_RANGES,
                           low=LOW_CUTOFF, high=HIGH_CUTOFF)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    row, errors = parse_features(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    values = dict(zip(FEATURES, row))
    vitals_present = [f for f in FEATURES[2:] if not math.isnan(values[f])]
    if len(vitals_present) < 2:
        return jsonify({"errors": {"_": "Enter at least two vital signs to score."}}), 400

    risk = float(session.run(["riskScore"], {"input": np.array([row], dtype=np.float32)})[0][0])
    sbp, dbp = values["SBP"], values["DBP"]
    map_value = dbp + (sbp - dbp) / 3  # same formula as the ONNX graph; NaN if either is missing
    level = tier(risk)

    flags = []
    for name, (lo, hi) in NORMAL_RANGES.items():
        v = map_value if name == "MAP" else values[name]
        if not math.isnan(v) and not (lo <= v <= hi):
            flags.append({"feature": name, "value": round(v, 1), "direction": "low" if v < lo else "high",
                          "range": [lo, hi]})

    return jsonify({
        "riskScore": round(risk, 4), "tier": level, "MAP": nan_to_none(round(map_value, 1)),
        "guidance": GUIDANCE[level], "abnormal": flags,
        "completeness": {"measured": vitals_present,
                         "missing": [f for f in FEATURES[2:] if f not in vitals_present]},
    })


@app.get("/api/metrics")
def metrics():
    ensure_data()
    return app.response_class(METRICS_PATH.read_text(), mimetype="application/json")


@app.get("/api/dataset")
def dataset():
    ensure_data()
    df = pd.read_csv(DATA_PATH)
    limit = min(int(request.args.get("limit", 200)), 2000)
    only = request.args.get("label")
    if only in ("0", "1"):
        df = df[df.SepsisLabel == int(only)]
    df = df.sample(n=min(limit, len(df)), random_state=0).sort_values(["patient_id", "hour"])
    cols = ["patient_id", "hour", "facility", *FEATURES, "SepsisLabel"]
    return jsonify(json.loads(df[cols].to_json(orient="records")))


@app.post("/api/generate-data")
def generate_data():
    payload = request.get_json(silent=True) or {}
    n = int(payload.get("patients", 400))
    seed = int(payload.get("seed", 42))
    if not 50 <= n <= 5000:
        return jsonify({"errors": {"patients": "must be between 50 and 5000"}}), 400
    regenerate(n, seed)
    return metrics()


if __name__ == "__main__":
    ensure_data()
    app.run(host="127.0.0.1", port=5000, debug=False)
