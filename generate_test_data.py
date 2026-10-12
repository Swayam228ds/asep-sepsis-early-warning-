"""Synthetic patient-hour generator for exercising sepsis-model.onnx.

Each patient is an hourly vital-sign trajectory. Septic patients drift toward
SIRS / Sepsis-3 physiology as they approach onset; SepsisLabel follows the
PhysioNet 2019 convention (1 from six hours before onset onward). Equipment is
assigned per facility profile, readings drop out hourly, and gaps are filled by
last-observation-carried-forward, mirroring how the application feeds the model.

This data is for demonstrating and smoke-testing the pipeline. It is not
PhysioNet, and metrics computed on it do not reproduce the README's held-out
results.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["age", "hoursSinceAdmission", "HR", "Temp", "Resp", "SBP", "DBP", "O2Sat", "EtCO2"]
VITALS = FEATURES[2:]

# Vitals each facility can measure at all.
FACILITY_PROFILES = {
    "full": {"weight": 0.35, "vitals": VITALS},
    "medium": {"weight": 0.30, "vitals": ["HR", "Temp", "Resp", "SBP", "DBP", "O2Sat"]},
    "minimal": {"weight": 0.20, "vitals": ["HR", "Temp", "Resp", "SBP", "DBP"]},
    "pulse_ox_only": {"weight": 0.15, "vitals": ["HR", "Temp", "O2Sat"]},
}

# Plausibility bounds; readings outside these are discarded before LOCF.
BOUNDS = {
    "HR": (20, 250), "Temp": (30, 43), "Resp": (4, 60), "SBP": (40, 260),
    "DBP": (20, 160), "O2Sat": (50, 100), "EtCO2": (5, 80),
}

BASELINE = {  # mean, between-patient sd
    "HR": (78, 10), "Temp": (36.9, 0.3), "Resp": (16, 2), "SBP": (122, 12),
    "DBP": (76, 8), "O2Sat": (97.5, 1.0), "EtCO2": (39, 3),
}
HOURLY_NOISE = {"HR": 5, "Temp": 0.2, "Resp": 1.5, "SBP": 7, "DBP": 5, "O2Sat": 0.8, "EtCO2": 2}

# Shift at full septic severity.
SEPTIC_SHIFT = {"HR": 35, "Temp": 1.8, "Resp": 11, "SBP": -32, "DBP": -18, "O2Sat": -6, "EtCO2": -8}


def _severity(hours, onset, rng):
    """0 well before onset, ramping to ~1 a few hours after it."""
    lead = rng.uniform(8, 18)  # how early physiology starts to turn
    return 1 / (1 + np.exp(-(hours - (onset - lead / 2)) / (lead / 5)))


def simulate_patient(pid, septic, rng, hourly_dropout):
    stay = int(rng.integers(12, 49))
    hours = np.arange(stay)
    admit_offset = int(rng.integers(0, 24))  # patient may arrive mid-stay
    age = float(np.clip(rng.normal(62, 16), 18, 95))

    facility = rng.choice(list(FACILITY_PROFILES), p=[p["weight"] for p in FACILITY_PROFILES.values()])
    available = set(FACILITY_PROFILES[facility]["vitals"])
    if facility == "full" and rng.random() < 0.3:
        available.discard("EtCO2")  # capnograph present but not in use

    base = {v: rng.normal(m, s) for v, (m, s) in BASELINE.items()}
    vitals = {v: base[v] + rng.normal(0, HOURLY_NOISE[v], stay) for v in VITALS}

    label = np.zeros(stay, dtype=int)
    onset = np.nan
    if septic:
        onset = int(rng.integers(8, stay + 6))  # onset may fall just after the window
        sev = _severity(hours, onset, rng) * rng.uniform(0.45, 1.2)
        hypothermic = rng.random() < 0.2
        for v, shift in SEPTIC_SHIFT.items():
            s = -1.5 if (v == "Temp" and hypothermic) else shift
            vitals[v] = vitals[v] + s * sev * rng.uniform(0.5, 1.3)
        label[hours >= onset - 6] = 1
    elif rng.random() < 0.15:
        # Non-septic mimic: isolated fever/tachycardia or hypotension episode.
        start = int(rng.integers(0, stay))
        bump = (hours >= start).astype(float) * rng.uniform(0.3, 0.8)
        for v in rng.choice(["HR", "Temp", "Resp", "SBP"], size=2, replace=False):
            vitals[v] = vitals[v] + SEPTIC_SHIFT[v] * bump

    df = pd.DataFrame({"patient_id": pid, "hour": hours, "facility": facility,
                       "age": round(age), "hoursSinceAdmission": hours + 1 + admit_offset})
    for v in VITALS:
        x = vitals[v].copy()
        lo, hi = BOUNDS[v]
        if v not in available:
            x[:] = np.nan
        else:
            x[rng.random(stay) < hourly_dropout] = np.nan
            x[(x < lo) | (x > hi)] = np.nan
        df[v] = x
    df[VITALS] = df[VITALS].ffill()  # LOCF within the patient
    df["O2Sat"] = df["O2Sat"].clip(upper=100)
    df["SepsisLabel"] = label
    df["onset_hour"] = onset
    return df


def generate(n_patients=400, septic_rate=0.073, hourly_dropout=0.12, seed=42):
    rng = np.random.default_rng(seed)
    septic_flags = np.zeros(n_patients, dtype=bool)
    septic_flags[: max(1, int(round(n_patients * septic_rate)))] = True  # exact count, at least one
    rng.shuffle(septic_flags)
    frames = [simulate_patient(i, s, rng, hourly_dropout) for i, s in enumerate(septic_flags)]
    df = pd.concat(frames, ignore_index=True)
    rounding = {"HR": 0, "Temp": 1, "Resp": 0, "SBP": 0, "DBP": 0, "O2Sat": 0, "EtCO2": 0}
    return df.round(rounding)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patients", type=int, default=400)
    ap.add_argument("--septic-rate", type=float, default=0.073)
    ap.add_argument("--dropout", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_clinical_test.csv")
    args = ap.parse_args()

    df = generate(args.patients, args.septic_rate, args.dropout, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_sep = df.groupby("patient_id")["SepsisLabel"].max().sum()
    print(f"wrote {args.out}: {len(df)} patient-hours, {df.patient_id.nunique()} patients, "
          f"{n_sep} septic, positive-hour rate {df.SepsisLabel.mean():.3%}")


if __name__ == "__main__":
    main()
