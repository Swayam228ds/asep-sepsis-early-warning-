"""Create a demo government hospital: 3 wards of 20 beds, a waiting list and 14 days of history.

Patients are real held-out Ludhiana admissions with made-up names. Run:  python -m hospital.seed_demo
"""
import math
import random
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from . import db, models, services

WARDS = {"Cardiology A": 24, "Cardiology B": 24, "Medicine": 12}
NON_CARDIAC = {"Acute kidney injury", "Stroke", "Urinary tract infection", "Chest Infection", "Severe Anaemia"}
COMPLAINTS = {"Cardiology": ["cc_chestpain", "cc_shortnessofbreath", "cc_palpitations", "cc_dizziness", "cc_syncope"],
              "Medicine": ["cc_weakness", "cc_shortnessofbreath", "cc_fever-9weeksto74years", "cc_alteredmentalstatus"]}
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Arjun", "Kavya", "Vikram", "Ananya", "Rahul", "Pooja", "Suresh",
         "Lakshmi", "Manoj", "Deepa", "Ravi", "Meena", "Sanjay", "Anita", "Ajay", "Sunita", "Gopal", "Rekha",
         "Harish", "Neha", "Prakash", "Jyoti", "Ramesh", "Geeta", "Kiran", "Asha"]
LAST = ["Sharma", "Patel", "Singh", "Das", "Mohanty", "Reddy", "Nayak", "Kumar", "Gupta", "Rao", "Sahoo",
        "Iyer", "Mishra", "Verma", "Behera", "Pillai", "Jain", "Panda", "Yadav", "Rout"]
DAILY_ARRIVALS = 10
DEMO_FILE = models.MODELS / "demo_patients.csv"


@lru_cache(maxsize=1)
def _real_admissions():
    """Held-out Ludhiana test admissions (no names, dates or record numbers), or None before training."""
    if not DEMO_FILE.exists():
        return None
    return pd.read_csv(DEMO_FILE).to_dict("records")


def _triage_fields(rng, age, gender, emergency_severity, group):
    """Made-up triage observations for the emergency intake model (the Ludhiana data has none)."""
    esi = 2 if emergency_severity else rng.choices([2, 3, 4], [0.3, 0.55, 0.15])[0]
    sick = (4 - esi) / 3
    return {"esi": esi, "age": age, "gender": gender, "arrivalmode": rng.choice(["ambulance", "Car", "Walk-in"]),
            "triage_vital_hr": round(rng.gauss(85 + 25 * sick, 12)), "triage_vital_sbp": round(rng.gauss(130 - 20 * sick, 18)),
            "triage_vital_dbp": round(rng.gauss(80 - 10 * sick, 10)), "triage_vital_rr": round(rng.gauss(17 + 5 * sick, 3)),
            "triage_vital_o2": min(100, round(rng.gauss(97 - 4 * sick, 2))), "triage_vital_temp": round(rng.gauss(98.4 + 1.2 * sick, 0.8), 1),
            "n_edvisits": rng.choice([0, 0, 1, 2, 4]), "n_admissions": rng.choice([0, 0, 0, 1, 2]), rng.choice(COMPLAINTS[group]): 1}


def random_patient(rng, on, added_on=None):
    """A demo patient. Clinical details and the true stay come from a real held-out Ludhiana admission;
    the name is made up. Returns (data, true_length_of_stay_days)."""
    real = _real_admissions()
    if real:
        r = dict(rng.choice(real))
        true_los, condition = float(r.pop("los")), r.pop("condition")
        age, gender = int(r["AGE"]), "Male" if r["male"] else "Female"
        source = "emergency" if r["emergency"] else "opd"
        clinical = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
    else:  # before m3_los.py has been run
        age, gender = int(min(90, max(18, rng.gauss(58, 14)))), rng.choice(["Male", "Female"])
        source = "emergency" if rng.random() < 0.7 else "opd"
        condition, clinical = "Cardiac evaluation", {}
        true_los = max(1.0, round(math.exp(rng.gauss(math.log(5), 0.5)), 1))
    group = "Medicine" if condition in NON_CARDIAC else "Cardiology"
    ward = "Medicine" if group == "Medicine" else rng.choice(["Cardiology A", "Cardiology B"])
    data = {"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "age": age, "gender": gender, "source": source,
            "ward": ward, "condition": condition, "added_on": (added_on or on).isoformat(),
            "urgency": rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]}
    if source == "emergency":
        severe = bool(clinical.get("CARDIOGENIC SHOCK") or clinical.get("STEMI") or clinical.get("SHOCK"))
        clinical.update(_triage_fields(rng, age, gender, severe, group))
        data["urgency"] = models.urgency_from_esi(clinical["esi"])
    data["clinical"] = clinical
    return data, true_los


def seed(con, today=date(2026, 10, 19), seed_value=7):
    rng = random.Random(seed_value)
    for t in ("beds", "patients", "wards", "daily_log", "settings"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    db.init(con)
    db.set_setting(con, "today", today.isoformat())
    for name, n in WARDS.items():
        wid = con.execute("INSERT INTO wards (name, kind) VALUES (?, 'general')", (name,)).lastrowid
        prefix = "".join(w[0] for w in name.split())
        for i in range(1, n + 1):
            con.execute("INSERT INTO beds (ward_id, label) VALUES (?, ?)", (wid, f"{prefix}-{i:02d}"))

    # ~50 of 60 beds occupied, each part-way through its stay.
    for _ in range(50):
        data, true_los = random_patient(rng, today)
        admitted_on = today - timedelta(days=rng.randint(0, max(0, round(true_los) - 1)))
        data["added_on"] = (admitted_on - timedelta(days=rng.randint(1, 20))).isoformat()
        pid = services.add_patient(con, data, true_los)
        bed = services.free_bed(con, data["ward"])
        if bed is None:
            continue
        services.admit(con, pid, bed, on=admitted_on)

    # Waiting list: ~40 patients who have waited 0 to 35 days.
    for _ in range(40):
        data, true_los = random_patient(rng, today, today - timedelta(days=rng.randint(0, 35)))
        services.add_patient(con, data, true_los)

    for i in range(14, 0, -1):
        d = today - timedelta(days=i)
        con.execute("INSERT INTO daily_log VALUES (?, ?, ?, ?, ?)",
                    (d.isoformat(), DAILY_ARRIVALS, DAILY_ARRIVALS, max(0, round(rng.gauss(DAILY_ARRIVALS, 3))), 50))


if __name__ == "__main__":
    with db.session() as con:
        seed(con)
        print(services.overview(con)["tomorrow"])
