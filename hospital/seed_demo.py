"""Create a demo government hospital: 3 general wards of 20 beds, a waiting list and 14 days of history.

All names are made up. Run:  python -m hospital.seed_demo
"""
import math
import random
from datetime import date, timedelta

from . import db, models, services

WARDS = {"General Medicine": 20, "Cardiology": 20, "Surgery": 20}
CONDITIONS = {
    "General Medicine": ["Fever, query dengue", "Pneumonia", "Uncontrolled diabetes", "Acute gastroenteritis",
                         "Urinary tract infection", "Anaemia work-up", "COPD flare"],
    "Cardiology": ["Chest pain, query ACS", "Heart failure", "Atrial fibrillation", "Hypertensive urgency",
                   "Post-angiography observation"],
    "Surgery": ["Appendicitis", "Hernia repair", "Gallstones", "Fracture fixation", "Wound debridement"],
}
COMPLAINTS = {"General Medicine": ["cc_fever-9weeksto74years", "cc_shortnessofbreath", "cc_weakness", "cc_emesis", "cc_abdominalpain"],
              "Cardiology": ["cc_chestpain", "cc_shortnessofbreath", "cc_palpitations", "cc_dizziness"],
              "Surgery": ["cc_abdominalpain", "cc_legpain", "cc_fall", "cc_abscess"]}
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Arjun", "Kavya", "Vikram", "Ananya", "Rahul", "Pooja", "Suresh",
         "Lakshmi", "Manoj", "Deepa", "Ravi", "Meena", "Sanjay", "Anita", "Ajay", "Sunita", "Gopal", "Rekha",
         "Harish", "Neha", "Prakash", "Jyoti", "Ramesh", "Geeta", "Kiran", "Asha"]
LAST = ["Sharma", "Patel", "Singh", "Das", "Mohanty", "Reddy", "Nayak", "Kumar", "Gupta", "Rao", "Sahoo",
        "Iyer", "Mishra", "Verma", "Behera", "Pillai", "Jain", "Panda", "Yadav", "Rout"]
DAILY_ARRIVALS = 12


def random_patient(rng, on, added_on=None):
    ward = rng.choice(list(WARDS))
    source = "emergency" if rng.random() < 0.4 else "opd"
    age = int(min(90, max(18, rng.gauss(52, 17))))
    gender = rng.choice(["Male", "Female"])
    clinical = {"DM": int(rng.random() < 0.3), "HTN": int(rng.random() < 0.35), "CAD": int(ward == "Cardiology" and rng.random() < 0.5),
                "rural": int(rng.random() < 0.5)}
    data = {"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "age": age, "gender": gender, "source": source,
            "ward": ward, "condition": rng.choice(CONDITIONS[ward]),
            "added_on": (added_on or on).isoformat(), "urgency": rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]}
    if source == "emergency":
        esi = rng.choices([1, 2, 3, 4], [0.03, 0.3, 0.5, 0.17])[0]
        sick = (4 - esi) / 3
        triage = {"esi": esi, "age": age, "gender": gender, "arrivalmode": rng.choice(["ambulance", "Car", "Walk-in"]),
                  "triage_vital_hr": round(rng.gauss(85 + 25 * sick, 12)), "triage_vital_sbp": round(rng.gauss(130 - 20 * sick, 18)),
                  "triage_vital_dbp": round(rng.gauss(80 - 10 * sick, 10)), "triage_vital_rr": round(rng.gauss(17 + 5 * sick, 3)),
                  "triage_vital_o2": min(100, round(rng.gauss(97 - 4 * sick, 2))), "triage_vital_temp": round(rng.gauss(98.4 + 1.2 * sick, 0.8), 1),
                  "n_edvisits": rng.choice([0, 0, 1, 2, 4]), "n_admissions": rng.choice([0, 0, 0, 1, 2]),
                  rng.choice(COMPLAINTS[ward]): 1}
        clinical.update(triage)
        data["urgency"] = models.urgency_from_esi(esi)
    data["clinical"] = clinical
    median = (6.0 if source == "emergency" else 4.0) + (1.5 if age >= 65 else 0)
    true_los = max(1.0, round(math.exp(rng.gauss(math.log(median), 0.45)), 1))
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
