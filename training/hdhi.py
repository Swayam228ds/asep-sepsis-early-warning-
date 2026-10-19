"""Loads the Hero DMC Heart Institute (Ludhiana) admissions data.

Dates in D.O.A are written two ways (M/D/YYYY in the first rows, DD/MM/YYYY later).
Each row's 'month year' column says which reading is right.
"""
import pandas as pd

from common import RAW

HDHI = RAW / "hdhi"

# Known on the day of admission: demographics, history, admission labs and the admitting diagnosis.
# Left out on purpose: discharge date, outcome and ICU days, which are only known later.
DEMOGRAPHIC = ["AGE", "male", "rural", "emergency"]
HISTORY = ["SMOKING", "ALCOHOL", "DM", "HTN", "CAD", "PRIOR CMP", "CKD"]
LABS = ["HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "BNP", "EF", "RAISED CARDIAC ENZYMES"]
DIAGNOSES = ["SEVERE ANAEMIA", "ANAEMIA", "STABLE ANGINA", "ACS", "STEMI", "ATYPICAL CHEST PAIN", "HEART FAILURE",
             "HFREF", "HFNEF", "VALVULAR", "CHB", "SSS", "AKI", "CVA INFRACT", "CVA BLEED", "AF", "VT", "PSVT",
             "CONGENITAL", "UTI", "NEURO CARDIOGENIC SYNCOPE", "ORTHOSTATIC", "INFECTIVE ENDOCARDITIS", "DVT",
             "CARDIOGENIC SHOCK", "SHOCK", "PULMONARY EMBOLISM", "CHEST INFECTION"]
FEATURES = DEMOGRAPHIC + HISTORY + LABS + DIAGNOSES


def load_admissions(path=HDHI / "HDHI Admission data.csv"):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    month = pd.to_datetime(d["month year"], format="%b-%y").dt.to_period("M")
    us = pd.to_datetime(d["D.O.A"], format="%m/%d/%Y", errors="coerce")
    eu = pd.to_datetime(d["D.O.A"], format="%d/%m/%Y", errors="coerce")
    d["admit_date"] = us.where(us.dt.to_period("M") == month, eu)
    if d["admit_date"].isna().any():
        raise ValueError(f"{d['admit_date'].isna().sum()} admission dates could not be read")
    d["male"] = (d["GENDER"] == "M").astype(int)
    d["rural"] = (d["RURAL"] == "R").astype(int)
    d["emergency"] = (d["TYPE OF ADMISSION-EMERGENCY/OPD"] == "E").astype(int)
    for c in HISTORY + LABS + DIAGNOSES:
        d[c] = pd.to_numeric(d[c], errors="coerce")  # a few cells hold text such as 'EMPTY' or '\\'
    d["los"] = d["DURATION OF STAY"].astype(float)  # days, counting admission and discharge day
    return d.sort_values("admit_date").reset_index(drop=True)


def load_weather(path=HDHI / "HDHI Pollution Data.csv"):
    w = pd.read_csv(path, encoding="utf-8-sig")
    w.columns = [c.strip() for c in w.columns]
    w["date"] = pd.to_datetime(w["DATE"], format="%m/%d/%Y", errors="coerce")
    keep = {"AQI": "aqi", "MAX TEMP": "max_temp", "MIN TEMP": "min_temp", "HUMIDITY": "humidity"}
    w = w.dropna(subset=["date"])[["date", *keep]].rename(columns=keep)
    for c in keep.values():
        w[c] = pd.to_numeric(w[c], errors="coerce")
    return w.drop_duplicates("date").set_index("date").sort_index()
