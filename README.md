# Hospital Bed Manager: admission, waiting list and bed planning for crowded hospitals

A web dashboard for government hospitals where patients wait days or weeks for a bed. It answers three questions every day:

1. **Will this patient need a bed?** A model trained on 560,486 real emergency visits predicts admission from what a triage nurse knows at the door.
2. **Who gets the next free bed?** A transparent priority score ranks the waiting list by urgency, days waited and admission probability.
3. **When will beds free up?** A length-of-stay model predicts each admitted patient's discharge date, so the hospital can call people off the waiting list early.

> **Decision support only.** The platform suggests; doctors decide who is admitted.

Data Science Programming course project.

| Team member | Register no. |
|---|---|
| Abhinay Kharb | 24MID0185 |
| Swayam Mohanty | 24MID0088 |
| Omi Menass | 24MID0236 |

---

## Status

| Part | Data | State |
|---|---|---|
| Triage admission model (M2) | Yale ED triage dataset | Trained. AUROC 0.880 vs 0.765 for ESI level alone, 0.853 on an ED never seen in training |
| Waiting list priority | Rule, weights editable in the dashboard | Done |
| Length-of-stay model (M3) | Hero DMC Heart Institute, Ludhiana | Waiting for the dataset; the dashboard uses a typical-stay placeholder until then |
| Admissions forecast (M1, optional) | Hero DMC Heart Institute, Ludhiana | Waiting for the dataset; flat placeholder until then |

## Run it

You need **Python 3.10 or newer**. Everything runs on a laptop CPU; no GPU is needed.

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. The first run creates a demo hospital in `data/hospital.db`: three 20-bed wards, about 50 admitted patients and 40 people on the waiting list. All names are made up. Use **Next day** to move the clock forward (patients are discharged, new ones arrive and the top of the queue is admitted) and **Reset demo** to start again.

### Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Beds occupied, waiting list size, discharges and admissions expected tomorrow, and tomorrow's bed gap. A 7-day chart of beds freeing up against expected admissions. |
| Waiting list | The queue ranked by priority. Change urgency, change the priority weights, admit the top patient, add a planned patient. |
| Emergency intake | Triage form that predicts admission probability with the main reasons, then adds the patient to the queue. |
| Bed board | Every bed by ward: free, freeing within a day, or occupied, with each patient's predicted discharge date and range. |
| Model report | Each model's results against its baseline. |

## Train the models

The trained triage model is already in `models/`, so the app works without retraining.

**M2, triage admission (Yale).** Download `Results/5v_cleandf.RData` from [yaleemmlc/admissionprediction](https://github.com/yaleemmlc/admissionprediction). It is too big for Python's `pyreadr` (it needs more than 14 GB of RAM), so convert it with R first, then train:

```bash
Rscript training/prepare_yale.R 5v_cleandf.RData data/raw/yale_triage_subset.csv.gz
python training/m2_triage.py
```

This takes about two minutes on a 4-core CPU. It uses only triage-time inputs (ESI level, age, sex, arrival mode and time, the 7 triage vitals, the 60 most common chief complaints and counts of past visits). Race, religion, insurance and other social columns are left out on purpose.

| Triage model results (test set, 112,098 visits) | AUROC | AUPRC |
|---|---|---|
| LightGBM, our model | 0.880 | 0.754 |
| Logistic regression, same inputs | 0.863 | 0.716 |
| ESI level alone (what triage uses today) | 0.765 | 0.496 |
| LightGBM trained on EDs A and B, tested on ED C | 0.853 | |

**M3 and M1 (Ludhiana).** Scripts follow once the Hero DMC files are added to `data/raw/`.

## API

| Method | Endpoint | Does |
|---|---|---|
| GET | `/api/overview` | Bed counts, tomorrow's discharges, admissions and bed gap, 7-day forecast |
| GET, POST | `/api/queue` | List the waiting list; add a patient |
| POST | `/api/queue/<id>/admit` | Put the patient in a free bed and predict their stay |
| POST | `/api/queue/<id>/urgency` | Set urgency 1 to 3 |
| POST | `/api/settings` | Change the priority weights |
| POST | `/api/triage` | Admission probability and reasons for one emergency patient |
| GET | `/api/beds` | Every bed with status and predicted discharge |
| POST | `/api/beds/<id>/discharge` | Free a bed |
| POST | `/api/simulate/next-day`, `/api/simulate/reset` | Demo clock |
| GET | `/api/metrics` | Results of every trained model |

## Project structure

```
app.py                  Flask routes for the dashboard
hospital/db.py          SQLite tables (wards, beds, patients, settings)
hospital/models.py      Loads trained models, with clearly labelled fallbacks
hospital/services.py    Waiting list priority, admit, discharge, overview, next-day simulation
hospital/seed_demo.py   Demo hospital with made-up patients
templates/index.html    Dashboard (Tailwind CSS + Chart.js)
training/               Training scripts for each model
models/                 Trained models and their metrics
sepsis_app.py           Earlier sepsis early-warning project (python sepsis_app.py)
```

## Limits

- The triage model learned from US emergency departments, and the length-of-stay model will learn from one Indian heart hospital. A real deployment would retrain on the hospital's own records.
- Demo patients are simulated; nothing here is live hospital data.

## Earlier work: sepsis early warning

The previous version of this project, a sepsis risk score from bedside vitals, still runs with `python sepsis_app.py` (files `sepsis_app.py`, `evaluate_model.py`, `generate_test_data.py`, `sepsis-model.onnx`, `MODEL_CARD.md`). It uses a model trained by another team and synthetic test data, so it is no longer part of the dashboard.

## References

- Hong WS, Haimovich AD, Taylor RA. *Predicting hospital admission at emergency department triage using machine learning.* PLoS ONE 2018;13(7):e0201016. https://doi.org/10.1371/journal.pone.0201016 (dataset: https://github.com/yaleemmlc/admissionprediction)
- [LightGBM](https://lightgbm.readthedocs.io/) · [Flask](https://flask.palletsprojects.com/) · [scikit-learn](https://scikit-learn.org/) · [Chart.js](https://www.chartjs.org/)
