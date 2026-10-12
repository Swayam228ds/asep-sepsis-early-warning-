# asep: Sepsis Early Warning When Vitals Are Missing

A web app that estimates a patient's risk of developing **sepsis within the next 6 hours** from bedside vital signs. It is designed for primary-care clinics that cannot measure every vital: any reading that wasn't taken is simply left blank, and the model still scores the patient.

> **Decision support only.** The app shows a risk estimate and how complete the data behind it is. The clinical decision stays with the health worker.

Data Science Programming course project: final review.

| Team member | Register no. |
|---|---|
| Abhinay Kharb | 24MID0185 |
| Swayam Mohanty | 24MID0088 |
| Omi Menass | 24MID0236 |

---

## What it does

- **Bedside triage:** enter whatever vitals you have and get a risk score from 0 to 1, a risk tier (Low < 0.3 ≤ Medium < 0.6 ≤ High), suggested next steps and flags for abnormal vitals.
- **Model validation dashboard:** ROC and precision–recall curves, observed sepsis rate per risk tier, confusion matrices, and a comparison with the qSOFA, SIRS and NEWS2 bedside rules.
- **Synthetic data explorer:** generate a new test cohort, browse it, and click any row to score it.

![Triage screen](docs/triage.png)

## Setup

You need **Python 3.10 or newer**. The app runs on CPU; no GPU or internet connection is needed once the packages are installed. (The dashboard loads Tailwind CSS and Chart.js from a CDN, so the browser needs internet access for styling and charts.)

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/asep-sepsis-early-warning.git
cd asep-sepsis-early-warning
```

**2. Create a virtual environment (recommended)**

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
python app.py
```

Open **http://localhost:5000** in your browser. On the first run the app generates a synthetic test dataset and its metrics in `data/`, which takes a few seconds.

### Optional: regenerate the data and metrics from the command line

```bash
python generate_test_data.py --patients 400 --seed 42
python evaluate_model.py
```

`generate_test_data.py` writes `data/synthetic_clinical_test.csv`; `evaluate_model.py` scores it and writes `data/model_metrics.json`, printing a summary like:

```
AUROC 0.8738  AUPRC 0.2785  lift 9.2x  base rate 3.01%
  Low    n=9677   observed sepsis 0.95%
  Medium n=1349   observed sepsis 5.63%
  High   n=891    observed sepsis 21.44%
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Score one patient. JSON body with any of the 9 features; omitted, `null` or blank fields are treated as not measured. |
| `GET` | `/api/metrics` | Evaluation metrics for the current synthetic dataset. |
| `GET` | `/api/dataset?limit=200&label=1` | Sample rows from the synthetic dataset (`label` optional: `0` or `1`). |
| `POST` | `/api/generate-data` | Regenerate the dataset and metrics. Body: `{"patients": 400, "seed": 42}` (50–5000 patients). |

Example:

```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d "{\"age\":65,\"hoursSinceAdmission\":10,\"HR\":126,\"Temp\":38.9,\"Resp\":28,\"SBP\":88,\"DBP\":52,\"O2Sat\":90}"
```

```json
{"riskScore": 0.9618, "tier": "High", "MAP": 64.0, "abnormal": [...], "completeness": {"missing": ["EtCO2"], ...}, "guidance": [...]}
```

### Model inputs

| Feature | Unit | Feature | Unit |
|---|---|---|---|
| `age` | years | `SBP` | mmHg |
| `hoursSinceAdmission` | hours | `DBP` | mmHg |
| `HR` | beats/min | `O2Sat` | % |
| `Temp` | °C | `EtCO2` | mmHg |
| `Resp` | breaths/min | | |

A missing vital must be sent as missing (`NaN`), **never as 0**. Mean arterial pressure is derived from SBP and DBP inside the model. The API rejects physiologically impossible values (for example SBP 0) and needs at least two vitals to score.

## Project structure

```
app.py                  Flask API and web server
generate_test_data.py   Synthetic patient-hour generator
evaluate_model.py       Scoring, metrics and baseline comparison
templates/index.html    Dashboard (Tailwind CSS + Chart.js)
sepsis-model.onnx       Pre-trained LightGBM model (ONNX)
MODEL_CARD.md           Model card from the model's authors
project_plan.md         Original architecture and implementation plan
asep_final_review.pptx  Final review presentation
docs/triage.png         Screenshot used in this README
```

## About the model and the data

- **Model:** `sepsis-model.onnx` is the pre-trained **asep-sepsis-onnx** model by **team MyLove (Datathon 2026)**, a LightGBM model trained on the PhysioNet/CinC Challenge 2019 dataset. See [MODEL_CARD.md](MODEL_CARD.md) for how it was trained and the authors' own results. Our work is the data generator, evaluation, API, dashboard and testing around it.
- **Evaluation data is synthetic.** Real ICU records require credentialed access, so `generate_test_data.py` creates realistic hourly vital-sign trajectories with sepsis onsets, equipment profiles and sensor dropouts. Results on this data show the system working end to end; they are not clinical performance figures.
- **Known limitation:** the generator shifts septic patients' vitals across the same thresholds that SIRS and NEWS2 check, so those rules score unrealistically well on synthetic data (partial NEWS2 beats the model here). Validating on real PhysioNet records is the main next step.

## References

- Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care Medicine 2020;48(2):210–217. https://physionet.org/content/challenge-2019/1.0.0/
- Singer M, et al. *The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3).* JAMA 2016;315(8):801–810.
- Royal College of Physicians. *National Early Warning Score (NEWS) 2.* 2017.
- [ONNX Runtime](https://onnxruntime.ai/) · [Flask](https://flask.palletsprojects.com/) · [scikit-learn](https://scikit-learn.org/) · [Chart.js](https://www.chartjs.org/)

The model weights are derived from the PhysioNet/CinC 2019 dataset; check its terms before redistributing them.
