---
license: other
license_name: physionet-cinc-2019-derived
license_link: https://physionet.org/content/challenge-2019/1.0.0/
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - sepsis
  - datathon-2026
  - early-warning
  - clinical
  - healthcare
  - tabular
  - lightgbm
  - onnx
  - vital-signs
  - physionet
model-index:
  - name: asep-sepsis-onnx
    results:
      - task:
          type: tabular-classification
          name: Sepsis onset within 6 hours
        dataset:
          type: physionet-cinc-2019
          name: PhysioNet/CinC Challenge 2019
        metrics:
          - type: average_precision
            value: 0.0825
            name: AUPRC (held-out, base rate 0.0181)
          - type: roc_auc
            value: 0.7702
            name: AUROC (held-out)
---

# asep-sepsis-onnx

Early warning for sepsis **six hours before onset**, from **seven bedside
vitals** plus the patient's age and hours since admission.

385 KB. Runs on a CPU, in-process, with no GPU, no network call, and no Python
at inference time. It is the inference half of **asep**, a triage system for
Indonesian *puskesmas* and other primary care facilities (FKTP), and it loads
directly into a Next.js server through `onnxruntime-node`.

> Clinical decision support. It presents an estimate alongside the completeness
> of the data behind it; the clinical decision stays with the health worker.

---

## Competition context

Built by **team MyLove** for **Datathon 2026**, a Data & AI implementation
competition.

The question we set ourselves was not "how high can AUPRC go on PhysioNet" —
that problem is well studied, and a leaderboard number would not have survived
contact with a real *puskesmas*. We asked instead:

> **What does a sepsis early-warning system look like when the facility running
> it cannot measure everything?**

Most sepsis models assume a full ICU monitor. Ours assumes the opposite and is
built to keep working as sensors disappear — because that is the situation in
the facilities where early detection would change the most outcomes.

---

## What makes this model different

**It is designed around missing data, not in spite of it.** `NaN` is a
first-class input. A facility with no capnograph sends `NaN` for `EtCO2` every
hour, forever, and the model routes it down a split it learned during training.
There is no imputation step to configure and no sentinel value to get wrong.

**It clears the clinical rules by a wide margin.** qSOFA, SIRS and a partial
NEWS2 — the scores a clinician can compute at the bedside from these same
numbers — reach 1.2–1.6× lift over chance. This model reaches 4.4×.

**It degrades gracefully.** Scored against synthetic IEEE 11073 device streams
with dropouts and dead devices across five facility profiles, it still alarmed
**41 of 48** septic patients at a **median 16 hours** before onset — including
facilities down to a blood-pressure cuff and a thermometer.

**Physiology drives the score.** TreeSHAP attribution is led by temperature
(21.8 %), heart rate (15.7 %) and respiratory rate (11.8 %). Length-of-stay
influence was deliberately constrained during training so the model reasons from
the patient's condition rather than from how long they have been admitted.

**The learned risk curves are clinically recognisable** — and nobody told the
model these thresholds. Temperature is U-shaped with fever dominant, heart rate
rises past 104 bpm, respiratory rate jumps past 24, systolic pressure below
98 mmHg raises risk, and MAP below 65 stands out as the shock threshold.

**Calibration travels inside the file.** The graph ends in a monotone map that
places the utility-optimal operating point exactly on the application's
high-risk cutoff. The constants live in the `.onnx` itself, so the calibration
can never drift away from the weights it was fitted against.

**The contract is machine-verified against the application's own source.** The
export tooling parses the feature order and the risk thresholds straight out of
the TypeScript and fails if either drifts — rather than keeping a second copy
that can quietly disagree.

**It is fast.** 17,824 patient-hours score in 0.029 s — about 620,000
patient-hours per second on one CPU core.

---

## The contract

One input tensor, one output tensor.

| | |
|---|---|
| Input | `input`, `float32`, `[N, 9]` |
| Column order | `age`, `hoursSinceAdmission`, `HR`, `Temp`, `Resp`, `SBP`, `DBP`, `O2Sat`, `EtCO2` |
| Measured at the bedside | the last seven; `age` and `hoursSinceAdmission` are patient context |
| Unknown values | **`NaN`** — never `0` |
| Output | `riskScore`, `float32`, `[N]`, range 0..1 |
| Opset | ai.onnx 15, ai.onnx.ml 1 |
| Horizon | 6 hours |

Units: years, hours, bpm, °C, breaths/min, mmHg, mmHg, %, mmHg.

### MAP is computed inside the graph

The booster reads **ten** columns; the graph accepts **nine** and derives the
tenth itself:

```
input[:, SBP], input[:, DBP]  ->  DBP + (SBP - DBP)/3  ->  concat  ->  booster
```

Mean arterial pressure needs an arterial line, so the application never sends
it. Deriving it in-graph keeps the `[N, 9]` contract intact and guarantees the
value the model scores is the same one the interface displays. If SBP or DBP is
absent, MAP stays `NaN` rather than becoming a plausible-looking number.

---

## Usage

### Python

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("sepsis-model.onnx")

# age, hoursSinceAdmission, HR, Temp, Resp, SBP, DBP, O2Sat, EtCO2
# NaN = never measured. This facility has no capnograph.
row = np.array([[65, 10, 126, 38.9, 28, 88, 52, 90, np.nan]], dtype=np.float32)

score = float(session.run(None, {"input": row})[0][0])
print(score)  # 0.9618
```

### Node.js

```js
import ort from "onnxruntime-node";

const session = await ort.InferenceSession.create("sepsis-model.onnx");
const values = Float32Array.from([65, 10, 126, 38.9, 28, 88, 52, 90, NaN]);
const out = await session.run({
  input: new ort.Tensor("float32", values, [1, 9]),
});
console.log(out.riskScore.data[0]); // 0.9618
```

### Reading the score

`riskScore` is a **calibrated ranking**, tuned so the cutoffs land on
actionable review tiers:

| `riskScore` | Category | Observed sepsis within 6 h (held-out) |
|---|---|---|
| < 0.3 | RENDAH (low) | 0.68 % |
| 0.3 – 0.6 | SEDANG (medium) | 1.80 % |
| ≥ 0.6 | TINGGI (high) | 6.66 % |

The three tiers separate cleanly: a patient in the high tier is roughly **10×**
more likely to develop sepsis within six hours than one in the low tier. Read
the score as a position in the queue — "this patient is in the top 16 % by
risk" — which is what makes it useful for triage.

---

## Training data

**PhysioNet/CinC Challenge 2019**, training sets A and B — 40,276 ICU patients,
1,549,880 patient-hours, 2,929 (7.27 %) septic.

The label is the Challenge's own `SepsisLabel`: 1 from six hours **before**
clinical onset onward. No relabelling was done, which is what makes this early
*prediction* rather than detection.

### Preprocessing, matched to the application

- **LOCF.** Each vital at hour *t* is the last observed value at or before *t* —
  exactly how the application carries values forward.
- **Gating.** Only rows the application would actually score are used: age ≥ 18,
  at least 3 variables present across at least 2 physiological domains.
- **Plausibility.** Values outside the application's clinical bounds are rejected
  before the forward fill, so a broken cuff reporting 0 mmHg never becomes a
  reading.
- **float32.** Training is quantised to the precision ONNX serves at, so the
  trainer and the runtime agree on every split.

---

## Results

LightGBM, 352 trees. Splits are **by patient**, never by row — a row split would
leak a patient into both sides. Train 910k / tune 299k / test 304k hours. Every
threshold was chosen on tune; the test split was scored **once**.

| | AUPRC | AUROC |
|---|---|---|
| tune | 0.0945 | 0.7633 |
| **test (held-out)** | **0.0825** | **0.7702** |

Against the test split's 1.88 % base rate, AUPRC 0.0825 is a **4.4× lift** over
chance. Official PhysioNet normalized utility at the operating threshold:
**0.0675**.

### It beats what a clinic already has

Every row below was scored on the **same test split** from the **same nine
columns** — nothing had more information than anything else.

| Model | AUPRC | Lift | AUROC |
|---|---|---|---|
| qSOFA (vitals only) | 0.0235 | 1.2× | 0.5765 |
| SIRS (vitals only) | 0.0273 | 1.5× | 0.6147 |
| NEWS2 (partial) | 0.0310 | 1.6× | 0.6274 |
| Logistic regression | 0.0681 | 3.6× | 0.7074 |
| Logistic + missing flags | 0.0709 | 3.8× | 0.7223 |
| **This model** | **0.0825** | **4.4×** | **0.7702** |

The bedside rules a clinician already applies — qSOFA, SIRS, a partial NEWS2 —
top out at 1.6× lift on this data. **This model reaches 2.7× the AUPRC of the
best of them**, and still adds 16 % over a logistic regression given the exact
same columns.

### Holding up as equipment disappears

500 patients, five facility profiles, synthetic device streams with dropouts and
dead devices:

| Facility profile | n | septic | AUROC | sensitivity |
|---|---|---|---|---|
| Full equipment | 191 | 20 | 0.709 | 0.90 |
| Minimal | 77 | 6 | **0.838** | **1.00** |
| Pulse oximeter only | 40 | 4 | 0.625 | 0.75 |
| Medium | 156 | 15 | 0.774 | 0.80 |

41 of 48 septic patients alarmed, **median lead time 16 hours** before onset.

### Feature attribution (TreeSHAP, held-out)

| Feature | Share | | Feature | Share |
|---|---|---|---|---|
| `Temp` | **21.8 %** | | `O2Sat` | 6.7 % |
| `HR` | 15.7 % | | `EtCO2` | 6.7 % |
| `hoursSinceAdmission` | 13.8 % | | `DBP` | 5.9 % |
| `Resp` | 11.8 % | | `age` | 5.4 % |
| `SBP` | 10.1 % | | `MAP` | 2.1 % |

Vital signs account for **80.8 %** of the attribution — the model reasons from
physiology.

---

## Verification

Every exported model passes an automated contract check before it ships: graph
shape and tensor names, NaN handling on every single-variable dropout, the
in-graph MAP derivation, numerical parity against the source LightGBM booster,
monotonic separation of the risk tiers against observed sepsis rates, and
clinical direction — worsening vitals must never lower the score.

## Team

**MyLove** — Datathon 2026.

## Citing the data

Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis From Clinical
Data: The PhysioNet/Computing in Cardiology Challenge 2019.* Critical Care
Medicine 2020;48(2):210-217.

The weights are derived from that dataset; check its terms before
redistributing them.
