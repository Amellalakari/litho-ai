# AI for Photolithography Process Development — Phases 1 & 2

A physics-based lithography simulator, ML surrogate models trained on it, and Bayesian
optimisation for process-window centring, with a browser dashboard that runs the model.

Target layer: **90 nm lines / 180 nm pitch, dry ArF (193 nm), NA 0.93, σ 0.75 (k₁ ≈ 0.43)**,
spec CD = 90 nm ± 10 %.

> All data is simulated. The simulator is built from textbook physics but is not calibrated to a
> real resist or scanner. Absolute numbers are illustrative; the workflow and trends are the point.

---

## 1. Simulator (`litho_ai/simulator.py`)

| Step | Model |
|---|---|
| Mask | Binary chrome line on a periodic grating, Fourier-series transmission |
| Illumination | Conventional disk source, Abbe summation over source points |
| Projection | Pupil cut-off NA/λ, exact defocus phase, small spherical aberration (Z9, 0.02 λ) |
| Resist film | Aerial image averaged through the film thickness; swing curve on coupled dose |
| Exposure | Photo-acid generation `a = 1 − exp(−C·E·I)` |
| PEB | Gaussian acid diffusion, L = √(2Dt) with Arrhenius D(T); base quencher; Arrhenius deprotection |
| Development | Threshold on deprotection → line CD; failure modes: line lost, space bridged |
| Metrology | CD, LER proxy (edge gradient + shot noise) |
| Noise | Field-to-field dose (0.5 %) and focus (6 nm) jitter, CD-SEM noise (0.7 nm) |

Behaviour it reproduces: Bossung curves with an isofocal dose (~31.5 mJ/cm²), smile/frown
inversion across it, pitch-dependent best-focus shift, swing-curve dose-to-size shifts, and
contrast loss at high PEB temperature.

## 2. Dataset (`generate_dataset.py`)

8,000 exposure fields, Latin-hypercube over six knobs:

| Knob | Range |
|---|---|
| Dose | 18–55 mJ/cm² |
| Focus | −180 to +180 nm |
| Pitch | 180–400 nm |
| Resist thickness | 100–160 nm |
| PEB temperature | 100–130 °C |
| PEB time | 45–90 s |

Each row stores the noisy measured CD (what a fab sees) and the noise-free true CD (only
possible in simulation), so model error can be compared with the measurement noise floor.
About half the fields fail to print, which is typical of a wide exploratory DOE.

## 3. Surrogate models (`train_surrogate.py`)

Held-out test set, 1,600 fields:

| Model | Result |
|---|---|
| Print / fail classifier (XGBoost) | 95.6 % accuracy, ROC-AUC 0.994 |
| Print / fail classifier (MLP) | 95.2 % accuracy, ROC-AUC 0.989 |
| CD, MLP 3×64 (all printed fields) | 3.9 nm RMSE vs true CD |
| CD, MLP, fields with CD 70–110 nm | **2.6 nm RMSE** (measurement noise floor there: 1.9 nm) |
| CD, XGBoost (tuned) | 6.2 nm RMSE vs true CD, 4.2 nm in 70–110 nm |
| LER proxy, MLP (log space) | R² 0.71 |
| Speed | MLP ≈ **2,000×** faster than the simulator; XGBoost ≈ 140× |

Findings worth discussing in an interview:
- The MLP beats XGBoost on this problem. CD is a smooth function of six continuous knobs, and
  tree ensembles approximate it with steps.
- Trained on noisy CD, the MLP's error against the *true* CD (3.9 nm) is lower than the raw
  measurement noise (4.4 nm): it averages the noise out.
- Dose and PEB temperature dominate CD sensitivity (XGBoost feature importance).
- The surrogate reproduces the process window closely (DOF at 5 % EL: 280 nm vs 290 nm true;
  max EL 16 % vs 17.5 %) but smooths the sharp window pinch near the failure boundary.

## 4. Process window and recipe centring (`process_window.py`, `bo_centring.py`)

**Metric.** Every method recommends a recipe, and the recipe is scored on the noise-free
simulator by its *centred* DOF at 5 % exposure latitude: 2 × the focus margin to the nearest
window edge, with CD in spec at dose × (1 ± 2.5 %). An off-centre recipe scores lower even if it
sits inside the window.

**Methods compared** (12 repeat runs each, all with scanner and metrology noise):
1. **Full FEM + Bossung fit** (classical): dose × focus matrix, polynomial Bossung fit per dose,
   centre on the fitted surface.
2. **Naive BO**: Gaussian process on one scalar score per experiment (RMS CD error over three
   fields at focus −100, 0, +100 nm).
3. **Model-based BO** (this project's method): a GP models CD itself from every field, a second
   GP models print probability, the window is computed from the model, and the next experiment is
   chosen by a lower-confidence-bound rule.

**Dose × focus (2 knobs).** True optimum: 304 nm at 31.5 mJ/cm², focus 0.

| Fields used | Model-based BO | Naive BO | Full FEM |
|---|---|---|---|
| ~20 | 248 nm | ~0 nm | 252 nm (4×5) |
| ~30–35 | **266 nm** | 12 nm | 218 nm (5×7) |
| ~57–63 | **274 nm** | 236 nm | 224 nm (7×9) |
| 99 | — | — | 234 nm (9×11) |
| 143 | — | — | 232 nm (11×13) |

Model-based BO reaches ~88–90 % of the optimum with 30–57 fields. The FEM stays at 72–83 %
all the way to 143 fields, because more grid points do not fix centring error from noise and fit bias.
Naive BO on a scalar score is slow: the score's noise is about as large as the signal.

**Dose, focus, PEB temperature, thickness (4 knobs).** Best recipe found: ~290–310 nm.

| Fields used | Model-based BO | Grid of FEMs |
|---|---|---|
| 120 | 250 nm | — |
| 140–150 | 255 nm | 250 nm (2×2 bake/thickness × 5×7) |
| 315 | — | 235 nm |
| 567 | — | 260 nm |
| 1,008 | — | 205 nm |

In 4-D the two approaches reach similar quality at similar budgets, about 85 % of the best
recipe. BO needs ~100 fields before the model is good enough to be useful. I have not shown a
clear BO advantage at four knobs; the 2-D result is the stronger one.

## 5. Dashboard (`docs/recipe-console.html`)

A single self-contained page. The MLP weights are embedded and inference runs in the
browser. Six knobs update predicted CD, LER, print probability, Bossung curves, the dose–focus
window and DOF @ 5 % EL live. **Centre this recipe** searches dose and focus for the widest
centred window under the current pitch, resist and bake.

## Run it

```bash
pip install -r requirements.txt
bash run_all.sh      # ~15 min on 2 CPU cores
```

## Known limitations / next steps

- Calibrate to real data: digitised Bossung or dose-to-size curves from a resist datasheet or
  SPIE paper, or a single real FEM wafer, to fit C, L, k and threshold.
- 1-D line/space only. Contacts and line ends need a 2-D image.
- LER is a proxy (edge gradient + shot noise), not a stochastic resist model.

---

# Phase 2 — CD-SEM image analysis

One U-Net reads top-down SEM images of the resist lines. It measures CD and LER, and finds
and locates bridges, breaks and particles. The baseline is a classical line-scan CD-SEM algorithm.

## 6. Synthetic SEM images (`sem.py`, `sem_dataset.py`)

Each image is 192 × 192 px at 3.0 nm/px (576 nm field). CD, pitch and LER come from Phase-1
simulator fields (CD 18–155 nm, pitch 180–280 nm, LER 3–16 nm).

| Element | Model |
|---|---|
| Line edges | Correlated Gaussian roughness (σ = LER/3, correlation length 12–35 nm), line-to-line CD variation |
| Defects | Bridge (4–40 nm tall, thin micro-bridge to full bridge), break (3–35 nm gap, rounded ends), particle (4–26 nm, irregular) |
| SEM signal | Material contrast, edge brightening from topography gradient, charging trail along the scan |
| Detector | Beam blur 1.2–3 nm, Poisson shot noise scaled by frames averaged (1–32), scan-line gain and jitter |
| Labels | Per pixel: space, line, particle, bridge, break; plus true CD and LER from the geometry |

Train 2,400 / validation 300 / test 900 images (150 at each of 1, 2, 4, 8, 16, 32 frames);
about 40 % defect-free, 20 % each defect type.

## 7. Model (`sem_unet.py`, `sem_metrology.py`)

- 4-level U-Net, 0.93 M parameters, cross-entropy + soft Dice, flip and gain augmentation, CPU training.
- **CD**: 0.5 crossing of the line probability on every scan row, sub-pixel, skipping rows that touch a
  predicted defect. **LER**: 3σ of edge residuals after removing tilt.
- **Defect call**: defect-pixel area above a threshold tuned on validation. The map also gives the location.
- **Classical baseline**: smooth 5 scan lines, find the bright edge peaks, parabolic sub-pixel fit,
  median/MAD outlier rejection. No noise correction for LER.
- Both methods get one constant CD offset calibrated on validation (U-Net −1.2 nm, classical −2.8 nm),
  as a real CD-SEM is calibrated against reference metrology.

## 8. Results (held-out test, 900 images)

| Frames | CD RMSE U-Net | CD RMSE classical | LER bias U-Net | LER bias classical | Defect calls correct |
|---|---|---|---|---|---|
| 1 | **0.43 nm** | 6.14 nm | +0.06 nm | +11.4 nm | 97.3 % |
| 2 | **0.30 nm** | 4.90 nm | −0.12 nm | +3.7 nm | 97.3 % |
| 4 | **0.21 nm** | 0.63 nm | −0.17 nm | +0.5 nm | 98.7 % |
| 8 | **0.21 nm** | 1.86 nm | −0.27 nm | +0.3 nm | 100 % |
| 16 | **0.24 nm** | 0.43 nm | −0.23 nm | −0.0 nm | 99.3 % |
| 32 | **0.23 nm** | 0.42 nm | −0.16 nm | −0.1 nm | 99.3 % |

- Defect classification: **98.7 %** correct over all 900 images. 10 of the 12 errors are false alarms on
  defect-free images (mostly narrow, rough lines called "break"); 2 are misses.
- The U-Net holds sub-half-nanometre CD accuracy even at 1–2 frames. The classical algorithm needs 4+ frames.
- Classical LER is inflated by image noise (+11 nm at 1 frame), the well-known noise bias in LER
  metrology. The U-Net's LER stays within 0.3 nm of truth at every noise level.
- Classical CD also fails on defect images (bridges merge lines), which is why its "all images" error
  is higher than its defect-free error at 2 and 8 frames.
- Line-segmentation IoU 0.984; defect-pixel IoU: particle 0.92, bridge 0.90, break 0.86.

## 9. Closing the loop with Phase 1 (`sem_eval.fem_loop`)

A 6-dose × 17-focus FEM wafer from the Phase-1 simulator; each of the 94 printed fields is imaged with a
fast 4-frame scan. Defect likelihood follows the process: bridges when lines print wide, breaks when narrow.

| | U-Net | Classical |
|---|---|---|
| CD RMSE, all fields | **1.0 nm** | 20.3 nm |
| CD RMSE, defect-free fields | **0.13 nm** | 0.22 nm |
| Defect calls correct | 99 % (93/94) | — |

The Bossung curves rebuilt from SEM images match the simulator, and the automatic defect map shows
bridges on the low-dose side and breaks on the high-dose side of the window.

## Phase 2 caveats

- Everything is synthetic. Real SEM images have more varied contrast, charging and defect shapes.
- An earlier version with larger defects scored 100 %, which was too easy; defect sizes were then
  widened down to a few nm and the model fine-tuned on the harder set (8 more epochs from the v1 weights).
  Real defects would still be harder.
- The classical baseline has no LER noise correction. PSD-based unbiased LER would narrow that gap.
- Next: domain adaptation to real SEM images (a few dozen labelled images plus the synthetic set),
  and 2-D patterns (contacts, line ends).
