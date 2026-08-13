# NotiFi AI v2

NotiFi AI v2 is the source-only development line for improving CSI action,
risk, and pelvis-relative 3D pose generalization to unseen people and sites.
It does not replace `NotiFi_AI_v1`; v1 remains the locked deployment baseline.

## Current stage

Stage 1 establishes and evaluates the motion-centered input contract:

- `yja/E02` is sealed in code and cannot enter training, validation, or model
  selection.
- `lmh/E02` and `lmh/E03` are excluded while their video/GT orientation is
  unresolved.
- raw I/Q or cached amplitude/phase is converted to relative spectral shape,
  robust temporal residuals, derivatives, local motion energy, and
  phase-offset-invariant differential phase.
- all three links use one shared encoder. Link identity is supplied only through
  explicit TX geometry.
- missing links and empty frames are masked without producing NaN values.

M1 source-only LOSO is complete and **not promoted**. The final M1b encoder
improved the first motion-only attempt, but did not meet the action transfer
gate. M2 will train support-conditioned calibration episodically rather than
trying to post-process M1 logits.

## Model path

```text
I/Q CSI + link mask + TX geometry
             |
             v
  PhysicsMotionFrontend
  - link-gain-invariant relative spectrum
  - robust amplitude residual
  - first/second temporal difference
  - local temporal energy
  - sanitized differential phase
             |
             v
  SharedLinkEncoder (same weights for TX1/TX2/TX3)
             |
             v
  GeometryAwareLinkFusion
             |
             v
  Temporal motion representation + ordered pyramid pooling
       |                 |
       v                 v
  17-action/3-risk    dense motion targets
```

The next stages add episodic support-query calibration and a train-only motion
prior with a CSI-conditioned body-part residual decoder.

## M1 source-only LOSO result

The protocol reserves two trials for each of eight basic calibration actions at
every source site. M1 receives none of those support trials. It is trained on
source-inner sites, selected on disjoint source-inner validation sites, and
evaluated once on the held-out source person. `yja/E02` is never opened.

| Run | Action macro-F1 | Risk macro-F1 | Danger recall | Safe to danger | Result |
|---|---:|---:|---:|---:|---|
| M1a, dynamic-only frontend | 5.15% | 23.86% | 64.29% | 59.73% | rejected |
| M1b, relative spectrum + ordered pooling | 7.49% | 36.77% | 36.67% | 25.97% | rejected |
| Promotion gate | >= 40% | diagnostic | diagnostic | <= 10% target | not met |

M1b shows that retaining link-relative spectral shape and event order is
necessary: risk macro-F1 increased by 12.91 percentage points and false danger
alarms fell by 33.76 points. However, held-person action transfer remains far
below the gate. The result rejects the hypothesis that source-only physical
normalization can solve the person shift without deployment support.

Full compact metrics are stored in
`results/m1_source_loso_seed24017_summary.json`. Failed M1 checkpoints are not
published or used to initialize later models.

## Data policy

Source development data:

| Subject | Environments |
|---|---|
| `ajh` | `E01`, `E02`, `E03` |
| `mhw` | `E01`, `E02`, `E03` |
| `lmh` | `E01` |

Sealed final evaluation data: `yja/E02`.

No metric from the sealed support or query partition may be used to choose an
architecture, threshold, epoch, augmentation, or hyperparameter. A final
calibration evaluation may use a predeclared subset as `sealed_support`; its
trials must be disjoint from `sealed_query`.

## Installation

```powershell
python -m pip install -e .
python -m unittest discover -s tests -p "test_*.py" -v
```

Quick contract check:

```powershell
python scripts/verify_stage1.py
```

M1 reproduction:

```powershell
python scripts/train_m1_source_loso.py `
  --cache-root "C:\path\to\work_v2\cache" `
  --run-dir "runs\m1b_source_loso_seed24017" `
  --epochs 36 --minimum-epochs 12 --patience 8 --seed 24017
```

## Stage plan

| Stage | Change | Promotion gate |
|---|---|---|
| M1 | physics frontend + shared link encoder | rejected at 0.0749 action macro-F1 |
| M2 | episodic meta-calibration | source LOSO action macro-F1 >= 0.45 and worst-site improvement |
| M3 | hierarchical action/risk training | danger recall >= 0.70 at safe-to-danger <= 0.10 |
| M4 | motion codec + retrieval residual | pose <= 27 cm and danger pose <= 33 cm |
| M5 | floor proximity/contact head | contact AP and danger distal both improve |
| M6 | locked sealed evaluation | open `yja/E02` once after configuration freeze |

## Repository boundary

All v2 code and reports live under this directory. Do not write v2 experiments
to `CSI-to-Pose`, `CSI-to-Pose-v2`, or `NotiFi_AI_v1`.
