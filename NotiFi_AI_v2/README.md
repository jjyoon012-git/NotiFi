# NotiFi AI v2

NotiFi AI v2 is the source-only development line for improving CSI action,
risk, and pelvis-relative 3D pose generalization to unseen people and sites.
It does not replace `NotiFi_AI_v1`; v1 remains the locked deployment baseline.

## Current stage

Stage 1 establishes the motion-centered input and evaluation contract:

- `yja/E02` is sealed in code and cannot enter training, validation, or model
  selection.
- `lmh/E02` and `lmh/E03` are excluded while their video/GT orientation is
  unresolved.
- raw I/Q is converted to robust log-amplitude residuals, temporal derivatives,
  local motion energy, and phase-offset-invariant differential phase.
- all three links use one shared encoder. Link identity is supplied only through
  explicit TX geometry.
- missing links and empty frames are masked without producing NaN values.

This stage is an implementation milestone, not a new performance claim. The
first promotion decision will be made by source-only LOSO evaluation.

## Model path

```text
I/Q CSI + link mask + TX geometry
             |
             v
  PhysicsMotionFrontend
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
  Temporal motion representation
       |                 |
       v                 v
  17-action/3-risk    dense motion targets
```

The next stages add episodic support-query calibration and a train-only motion
prior with a CSI-conditioned body-part residual decoder.

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

## Stage plan

| Stage | Change | Promotion gate |
|---|---|---|
| M1 | physics frontend + shared link encoder | source LOSO action macro-F1 >= 0.40 |
| M2 | episodic meta-calibration | source LOSO action macro-F1 >= 0.45 and worst-site improvement |
| M3 | hierarchical action/risk training | danger recall >= 0.70 at safe-to-danger <= 0.10 |
| M4 | motion codec + retrieval residual | pose <= 27 cm and danger pose <= 33 cm |
| M5 | floor proximity/contact head | contact AP and danger distal both improve |
| M6 | locked sealed evaluation | open `yja/E02` once after configuration freeze |

## Repository boundary

All v2 code and reports live under this directory. Do not write v2 experiments
to `CSI-to-Pose`, `CSI-to-Pose-v2`, or `NotiFi_AI_v1`.
