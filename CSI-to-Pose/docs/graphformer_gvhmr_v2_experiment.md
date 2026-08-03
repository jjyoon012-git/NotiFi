# GVHMR v2 Robust GraphFormer Experiment

## Objective

Reconstruct a time-varying SMPL-22 pose from CSI only, with enough dynamic detail to
analyze falls and with explicit empty-room calibration for unseen environments.

## Data contract

- Source development data: `ajh`, `lmh`, `mhw`, all E01/E02/E03 sites, 2,474 trials.
- Sealed target: `yja/E02`, 275 trials = 263 pose + 12 absence calibration trials.
- Excluded target sites: `yja/E01` (invalid CSI) and `yja/E03` (zero CSI).
- Pose GT: GVHMR `joints_world`, converted to pelvis-relative SMPL-22 plus pelvis root.
- Exact timestamps: 2,511 trials, including classification-only trials.
- Partial recorded timestamps: 788 lmh pose trials. Recorded times are scaled over the
  full GT frame axis; `k/30` is not used.
- Model input: `[304, 3, 114, 2]` amplitude and sanitized phase with link masks.
- Split: fixed contiguous per-site/per-class train/val/test blocks. Source counts are
  1,628 / 423 / 423; all 17 classes occur in every split.

## Model

1. Per-link and per-subcarrier normalization stores calibration statistics in the
   checkpoint.
2. A shared Conv1d encoder preserves local subcarrier structure for each CSI link.
3. Masked multi-head attention fuses TX1/TX2/TX3 without losing link identity.
4. Local dilated temporal blocks plus a 3-layer Transformer cover short impacts and
   the complete 10-second action.
5. Learned SMPL-22 joint queries are refined with normalized skeleton graph blocks.
6. The hybrid decoder blends direct joint coordinates with parent-relative tree
   coordinates. This keeps a skeletal prior without accumulating pelvis errors at
   the wrists and feet.
7. Root, frame motion, 17-class action, and 3-class risk use separate heads.

The model has 1,096,799 parameters.

## Loss

The objective combines SmoothL1 pose/root loss, bone-length consistency, action/risk
cross entropy, pose/root velocity, and frame motion regression. GT body speed raises
the weight of dynamic frames up to 4x so static frames cannot dominate a fall trial.

## Calibration protocol

`baseline=sub` removes a site-specific empty-room profile. The yja E02 profile uses
only its 12 absence trials and no pose/action labels. Links without a valid empty-room
profile bypass calibration instead of being disabled.

## Results

All distances are meters. Source test is in-domain across held-in people/sites. The
sealed result uses only yja E02 CSI as model input and a 5-frame moving average chosen
on source validation, not on yja.

| Model | Source MPJPE | Source root | yja E02 MPJPE | Dynamic MPJPE | yja root | Risk acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous timestamp-v2, yja E02 | - | - | 0.7410 | - | 0.6266 | - |
| GraphFormer hard tree | 0.2674 | 0.3349 | 0.2959 | 0.3038 | 0.5957 | 0.2167 |
| GraphFormer hybrid + smooth5 | **0.2427** | **0.3272** | **0.2908** | **0.3024** | 0.6277 | **0.3308** |

The hybrid decoder improves source pose by 2.47 cm and sealed pose by 0.50 cm over
the hard tree. Relative to the previous timestamp-v2 yja E02 result, sealed MPJPE
improves from 74.10 cm to 29.08 cm. Predictions are no longer identical static poses,
but dangerous-action speed is still under-reconstructed after smoothing.

## Source-subject LOSO results

These folds reproduce the `feature/goal1/work_v2/splits` protocol after restoring
the repaired lmh E02/E03 GT. Each model uses two source subjects for train/validation
and evaluates the same fixed 141-trial test portion of the third subject. The sealed
`yja/E02` set is never used for LOSO training, validation, or model selection.

| Held-out subject | Best epoch | Val MPJPE | Test MPJPE | Test root | Action acc | Risk acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ajh | 47 | 23.54 cm | 30.51 cm | 54.29 cm | 14.2% | 39.7% |
| lmh | 45 | 23.10 cm | 32.91 cm | 44.92 cm | 16.3% | 53.9% |
| mhw | 50 | 23.71 cm | 28.13 cm | 57.97 cm | 14.2% | 46.8% |
| Mean | - | **23.45 cm** | **30.51 cm** | **52.39 cm** | **14.9%** | **46.8%** |

The 7.06 cm gap between mean held-in validation and unseen-subject test MPJPE is the
current domain-generalization gap. Root localization and action/risk classification
shift more severely than pelvis-relative pose. These LOSO folds should therefore be
the development benchmark for domain-generalization changes; yja E02 must not be
used to choose those changes.

All lmh trials use recorded timestamp files. Of 824 trials, 36 have complete
frame-level records and 788 use partial recorded times interpolated over the full GT
frame axis. The cache label `uniform_30fps` is a legacy compatibility name for the
latter method; the implementation does not use `k/30`.

## Honest limitations

- 29.08 cm sealed MPJPE is a large improvement but not sufficient for reliable injury
  localization. Feet/ankles and bed-related actions remain the hardest cases.
- Sealed root error is 62.77 cm. Absolute room position is under-constrained without
  measured board geometry or a stronger location calibration target.
- The smoothed yja pose speed is 1.97x GT and root speed is 6.74x GT, so temporal
  jitter remains. This aggregate is inflated by nearly static actions; danger trials
  average only 0.32x GT pose speed and therefore lose important impact dynamics.
- yja class/risk accuracy is 0.114/0.331 despite strong source accuracy. The current
  representation is not yet domain invariant.
- The current checkpoint is a candidate baseline, not a finished deployment model.

## Next controlled experiments

1. Compare domain-generalization changes against the completed source-subject LOSO
   baseline above; do not tune them on yja E02.
2. Add link-masked self-supervised pretraining over all source CSI before pose labels.
3. Measure board coordinates and constrain root with geometry-aware link tokens.
4. Replace fixed smoothing with a learned velocity/acceleration decoder evaluated on
   fall impact timing and per-joint dynamic MPJPE.
5. Report three protocols separately: no calibration, empty-room calibration, and
   any labeled adaptation.
