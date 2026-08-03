# Robust GraphFormer experiment

## Protocols

- Protocol A: train `ajh + lmh + mhw` source roles `train + test` (2,051 trials),
  validate on their fixed `val` role (423), and test only the 263 pose trials from
  `yja/E02`. The 12 absence trials are calibration-only and have no pose loss.
- Protocol B: preserve `work_v2/splits` exactly. For each of `ajh`, `lmh`, and
  `mhw`, train/validate on the other two subjects and evaluate the held-out
  subject's fixed 141 test trials.

Each LOSO test split contains 135 pose trials and 6 absence/classification trials;
MPJPE and root metrics use only the 135 trials with pose GT.

No held-out pose GT is used for model selection, normalization fitting, motion
prior fitting, or early stopping.

## Model

The final measured model keeps the proven GraphFormer backbone and adds:

1. Per-link calibration subtraction and train-only normalization.
2. Complex gain/phase/slope augmentation, subcarrier masking, temporal jitter,
   and link dropout.
3. Class-balanced cross-domain batches and GroupDRO over subject/environment/risk.
4. Gradient-reversed domain classification and cross-domain supervised contrastive loss.
5. Fall phase, foot contact, velocity, acceleration, foot sliding, and floor losses.
6. Hybrid direct/kinematic SMPL-22 graph decoder and separate root trajectory head.
7. Validation-selected dense parameter averaging. Boolean normalization buffers are
   copied rather than averaged.

The experimental V3 frequency-token and interpretable kinematic-code model remains
in code, but was not selected because its early validation pose error was worse than
the GraphFormer backbone.

## Results

All pose/root values below use CSI-only inference and a 5-frame output smoother.

| Test domain | MPJPE | Dynamic MPJPE | Root | Raw pose speed / GT | Smooth pose speed / GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| yja E02 | 29.57 cm | 30.94 cm | 59.23 cm | 3.16 | 0.72 |
| LOSO ajh | 28.10 cm | 25.98 cm | 48.44 cm | 1.77 | 0.49 |
| LOSO lmh | 32.88 cm | 31.60 cm | 43.24 cm | 1.69 | 0.46 |
| LOSO mhw | 27.16 cm | 26.23 cm | 53.41 cm | 0.60 | 0.16 |
| LOSO mean | **29.38 cm** | **27.94 cm** | **48.36 cm** | **1.35** | **0.37** |

Compared with the previous GraphFormer LOSO mean, MPJPE improves from 30.51 cm
to 29.38 cm and root error improves from 52.39 cm to 48.36 cm. The yja E02 MPJPE
does not improve (29.08 cm to 29.57 cm), although root error improves.

## Interpretation

The domain-robust training improves average unseen-subject pose and root localization,
but it does not solve frame-coherent motion. Raw speed is too high on ajh/lmh, while
the mhw trajectory loses most motion after smoothing. Action accuracy also drops.
This model is therefore a stronger domain-generalization baseline, not yet a reliable
injury-localization model.

The next controlled experiment should replace output smoothing with a learned temporal
denoiser or diffusion/flow motion refiner and select checkpoints using dynamic joint
metrics, especially wrists, ankles, head, and impact frames. Measured board and camera
extrinsics are still required before absolute root position can generalize physically.

## Reproduce

```powershell
python -m notifi_pose.tools.run_robust_protocols --only yja_e02
python -m notifi_pose.tools.run_robust_protocols --only loso
python -m notifi_pose.tools.summarize_robust_runs
```
