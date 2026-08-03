# Impact-aware calibrated GraphFormer results

All checkpoint and residual-scale choices use validation data only. Test sets are used once for the table below.

| Protocol | MPJPE (cm) | Dynamic (cm) | Distal (cm) | Impact (cm) | Head (cm) |
|---|---:|---:|---:|---:|---:|
| yja_E02 | 29.57 -> 29.45 | 30.94 -> 30.79 | 43.86 -> 43.75 | 84.14 -> 83.84 | 38.84 -> 38.84 |
| loso_ajh | 28.10 -> 28.10 | 25.98 -> 25.98 | 42.38 -> 42.38 | 67.14 -> 67.14 | 35.68 -> 35.68 |
| loso_lmh | 32.88 -> 32.81 | 31.60 -> 31.50 | 49.00 -> 48.78 | 60.41 -> 60.14 | 43.97 -> 43.97 |
| loso_mhw | 27.16 -> 27.19 | 26.23 -> 26.23 | 40.53 -> 40.57 | 72.18 -> 72.00 | 35.48 -> 35.48 |

## LOSO mean

- MPJPE: 29.38 -> 29.36 cm
- Dynamic MPJPE: 27.94 -> 27.91 cm
- Distal MPJPE: 43.97 -> 43.91 cm
- Impact MPJPE: 66.58 -> 66.43 cm

## Rejected experiment

The coherent-displacement variant was rejected: on yja E02 its smoothed pose-speed ratio changed from 0.721 to 0.714 while MPJPE changed from 29.57 to 29.48 cm.

The calibrated impact model is a conservative spatial improvement. It does not solve the remaining low-frequency motion-amplitude collapse.
