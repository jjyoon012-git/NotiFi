# NotiFi AI v2 LOSO Evaluation Audit

평가일은 2026-08-14이다. 이 문서는 현재 배포 artifact와 같은 checkpoint 계보를
사용해 full-support calibration을 다시 수행한 source nested subject-LOSO 결과다.

## Evaluation Contract

- Source subject: `ajh`, `mhw`, `lmh`
- Source site: `ajh/E01-E03`, `mhw/E01-E03`, `lmh/E01`
- Outer fold: 한 사람의 모든 환경을 학습과 설정 선택에서 제외
- Calibration per target site: absence 12, basic 8종 x 2, warning 3종 x 1, danger 5종 x 1
- Query: calibration support와 겹치지 않는 총 1,042 trials
- Repetition: support seed `17017`, `17027`, `17037`, `17047`, `17057`
- Query action/risk label과 pose GT는 추론이 끝난 뒤 metric 계산에만 사용
- `yja/E02`는 학습, 설정 선택, LOSO 평가에 사용하지 않음

Checkpoint lineage는 다음과 같다.

| Encoder | Fold lineage | Deployment SHA-256 |
|---|---|---|
| Primary | `cal60_seed22012_reflect025_warp025_swa6_ft12` | `b69d8723ff20265a7525aaf383e9e73cf6439b03217dccede8b95c185451ea6e` |
| Secondary | `cal66_grl0_seed22012_swa4_ft8` | `9259d2eb816cb85125ab7d949371592250f8266f13f957653980580a50dbef74` |

## Classification Result

평균과 표준편차는 5개 support seed 기준이다. Raw와 calibrated 열은 같은
checkpoint, 같은 query를 사용한다.

| Metric | Matched raw | Full-support calibration | Change |
|---|---:|---:|---:|
| 17-action accuracy | 44.89 +/- 0.87% | **53.38 +/- 1.52%** | +8.48%p |
| 17-action macro-F1 | 35.33 +/- 0.94% | **47.55 +/- 1.83%** | +12.22%p |
| Danger subtype accuracy | 13.00 +/- 2.12% | **38.73 +/- 6.11%** | +25.73%p |
| 3-risk accuracy | 52.88 +/- 0.86% | **68.50 +/- 2.01%** | +15.62%p |
| 3-risk macro-F1 | 43.50 +/- 1.20% | **62.00 +/- 2.87%** | +18.51%p |
| Danger recall | 53.12 +/- 3.07% | **56.90 +/- 5.49%** | +3.78%p |
| Safe to danger false alarm | 18.68 +/- 0.70% | **8.21 +/- 1.85%** | -10.47%p |
| Worst-site action accuracy | 39.60 +/- 1.12% | **42.15 +/- 4.96%** | +2.55%p |

### Held-out subject

| Held-out | Query | Action Acc | Action F1 | Risk Acc | Risk F1 | Danger Recall | Danger Subtype | Safe to Danger |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ajh` | 447 | 50.02% | 44.78% | 59.69% | 50.78% | 44.53% | 41.33% | 8.77% |
| `mhw` | 447 | 53.02% | 46.53% | 75.62% | 71.86% | 65.33% | 31.73% | 10.00% |
| `lmh` | 148 | 64.59% | 58.99% | 73.65% | 66.14% | 68.80% | 52.00% | 1.07% |

`lmh`는 E01 하나만 평가하므로 세 환경을 가진 `ajh`, `mhw`보다 쉬운 조건이다.
가장 큰 분류 병목은 `ajh`의 danger recall 44.53%와 `mhw`의 danger subtype
31.73%다. 정상 calibration은 평균 성능을 높였지만 어떤 held-out 사람에서도
균일한 성능을 보장하지 않는다.

## 3D Pose Result

Motion calibration은 기본·danger support의 CSI motion signature만 사용한다.
Calibration pose GT는 사용하지 않았고, `regularization=100`, `mixture=0.5`,
`risk_sqrt` gate를 5개 seed에서 고정했다.

| Metric | Raw retrieval | Motion calibration | Change |
|---|---:|---:|---:|
| Pose MPJPE | 29.37 +/- 0.17 cm | **28.97 +/- 0.16 cm** | -0.40 cm |
| Distal MPJPE | 43.41 +/- 0.32 cm | **42.88 +/- 0.31 cm** | -0.53 cm |
| PA-MPJPE | **10.45 +/- 0.09 cm** | 10.61 +/- 0.10 cm | +0.16 cm |
| Danger pose MPJPE | 38.35 +/- 0.13 cm | **37.23 +/- 0.38 cm** | -1.13 cm |
| Danger distal MPJPE | 56.99 +/- 0.17 cm | **55.47 +/- 0.53 cm** | -1.52 cm |

### Held-out subject

| Held-out | Pose | Distal | PA-Pose | Danger Pose | Danger Distal |
|---|---:|---:|---:|---:|---:|
| `ajh` | 29.33 cm | 43.56 cm | 10.45 cm | 36.85 cm | 54.66 cm |
| `mhw` | 28.64 cm | 42.17 cm | 10.73 cm | 36.42 cm | 54.51 cm |
| `lmh` | 28.86 cm | 42.98 cm | 10.76 cm | 40.80 cm | 60.76 cm |

전체 궤적과 danger 사지 오차는 개선됐지만 PA-MPJPE는 0.16 cm 악화됐다. 즉
motion calibration은 검색된 동작의 움직임과 낙상 궤적을 조금 보정하지만, 회전과
크기를 제거한 순수 자세 형상을 개선하지는 못했다. `lmh` danger distal 60.76 cm는
바닥 접근 신체 부위나 부상 부위를 정밀하게 판단하기에 부족하다.

## Audit Correction

이전 README의 action accuracy 54.51%는 비-seed primary fold checkpoint로 계산된
값이었다. 실제 `notifi_ai_v2.pt`의 primary deployment model은 seed-22012 SHA-256과
일치하므로 현재 README 표를 53.38% 결과로 정정했다. 수정 전
`results/full_support_ridge_risk_5seed_summary.json`은 배포 artifact provenance
hash 보존을 위해 그대로 두고, 정정 결과는
`results/full_support_loso_recheck_20260814.json`에 별도로 저장했다.

Pose 5-seed 결과와 최종 봉인 `yja/E02` 결과는 처음부터 배포 artifact 계보와
일치했으므로 변경하지 않았다.
