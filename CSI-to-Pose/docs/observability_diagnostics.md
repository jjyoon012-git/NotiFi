# CSI Motion Observability 진단

## 목적

현재 모델의 낮은 동작 진폭과 낙상 복원 성능이 decoder 용량, domain shift, 또는 CSI encoder의 motion 정보 부족 중 어디에서 시작되는지 분리해서 확인한다. 모든 수치는 `yja_holdout` protocol의 pose GT가 있는 `train 1,961 / validation 405 / yja E02 test 263` trial과 robust GraphFormer checkpoint를 사용했다. 평가는 기존과 동일하게 5-frame smoothing을 적용했다.

실행 명령:

```powershell
python -m notifi_pose.tools.diagnose_observability `
  --probe-epochs 15 --overfit-steps 200 --overfit-trials 1 10
```

원시 결과는 `work_v2/reports/observability_diagnostics.json`에 저장된다.

## 결과

### 1. 정적 평균 자세와 CSI shuffle

| 입력 | MPJPE | Dynamic MPJPE | Root error | Pose-speed ratio |
|---|---:|---:|---:|---:|
| Train mean pose | 30.59 cm | 31.61 cm | 59.97 cm | 0.000 |
| 정상 CSI | 29.57 cm | 30.94 cm | 59.23 cm | 0.313 |
| 다른 test trial의 CSI | 29.59 cm | 30.96 cm | 59.21 cm | 0.330 |

정상 CSI 대신 trial 전체를 무작위로 바꿔도 MPJPE가 `0.017 cm`, dynamic MPJPE가 `0.020 cm`만 나빠졌다. 이 차이는 의미 있는 낙상·동작 복원 차이로 보기 어렵다. 현재 checkpoint는 정적 평균 자세보다 약 `1.02 cm` 좋지만, 그 개선 대부분은 test trial에 맞는 CSI motion을 사용한 결과가 아니다.

### 2. Frozen encoder observability probe

GraphFormer의 `temporal_features`를 고정한 뒤 작은 MLP만 학습하여 body speed, moving 여부, fall phase, impact frame을 읽었다.

| 지표 | Source validation | yja E02 |
|---|---:|---:|
| Speed MAE | 0.174 m/s | 0.196 m/s |
| Speed R2 | 0.004 | -0.109 |
| Speed correlation | 0.318 | 0.141 |
| Moving F1 | 0.553 | 0.479 |
| Phase macro-F1 | 0.547 | 0.514 |
| Impact F1 | 0.148 | 0.103 |
| Impact timing MAE | 43.0 frames | 48.4 frames |

source에서도 speed의 설명력은 사실상 0이고 impact F1이 매우 낮다. yja에서는 모든 motion 지표가 추가로 하락한다. 따라서 domain shift가 존재하지만, 주 병목은 domain adaptation 이전에 encoder가 낙상 시점과 동작 크기를 보존하지 못한다는 점이다.

### 3. 1/10-trial overfit

가장 동적인 source trial만 골라 작은 GraphPoseNet을 200 step 동안 직접 외우게 했다.

| Trial 수 | MPJPE | Dynamic MPJPE | Root error | Pose-speed ratio |
|---|---:|---:|---:|---:|
| 1 | 9.35 cm | 10.52 cm | 4.40 cm | 0.534 |
| 10 | 18.66 cm | 20.15 cm | 41.63 cm | 0.172 |

1개 trial도 충분히 외우지 못했고 동작 진폭은 GT의 53%만 복원했다. 10개에서는 다시 평균 자세 쪽으로 붕괴한다. 현재 loss와 frame-wise decoder가 CSI 시계열을 자세 궤적으로 바꾸는 데 충분하지 않다는 증거다.

## 결론

현재 우선순위는 더 큰 pose decoder나 생성 모델을 붙이는 것이 아니다. CSI를 바꿔도 출력이 유지되는 상태에서는 어떤 decoder도 실제 낙상 자세를 복원할 근거가 없다.

1. **Alignment gate**: 각 trial의 CSI motion energy와 GT body speed의 lag/correlation을 계산하고, 최적 lag가 비정상적이거나 correlation이 낮은 trial을 학습에서 격리한다.
2. **Motion-first encoder**: pose 회귀 전에 CSI encoder가 speed, moving, phase, impact를 직접 예측하도록 pretrain한다. static/site 성분과 temporal difference 성분을 분리하고, 동일 action의 subject/environment positive pair로 contrastive 학습한다.
3. **CSI-dependence gate**: 정상 CSI 대비 shuffled CSI가 dynamic/impact 지표에서 명확히 악화되지 않으면 pose 학습을 통과시키지 않는다.
4. **Memorization gate**: 1-trial overfit에서 MPJPE `< 3 cm`, pose-speed ratio `0.9-1.1`을 먼저 달성한다. 이 조건 전에는 LOSO 학습을 반복하지 않는다.
5. **Sequence decoder**: gate 통과 후 velocity/acceleration trajectory를 먼저 예측하고 kinematic integration으로 pose를 복원한다. impact 구간에는 event-relative temporal query를 사용한다.
6. **Domain calibration**: 마지막 단계에서만 unlabeled calibration 구간의 CSI 통계로 adaptive normalization 또는 subject prototype을 추정한다. yja test pose/label은 calibration에 사용하지 않는다.

## 다음 실험의 합격 기준

| Gate | 합격 기준 |
|---|---|
| Alignment | GT speed와 CSI motion energy의 median correlation > 0.3, 비정상 lag trial 별도 격리 |
| Encoder source | speed R2 > 0.30, moving F1 > 0.75, impact F1 > 0.40 |
| Encoder yja | speed R2 > 0.15, moving F1 > 0.65, impact F1 > 0.25 |
| CSI shuffle | shuffled dynamic MPJPE가 정상 대비 최소 10% 악화 |
| 1-trial overfit | MPJPE < 3 cm, pose-speed ratio 0.9-1.1 |
| 최종 LOSO | MPJPE뿐 아니라 dynamic, distal, impact, speed ratio를 함께 개선 |

이 기준은 다음 모델의 절대 성능 목표가 아니라, CSI 정보를 실제로 사용하는 모델인지 확인하기 위한 최소 진단선이다.
