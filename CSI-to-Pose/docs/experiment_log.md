# Seen-first experiment log

모든 시각은 KST(UTC+9)이다. 결과는 별도 표기가 없으면 pose task만 포함하고,
5-frame smoothing을 적용한 CSI-only 추론 결과다. 모델 선택과 residual scale 선택에는
validation만 사용하며 test는 최종 확인에만 사용한다.

## 현재 데이터 protocol

사용자가 지정한 `feature/goal1/work_v2/splits/experiments.json`의
`single_split`을 그대로 사용한다.

| Split | Pose trials | Subjects | Environments |
|---|---:|---|---|
| train | 1,556 | ajh, lmh, mhw | E01, E02, E03 |
| validation | 405 | ajh, lmh, mhw | E01, E02, E03 |
| test | 405 | ajh, lmh, mhw | E01, E02, E03 |

trial ID의 train/validation/test 교집합은 0이다. 같은 사람과 같은 환경이 모든 split에
포함되고 trial만 분리되므로, 현재 결과는 **seen-subject + seen-environment + unseen-trial**
성능이다. yja와 LOSO는 이 단계의 학습, validation, test에 사용하지 않는다.

## 누적 요약

| 번호 | 완료 시각 | 목적 | 핵심 결과 | 결정 |
|---|---|---|---|---|
| EXP-001 | 2026-08-03 20:28 | 기존 모델이 CSI 동작을 실제로 쓰는지 진단 | 정상/셔플 CSI 차이 0.02cm, source speed R2 0.004 | 기존 encoder 병목 확인 |
| EXP-002 | 2026-08-03 22:13 | CSI motion과 GT 정렬/관측성 감사 | 2,629개 중 aligned 639, lag 후보 1,219, low 771 | 자동 lag 보정 금지, 품질 지표로만 사용 |
| EXP-003 | 2026-08-03 22:30 | motion-first encoder가 CSI 동작을 학습하는지 확인 | source speed R2 0.475, yja 0.146, shuffled yja -0.382 | encoder 채택, unseen 실험은 보류 |
| EXP-004 | 2026-08-03 22:43 | decoder의 1-trial memorization 상한 확인 | keyframe MPJPE 2.87cm, speed ratio 0.572 | 위치 gate 통과, 동작 진폭 gate 실패 |
| EXP-005 | 2026-08-03 22:47 | 제공된 seen split 기준선 확정 | test MPJPE 24.17cm, dynamic 23.39cm, impact 58.24cm | seen 기준 모델 |
| EXP-006 | 2026-08-03 22:58 | motion-first를 seen split에서 재학습 | test speed R2 0.438, shuffled -0.409 | CSI 의존성 통과 |
| EXP-007 | 2026-08-03 23:16 | motion-first keyframe pose decoder 전체 학습 | test MPJPE 25.84cm | 기준선보다 악화, 폐기 |
| EXP-008 | 2026-08-03 23:21 | zero-init 일반 motion residual | validation이 epoch 0보다 계속 악화 | residual 전체 폐기 |
| EXP-009 | 2026-08-03 23:33 | action-conditioned pose residual | test MPJPE 19.94cm, speed ratio 1.64 | 위치 개선, 과도한 움직임으로 보정 필요 |
| EXP-010 | 2026-08-03 23:40 | validation-only residual scale 보정 | scale 0.5 선택, test MPJPE 21.68cm, speed ratio 1.14 | 채택 |
| EXP-011 | 2026-08-03 23:53 | 저주파 keyframe root residual | root 33.06→32.36cm, impact 57.04→55.27cm | 최종 seen 조합에 채택 |
| EXP-012 | 2026-08-04 10:50 | 개선안 1-7 통합 학습 | test MPJPE 18.11cm, speed ratio 2.088 | 위치는 개선, 물리 속도 위반으로 무보정 모델 거절 |
| EXP-013 | 2026-08-04 11:06 | V2 branch별 validation calibration | rotation 0.10/high 0/root 0.50 선택, MPJPE 21.29cm, speed 1.167 | V2 calibrated 모델 채택 |
| EXP-014 | 2026-08-04 11:59 | contact-guided root Stage A | strength 0.50 선택, test root 31.81cm, impact 54.89cm | root 개선으로 채택, impact는 다음 단계에서 재개선 |
| EXP-015 | 2026-08-04 13:21 | event-centric impact/contact Stage | contact 0.75만 선택, test injury F1 0.354→0.423 | contact branch 채택, event/joint/speed branch 거절 |
| EXP-016 | 2026-08-04 14:18 | 충돌 휴리스틱 없는 전체 낙상 궤적 9A | MPJPE 20.60cm, danger 51.15cm, speed 1.217 | 위치 개선, 속도 gate 초과로 단독 채택 보류 |
| EXP-017 | 2026-08-04 14:34 | 구간 순서를 보존하는 bounded alignment 9B | pose strength 0 선택, MPJPE 21.29cm | 정렬 branch 기각 |
| EXP-018 | 2026-08-04 14:56 | GT-only temporal denoising prior 9C | MPJPE 20.68cm, danger 51.14cm, speed 1.163 | 현재 권장 seen 모델로 채택 |

## 상세 로그

### EXP-001: CSI observability diagnostic

- 목적: 기존 robust GraphFormer가 trial마다 다른 CSI를 pose 생성에 사용하는지 확인한다.
- 방법: 정상 CSI, trial-shuffled CSI, mean-pose baseline, frozen encoder motion probe,
  1/10 trial overfit을 비교했다.
- 결과: 정상 CSI MPJPE 29.57cm와 shuffled CSI 29.59cm의 차이가 0.02cm뿐이었다.
  frozen encoder의 source speed R2는 0.004, yja E02는 -0.109였다.
- 판단: 당시 모델은 CSI의 동작 정보를 거의 사용하지 않았고 평균 pose에 가까운 출력을 냈다.

### EXP-002: motion alignment audit

- 목적: encoder 문제와 timestamp/관측성 문제를 분리한다.
- 방법: 각 trial의 CSI amplitude/phase temporal energy와 GT body speed의 zero-lag 및
  best-lag correlation을 계산했다.
- 결과: 전체 2,629개 중 aligned-observable 639, lag-candidate 1,219,
  low-observability 771이었다. source train의 median zero-lag correlation은 0.340,
  yja test는 0.119였다.
- 판단: 정적인 safe trial의 상관 peak는 가짜 lag를 만들 수 있으므로 best lag를 timestamp
  보정값으로 자동 적용하지 않는다. 데이터 삭제나 시간 이동도 수행하지 않았다.

### EXP-003: motion-first CSI pretraining

- 목적: pose를 바로 회귀하기 전에 CSI에서 speed, moving, phase, impact, action, risk를
  먼저 복원한다.
- 구조: raw CSI branch와 temporal-difference CSI branch를 분리 인코딩한 뒤 link attention,
  temporal Transformer, multi-task heads를 적용했다.
- 결과: source validation speed R2 0.475, yja E02 0.146이었다. yja CSI를 섞으면
  R2가 -0.382로 하락했다.
- 판단: 새 encoder는 trial-specific CSI motion을 실제로 사용한다. 사용자의 결정에 따라
  이후 학습은 seen `single_split`으로 전환했다.

### EXP-004: decoder overfit ladder

- 목적: 데이터 일반화 전에 decoder가 단일 trial을 정확히 외울 수 있는지 확인한다.
- 결과: direct/velocity decoder는 MPJPE 4.69~5.72cm와 고주파 jitter 또는 motion collapse를
  보였다. keyframe trajectory decoder는 raw 2.73cm, smoothed 2.87cm까지 내려갔지만
  smoothed speed ratio는 0.572였다.
- 판단: 위치 표현력은 충분하지만 coherent motion amplitude가 아직 작다.

### EXP-005: provided single_split baseline

- 목적: unseen 적응 전에 개선해야 할 공식 seen 기준선을 확정한다.
- checkpoint: `graphformer_hybrid_dynamic_v1/best_model.pt`
- test: MPJPE 24.17cm, dynamic 23.39cm, distal 35.44cm, head 31.55cm,
  impact 58.24cm, root 33.06cm, pose-speed ratio 1.058, action 87.65%, risk 95.31%.
- 판단: 움직임 크기는 있으나 관절 방향/배치와 절대 root 위치가 부정확하다.

### EXP-006: seen motion-first encoder

- 목적: 같은 protocol에서 CSI motion 표현을 먼저 안정화한다.
- 결과: seen test speed R2 0.438, correlation 0.675, moving F1 0.775,
  impact F1 0.319였다. shuffled test는 speed R2 -0.409, impact F1 0.093이었다.
- 판단: train/test leakage 없이 CSI 의존성이 확인되어 frozen motion backbone으로 채택했다.

### EXP-007: full keyframe pose decoder

- 목적: motion-first backbone에서 pose를 end-to-end 복원한다.
- 결과: seen test MPJPE 25.84cm, dynamic 24.21cm, root 39.73cm로 기준선보다 나빴다.
- 판단: impact는 소폭 좋아졌지만 전체 위치와 root가 악화되어 폐기했다.

### EXP-008: generic motion residual

- 목적: frozen baseline에 motion feature residual만 추가한다.
- 결과: epoch 1부터 validation이 identity 상태인 epoch 0보다 나빠졌고 early stopping됐다.
- 판단: action 조건 없는 residual은 최적화 방향이 불안정하므로 폐기했다.

### EXP-009: action-conditioned metric residual

- 목적: predicted action 확률과 motion feature를 결합하고, distal/impact 관절에 직접
  metric distance loss를 적용한다.
- 결과: test MPJPE 19.94cm, dynamic 19.89cm, distal 29.73cm로 크게 개선됐지만
  pose-speed ratio가 1.64로 증가했다.
- 판단: 공간 pose는 좋아졌지만 프레임별 residual이 고주파 움직임도 키웠다. 그대로는 폐기하고
  validation scale calibration 대상으로 유지했다.

### EXP-010: validation-only residual calibration

- 목적: 위치 개선을 보존하면서 동작 진폭을 정상화한다.
- 방법: scale 0/0.25/0.5/0.75/1.0을 validation에서만 비교했다. 점수는 MPJPE,
  dynamic, distal, impact와 `abs(log(pose-speed-ratio))`를 결합했다.
- 결과: validation이 scale 0.5를 선택했다. test MPJPE 21.68cm, dynamic 21.22cm,
  distal 32.16cm, impact 57.04cm, pose-speed ratio 1.141이었다.
- 판단: 기준선 대비 위치 오차를 줄이고 과도한 움직임을 제거해 채택했다.

### EXP-011: keyframe root residual

- 목적: 상대 관절 pose는 고정한 채 pelvis의 절대 이동 경로와 낙상 절대위치를 개선한다.
- 방법: action-conditioned fused feature를 4-frame keyframe으로 pooling하고 root residual만
  학습했다. checkpoint는 validation root/impact 조합으로 선택했다.
- 결과: best epoch 19. test root 32.36cm, impact 55.27cm였다. pose MPJPE 21.68cm와
  speed ratio 1.141은 그대로 유지됐다. shuffled CSI에서는 MPJPE 31.25cm,
  root 57.67cm로 악화돼 CSI 의존성도 유지됐다.
- 판단: 개선 폭은 작지만 pose 회귀 없이 root와 impact가 함께 개선되어 최종 seen 조합에 채택했다.

### EXP-012: seven-part seen reconstruction V2

- 목적: 품질 가중, root 속도 표현, phase 조건, 6D rotation decoder, low/high 분리,
  injury head, partial fine-tuning을 하나의 identity-initialized 모델에 통합한다.
- 방법: timestamp/link/observability 품질 점수와 class-balanced sampler를 사용했다. 기존
  calibrated pose/root cascade와 motion-first encoder를 먼저 동결해 12 epoch 학습한 뒤,
  양쪽 마지막 temporal block만 0.1배 learning rate로 6 epoch 미세조정했다.
- 결과: validation MPJPE 18.14cm, test MPJPE 18.11cm, distal 26.56cm까지 내려갔다.
  injury-contact F1 0.354, feet-contact F1 0.708, floor-height MAE 2.72cm였다.
  그러나 test pose-speed ratio가 2.088이었다.
- 판단: 위치 정확도만 보면 최고지만 GT보다 두 배 빠른 자세 변화라 공식 모델로 채택하지
  않는다. component 진단에서 rotation-only speed ratio가 1.971로 주원인임을 확인했다.

### EXP-013: branch-wise validation calibration

- 목적: V2 위치 개선 중 물리적으로 허용되는 부분만 사용한다.
- 방법: validation에서 rotation strength 10개, high-pose strength 3개를 비교했다.
  pose-speed ratio 0.8~1.2를 하드 게이트로 사용하고, 이후 root strength 0/0.5/1을
  root+impact 점수로 선택했다. test는 선택에 사용하지 않았다.
- 결과: rotation 0.10, high-pose 0, root 0.50이 선택됐다. test MPJPE 21.29cm,
  dynamic 20.90cm, distal 31.53cm, impact 54.72cm, root 32.33cm,
  pose-speed ratio 1.167이다.
- 판단: 이전 최종 seen 모델보다 모든 위치 지표가 소폭 개선되고 speed gate도 통과해
  현재 권장 seen 모델로 채택한다. root 개선은 0.03cm에 불과해 여전히 별도 병목이다.

### EXP-014: contact-guided root Stage A

- 목적: V2의 상대 pose와 동작 진폭을 그대로 보존하면서 절대 pelvis trajectory만 개선한다.
- 구조: calibrated V2를 동결하고 CSI temporal feature, V2 root velocity, 예측 foot contact,
  fall phase, impact, 상대 foot speed를 dilated temporal block에 입력한다. 새 branch는 root
  anchor와 velocity residual을 예측하며, 발이 지면을 지지한다고 예측한 구간에서는 상대 발
  속도와 일관되는 support velocity를 혼합한다.
- 손실: quality-weighted root position/velocity/5-frame displacement/anchor와 foot-contact BCE,
  foot slip, contact height, floor penetration을 사용했다. GT contact와 floor는 학습 loss에만
  사용하고 validation/test 추론 입력은 CSI뿐이다.
- 선택: epoch 8 checkpoint에서 validation root+impact+MPJPE 점수가 가장 낮았다. test를
  열기 전에 validation이 root strength `0.50`을 선택했다. strength `0`은 V2 출력을 정확히
  보존하는 identity 후보로 함께 비교했다.
- 결과: test root error `32.33 -> 31.81cm`, total evaluation loss
  `0.2612 -> 0.2549`로 개선됐다. pose branch를 고정했으므로 MPJPE `21.29cm`, dynamic
  `20.90cm`, distal `31.53cm`, pose-speed ratio `1.167`은 동일하다. impact MPJPE는
  `54.72 -> 54.89cm`, foot-contact F1은 `0.708 -> 0.701`로 소폭 악화됐다. shuffled CSI는
  root `58.54cm`로 악화돼 trial-specific CSI 의존성은 유지됐다.
- 판단: 사전 정의한 root 중심 validation 선택과 test root 개선에 따라 7안 Stage A로
  채택한다. 다만 낙상 impact 및 contact 성능 개선으로 해석하지 않으며, 다음 단계는 pose를
  건드리지 않고 event-level impact/contact localization을 별도로 개선한다.

### EXP-015: event-centric impact and body-part localization

- 목적: 전체 pose loss에 묻히던 낙상 순간을 `(frame, injury joint)` event로 분리하고,
  최초 충돌 시점과 부위를 CSI-only로 복원한다.
- target 변경: 기존 `height < 12cm`만 사용하지 않고 표면 근접, 하강 속도, 관절 감속,
  가속도, 낙상 진행도를 결합한 physical impact proxy를 만들었다. 관절별 시간 정규화를
  적용해 train 270 danger trial에서 pelvis/hip/knee/head/wrist 8개 관절이 모두 사용됐다.
- 구조: frozen 7안의 V2/V3 temporal feature, raw motion feature, baseline feature와 raw CSI
  amplitude/phase 변화량의 1/3/7/15-frame 표현을 결합했다. event frame, joint-time,
  4개 body region, legacy contact, impact speed head를 학습했다.
- 정렬 감사: danger 360개 train/validation trial에서 CSI 최대 motion peak와 GT event의
  중앙 오차는 train 30.0, validation 24.5프레임이었다. 상위 5% CSI motion 후보와 event의
  최소 거리는 각각 8.0, 5.5프레임이었다. validation best-correlation의 중앙 절대 lag는
  13.5프레임으로, 충돌 후보는 존재하지만 다른 큰 동작에 묻히고 trial별 lag도 일정하지 않았다.
- 실패 실험: exact joint-time head는 validation timing 29.24→25.94프레임으로 개선했으나
  joint accuracy가 16.7→13.3%로 악화됐다. hierarchical region 및 raw CSI 반복도 timing은
  25.14프레임까지 개선했지만 test region 23.3%, exact joint 13.3%로 일반화하지 못했다.
- branch calibration: validation에서 event/joint/speed strength는 모두 `0`, contact만
  `0.75`가 선택됐다. test injury-contact F1은 `0.354→0.423`, first-contact accuracy는
  `37.8%`로 유지됐다. impact-speed MAE `0.553m/s`와 pose/root 출력도 그대로다.
- 판단: 8안은 **contact-calibrated Stage A만 채택**한다. event timing과 최초 충돌 부위는
  개선됐다고 주장하지 않는다. 다음 단계는 영상을 이용해 최소한 danger trial의 실제 impact
  frame/body region annotation을 만들거나, 현재 proxy의 표본 검증을 먼저 해야 한다.

### EXP-016: full-sequence fall trajectory without impact heuristic

- 목적: 어디가 먼저 충돌했는지를 맞히는 대신 사람이 어떤 자세와 경로로 넘어지는지를
  전체 sequence로 복원한다.
- 구조: frozen 7안 base의 CSI/pose/root feature에 raw CSI multi-scale motion, body-group
  speed, risk probability를 결합하고 dilation 1/2/4/8 temporal block과 Transformer를
  통과시켰다. 6D rotation residual은 bone length를 보존하고 root는 anchor/step residual로
  분리했다.
- 손실: frame pose/root, 5-frame displacement, root drop, torso/shoulder orientation,
  endpoint만 사용했다. 최대 가속도 frame, 최초 충돌 joint, impact score는 사용하지 않았다.
- 선택: epoch 10에서 validation이 pose strength 0.15, root strength 0.5를 선택했다.
- 결과: test MPJPE 20.60cm, dynamic 20.40cm, root 31.61cm, danger 51.15cm,
  danger distal 55.72cm, danger endpoint 69.72cm였다. speed ratio는 1.217이었다.
- 판단: 7안보다 위치는 좋아졌지만 움직임을 21.7% 과장해 1.2 speed gate를 넘었다.
  trajectory branch는 9C의 source로만 사용하고 9A 단독 모델은 최종 채택하지 않는다.

### EXP-017: bounded piecewise temporal alignment

- 목적: timestamp를 버리거나 GT를 이동하지 않고, trial 내부의 작은 비선형 시차를 sequence
  문맥으로 흡수한다.
- 방법: 8개 연속 구간마다 ±15-frame offset 후보를 만들고 offset 크기/변화에 벌점을 둔
  dynamic programming 경로를 trajectory descriptor loss에 추가했다. frame-aligned loss는
  그대로 유지했다.
- 결과: validation calibration이 pose strength 0, root strength 0.5를 선택했다. test
  MPJPE 21.29cm, danger 51.95cm, danger endpoint 71.23cm로 9A보다 나빴다.
- 판단: 현재 CSI/GT descriptor로 선택한 구간 offset은 학습 신호로 신뢰할 수 없다.
  9B를 기각하고 alignment weight 0을 공식 설정으로 사용한다. timestamp와 원본 GT는
  변경하지 않았다.

### EXP-018: temporal denoising motion prior

- 목적: 9A의 과도한 frame-to-frame 움직임을 줄이되 낙상 방향과 전체 경로를 보존한다.
- 학습 데이터: `single_split` train GT만 사용했다. 외부 UP-Fall 33-joint 데이터는
  MediaPipe camera 좌표라 GVHMR/SMPL-22 metric 좌표와 직접 섞지 않았다.
- 방법: Gaussian noise와 frame/joint masking으로 GT trajectory를 오염시킨 뒤 temporal
  convolution과 Transformer가 원본을 복구하도록 학습했다. impact/contact label은 없다.
- prior 검증: noisy validation source MPJPE 4.55cm를 3.70cm로 낮췄고, clean GT에 대한
  distortion은 1.74cm였다. validation calibration이 prior strength 1.0을 선택했다.
- 결과: 9A 대비 test speed ratio가 1.217→1.163으로 개선됐다. MPJPE는
  20.60→20.68cm로 0.07cm 악화됐지만 danger MPJPE 51.15→51.14cm, danger distal
  55.72→55.64cm, endpoint 69.72→69.66cm로 소폭 개선됐다. shuffled CSI MPJPE는
  31.72cm로 trial-specific CSI 의존성을 유지했다.
- 판단: 9C를 현재 권장 seen 모델로 채택한다. 속도 안정화에는 성공했지만 danger
  absolute pose 51.14cm는 여전히 크므로 낙상 복원 문제가 해결됐다고 보지 않는다.

## 현재 seen 결과

| Metric | 기존 기준선 | 7안 Stage A | 8안 contact | 9A trajectory | 9B alignment | 9C prior |
|---|---:|---:|---:|---:|---:|---:|
| MPJPE | 24.17cm | 21.29cm | 21.29cm | **20.60cm** | 21.29cm | 20.68cm |
| Dynamic MPJPE | 23.39cm | 20.90cm | 20.90cm | **20.40cm** | 20.90cm | 20.41cm |
| Root error | 33.06cm | 31.81cm | 31.81cm | **31.61cm** | 31.94cm | **31.61cm** |
| Danger MPJPE | - | - | - | 51.15cm | 51.95cm | **51.14cm** |
| Danger distal | - | - | - | 55.72cm | 56.93cm | **55.64cm** |
| Danger endpoint | - | - | - | 69.72cm | 71.23cm | **69.66cm** |
| Pose-speed ratio | 1.058 | 1.167 | 1.167 | 1.217 | 1.167 | **1.163** |
| Injury-contact F1 | - | 0.354 | **0.423** | - | - | - |

현재 seen gate는 완전히 통과하지 않았다. 다음 seen 우선순위는 danger 전체 trajectory와
endpoint 절대위치이며, 목표는 MPJPE 20cm 이하, danger MPJPE 45cm 이하, root 25cm 이하,
pose-speed ratio 0.8~1.2다. 이 gate에 가까워진 뒤 동일 구조에 calibration/domain adaptation을
붙여 LOSO와 yja E02 unseen 평가를 재개한다.

## 재현 명령

```powershell
python -m notifi_pose.tools.audit_motion_alignment
python -m notifi_pose.tools.train_motion_first --exp single_split `
  --epochs 12 --patience 4 --batch-size 12 `
  --run-dir work_v2/runs/motion_first_seen
python -m notifi_pose.tools.train_seen_action_residual `
  --epochs 20 --patience 6 --batch-size 12
python -m notifi_pose.tools.calibrate_seen_action_residual
python -m notifi_pose.tools.train_seen_root_residual `
  --epochs 20 --patience 6 --batch-size 12
python -m notifi_pose.tools.train_seen_v2 `
  --head-epochs 12 --finetune-epochs 6 --patience 4 --batch-size 8
python -m notifi_pose.tools.diagnose_seen_v2_components --dataset val
python -m notifi_pose.tools.calibrate_seen_v2
python -m notifi_pose.tools.train_seen_v3_root `
  --epochs 16 --patience 5 --batch-size 8 `
  --run-dir work_v2/runs/seen_v3_contact_root
python -m notifi_pose.tools.audit_impact_targets --split val
python -m notifi_pose.tools.audit_impact_alignment
python -m notifi_pose.tools.train_impact_event `
  --epochs 15 --patience 5 --batch-size 8 `
  --run-dir work_v2/runs/impact_event_v8c_raw
python -m notifi_pose.tools.calibrate_impact_event
python -m notifi_pose.tools.train_seen_v4_trajectory `
  --epochs 12 --batch-size 8 --alignment-weight 0 `
  --run-dir work_v2/runs/seen_v4_v9a_no_impact
python -m notifi_pose.tools.train_seen_v4_trajectory `
  --epochs 12 --batch-size 8 --alignment-weight 0.15 `
  --run-dir work_v2/runs/seen_v4_v9b_bounded_alignment
python -m notifi_pose.tools.train_motion_prior_v9 `
  --epochs 10 --batch-size 12 `
  --run-dir work_v2/priors/temporal_denoiser_v9
python -m notifi_pose.tools.calibrate_motion_prior_v9 `
  --trajectory-run work_v2/runs/seen_v4_v9a_no_impact `
  --prior-run work_v2/priors/temporal_denoiser_v9 `
  --run-dir work_v2/runs/seen_v4_v9c_motion_prior
```
