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

## 현재 seen 결과

| Metric | 기존 seen 기준선 | 최종 seen | 변화 |
|---|---:|---:|---:|
| MPJPE | 24.17cm | **21.68cm** | -2.49cm (-10.3%) |
| Dynamic MPJPE | 23.39cm | **21.22cm** | -2.17cm (-9.3%) |
| Distal MPJPE | 35.44cm | **32.16cm** | -3.28cm (-9.3%) |
| Impact MPJPE | 58.24cm | **55.27cm** | -2.97cm (-5.1%) |
| Root error | 33.06cm | **32.36cm** | -0.70cm (-2.1%) |
| Pose-speed ratio | 1.058 | 1.141 | 정상 범위 유지 |

현재 seen gate는 완전히 통과하지 않았다. 다음 seen 우선순위는 root trajectory와 impact
절대위치이며, 목표는 MPJPE 20cm 이하, impact 50cm 이하, root 25cm 이하,
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
```

