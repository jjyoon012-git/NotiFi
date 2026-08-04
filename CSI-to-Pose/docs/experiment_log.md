# Seen-first experiment log

## 최신 기록: V12 clean-protocol multi-expert

아래 실험은 모두 `single_split_lmh_e01`의 validation만으로 선택했다. 최종 test는
EXP-075에서 한 번만 열었으며, 그 결과를 보고 추가로 모델이나 calibration을 변경하지 않았다.

| 번호 | 완료 시각(KST) | 상세 내용 | 목적 | 결과 | 결정 |
|---|---|---|---|---|---|
| EXP-101 | 2026-08-05 07:33 | raw final evaluation과 summary/config 28개 지표 자동 대조 | 수기 전사 오류 제거 | validation PA-MPJPE 6.7684→6.7926cm 1건 수정 후 28/28 일치 | 원시 JSON 기준값 채택, manifest 재고정 |
| EXP-100 | 2026-08-05 07:30 | lmh/E01 uniform timestamp 262개를 scenario/risk/split별 재집계 | 부분 timestamp 보정의 분포 왜곡 여부 판단 | pose 16 scenario 전체, split 172/45/45, danger 30/10/10에 분산 | 262개 전체 복구 후 새 dataset/split version과 sealed 평가 |
| EXP-099 | 2026-08-05 07:28 | V12RG danger 5-class validation 오차 분해 | 다음 shared loss의 과적합 없는 우선순위 결정 | D02 root 45.84cm, D03 shift 12.86f, D04 distal 50.22cm, D05 endpoint 58.25cm가 각 병목 | class 전용 head 대신 root/temporal/distal shared loss 순차 ablation |
| EXP-098 | 2026-08-05 07:24 | Git archive clean-checkout audit와 text-lf/raw 이중 해시 | Windows CRLF로 인한 가짜 release mismatch 제거 | clean snapshot에서 문제 재현 후 LF-normalized text hash 구현, binary checkpoint는 raw 유지, 94/94 테스트 통과 | 플랫폼 독립 manifest 채택 |
| EXP-097 | 2026-08-05 07:21 | release manifest 검증 CLI와 누락 역할 allowlist, 3개 회귀 테스트 추가 | Git 공개본과 전체 연구본의 재현 범위를 자동 구분 | 연구본 40/40, 공개본 20/20 문서 검증; 외부 checkpoint/cache 20개만 명시적 누락; 전체 93/93 테스트 통과 | 릴리스 검증 절차 채택 |
| EXP-096 | 2026-08-05 07:15 | 공개본 protocol audit의 비공개 run-file 의존 제거와 분류 lock 요약 고정 | 체크포인트 없이도 split/selection 무결성 검사를 재현 | 공개된 lock만으로 20/20 통과, 원본 9MB 탐색 결과 SHA-256과 선택 제약 보존, manifest 39개로 확대 | 배포 재현성 수정 채택 |
| EXP-095 | 2026-08-05 07:04 | danger root 5-frame shift-robust warm-start + seed7 ensemble + V12G 결합 | timestamp 혼재에 둔감한 absolute root 복원 | V12RG clean root 31.28→31.03cm, danger 44.45→44.35cm, endpoint 55.52→55.25cm; link 장애 성능 유지 | test 미개봉 validation candidate 채택 |
| EXP-094 | 2026-08-05 06:55 | split/lock SHA-256 무결성 audit와 timestamp strata 진단 | 누수 및 정렬 품질 잔여 위험 확인 | 20/20 검사 통과; pose 262개가 lmh/E01 uniform_30fps, val root가 exact군보다 +7.43cm이나 subject와 confounded | 누수 없음, lmh timestamp 재수집 우선 |
| EXP-093 | 2026-08-05 06:47 | best/last snapshot root simplex 66조합 | 추가 학습 없는 root 분산 감소 검증 | 선택 `[0.8,0.0,0.2]`; last snapshot 가중치 0 | 기각, 기존 두 best 유지 |
| EXP-092 | 2026-08-05 06:39 | direct-root seed23 + 3-seed simplex 66조합 | clean root ensemble 다양성 확대 | seed23 단독 root 33.64cm, 최적 `[0.8,0.2,0.0]` | 새 seed 기각 |
| EXP-091 | 2026-08-05 06:31 | early/middle/late/shifted 50% link burst 외삽 audit와 이중 coverage gate | 한 burst 위치 과적합 및 분류 OOD 사용 방지 | 네 위치 recall 보존; root 1.0~1.7cm, danger 1.9~2.4cm 개선; FP 최대 +1 | pose/root threshold 0.5, class/risk 0.0 채택 |
| EXP-090 | 2026-08-05 06:18 | 부분 link coverage 0/0.5/0.75 비교와 최저 coverage link 식별 수정 | 완전 소실 외 간헐 단절 대응 | 0.5가 clean root 31.16cm, middle-burst root 38.82→37.41cm; 0.75는 clean 비열화 | trajectory threshold 0.5 채택 |
| EXP-089 | 2026-08-05 06:08 | V12G specialist 3개의 bitwise 동일 P2 backbone 공유 | 장애 시 중복 encoder 실행 제거 | shared/unshared robustness JSON exact match | shared guard execution 채택 |
| EXP-088 | 2026-08-05 05:58 | link 0/1/2 고정 소실 audit + link별 분류 calibration | 순환 평균이 숨긴 보드별 실패 검출 | 전역 guard가 link0에서 악화되어 기각; link별 gate로 세 link recall 비열화 0, 순환 recall 57→60/70 | link-specific V12G로 교체 |
| EXP-087 | 2026-08-05 05:48 | pose/root/class missing-link guard 통합 강건성 audit | 정상 경로 보존과 장애 복원 동시 확인 | clean root 31.28→31.18cm, drop-link root 46.35→43.09cm, danger 62.11→57.61cm, recall 57→60/70 | test 미개봉 V12G validation candidate 채택 |
| EXP-086 | 2026-08-05 05:45 | 80% link-dropout direct-root expert + 실패 validation 선택 | link 소실 root drift 완화 | drop root 46.35→43.09cm, danger root 54.81→49.59cm; clean root도 31.18cm | 조건부 root strength 1.0 채택 |
| EXP-085 | 2026-08-05 05:41 | link-failure class/risk probability blend와 오경보 hard gate | recall 증가 꼼수 방지 | recall 57→59/70, FP 13→14, risk acc 82.37% 유지; 단독 expert 62/70·FP20은 기각 | class 1.0, risk 0.5, failure bias -0.25 채택 |
| EXP-084 | 2026-08-05 05:38 | missing-link classification specialist를 장애 validation으로 선택 | link 장애 danger miss 감소 | 단독 recall 53→62/70이나 FP 13→20 | 단독 기각, 제한 blend만 사용 |
| EXP-083 | 2026-08-05 05:34 | missing-link pose specialist와 조건부 blend | 정상 pose를 유지하며 장애 pose 보정 | drop MPJPE 20.89→20.45cm, danger 62.11→61.83cm; clean +0.013cm | strength 0.5 guard 채택 |
| EXP-082 | 2026-08-05 05:31 | 80% link-dropout pose expert를 clean validation으로 선택 | 기존 selector의 robustness 적합성 점검 | clean 기준이 epoch 0 warm start를 선택해 장애 학습을 폐기 | 실패 기록, perturbation-specific selector 추가 |
| EXP-081 | 2026-08-05 05:29 | normalized CSI/motion cache + classification logit-only fast path | 동일 출력으로 multi-expert 비용 추가 절감 | validation JSON 전 항목 동일, mean 147.0→101.0ms, p95 165.8→113.7ms | 채택 |
| EXP-080 | 2026-08-05 05:27 | shared backbone 안에서 motion/normalization feature 재사용 | expert별 중복 전처리 제거 | 329개 tensor 최대 절대차 0 | 채택 |
| EXP-079 | 2026-08-05 05:26 | shared/unshared 실행을 validation 329개 tensor로 전수 비교 | 최적화 correctness를 자동 회귀검사로 고정 | pose/root/class/risk 최대 절대차 모두 0 | 동등성 audit 채택 |
| EXP-078 | 2026-08-05 05:19 | bitwise 동일 expert backbone 공유 + `norm` proxy | 정확도 보존 실행 최적화 | 모든 val metric delta 0, mean 147.0→127.0ms, p95 165.8→140.3ms, state 20.2→11.0MB | 채택 |
| EXP-077 | 2026-08-05 05:12 | expert의 p2 forward를 한 번으로 공유하는 1차 구현 | 정확도 변화 없이 latency 절감 | cache wrapper가 `norm`을 가려 val MPJPE 13.30→13.87cm | 기각 후 원인 수정, 잘못된 결과 미사용 |
| EXP-076 | 2026-08-05 05:06 | 최종 validation lock의 batch-1 GPU latency 측정 | multi-expert 배포 비용 공개 | 4.93M params, mean/p95 147.0/165.8ms, test 미사용 | 재현 benchmark 채택, distillation을 후속 과제로 기록 |
| EXP-075 | 2026-08-05 04:55 | 완전히 잠근 V12를 test 329 trial에 1회 평가 | test 재사용 없는 최종 일반화 측정 | MPJPE 15.07cm, PA 7.12cm, root 33.79cm, class 94.22%, risk 97.26%, danger recall 64/70 | 최종 보고, 이후 조정 금지 |
| EXP-074 | 2026-08-05 04:52 | pose seed23을 10/20/30%로 제한 ablation | seed diversity의 낙상 복원 이득 검증 | 30%에서 val danger 44.45cm, distal 45.72cm, endpoint 55.52cm, speed ratio 1.003 | pose 45/25/30 채택 |
| EXP-073 | 2026-08-05 04:46 | 공통 V11 pose에서 seed23 shift/RF 학습 | 두 seed 개선의 우연성 점검 | 단독 MPJPE 13.85cm이나 speed ratio 1.80 | 단독 기각, ensemble만 검증 |
| EXP-072 | 2026-08-05 04:43 | clean/RF-robust class expert logit ensemble | clean 비열화 없이 link 유실 분류 개선 | clean 동일, drop-link risk 81.46→82.37%, 오경보 15→13 | 0.5/0.5 채택 |
| EXP-071 | 2026-08-05 04:40 | corrected amp-phase RF로 class-only fine-tune | link 유실 danger recall 개선 | clean class/risk가 단독 expert에서 하락 | 단독 기각, ensemble 후보로만 사용 |
| EXP-070 | 2026-08-05 04:38 | corrected RF로 direct-root fine-tune | link 유실 root 강건성 개선 | validation 선택기가 epoch 0 초기값 선택 | 기각 |
| EXP-069 | 2026-08-05 04:35 | pose 2-seed와 root 2-seed 최종 조합 | seed 편향과 root drift 동시 감소 | MPJPE 13.31cm, root 31.28cm, danger 44.55cm | V12T 채택 후 3번째 seed 검증 |
| EXP-068 | 2026-08-05 04:31 | amp-phase 표현에 맞춘 RF pose fine-tune | 잘못된 I/Q 회전 제거 및 RF 강건성 개선 | clean/phase/subcarrier 모두 개선 | 채택 |
| EXP-067 | 2026-08-05 04:23 | direct-root seed17/7 ensemble | root 학습의 seed 의존성 완화 | 0.8/0.2에서 root 31.28cm, danger root 39.04cm | 채택 |
| EXP-066 | 2026-08-05 04:16 | seed7 direct-root multiscale 재학습 | root 개선 재현성 확인 | seed17과 다른 오차로 ensemble 이득 발생 | 채택 |
| EXP-065 | 2026-08-05 04:14 | risk-adaptive cross-seed pose blend | danger에 별도 pose 비율 적용 | 평균은 소폭 개선했으나 낙상 지표 악화 | 기각 |
| EXP-064 | 2026-08-05 04:08 | seed7 shift-robust pose 재학습 | shift loss 방향성 재현 | 독립 seed에서도 개선 방향 재현 | 2-seed ensemble 채택 |
| EXP-063 | 2026-08-05 04:03 | predicted danger 기반 hard/probability pose gate | 낙상 expert 선택 자동화 | seed 간 개선이 재현되지 않고 fall metric 불안정 | 기각 |
| EXP-062 | 2026-08-05 03:57 | validation-selected checkpoint model soup | 추가 추론비 없이 seed 이득 획득 | pose/danger 복합 지표 악화 | 기각 |
| EXP-061 | 2026-08-05 03:53 | 17-class와 3-risk 확률의 계층 calibration | danger recall 유지하며 warning/safe 오류 감소 | risk acc 97.26→97.87%, macro F1 96.90→97.63% | 채택 |
| EXP-060 | 2026-08-05 03:49 | trial-level global shift-robust pose loss | CSI-GT 수 프레임 오차에 대한 문맥 내성 | 평균·dynamic·danger가 함께 개선 | 채택 |
| EXP-059 | 2026-08-05 03:45 | direct absolute-root + 5/15/30-frame loss | velocity 누적 drift 제거 | root 33.86→31.34cm, danger root 개선 | 채택 |
| EXP-058 | 2026-08-05 03:30 | danger-weighted pose-only fine-tune와 분리 학습 | multi-task gradient 충돌 진단 | pose 평균 개선, 일부 distal trade-off 확인 | expert 분리 설계의 출발점 |

### V12 최종 판정

- validation: MPJPE `13.30cm`, PA-MPJPE `6.79cm`, root `31.28cm`, danger `44.45cm`,
  danger endpoint `55.52cm`, class/risk `96.05/97.87%`.
- test: MPJPE `15.07cm`, PA-MPJPE `7.12cm`, root `33.79cm`, danger `50.71cm`,
  danger endpoint `64.28cm`, class/risk `94.22/97.26%`.
- 최종 채택 이유: test 선택 없이 전체 pose와 분류가 강하고 validation의 root/danger도 개선됐다.
- 남은 문제: test danger recall `91.43%`, danger endpoint `64.28cm`, drop-one-link danger
  `57.61cm`(V12G validation). V12G는 test를 열지 않았으므로 새 최종 test 점수로 쓰지 않는다.
- 원시 결과: [`results/v12_final_evaluation.json`](results/v12_final_evaluation.json),
  [`results/v12_input_robustness.json`](results/v12_input_robustness.json),
  [`results/v12_link_failure_guard_robustness.json`](results/v12_link_failure_guard_robustness.json).

## 최신 기록: V10 P2-V9 dual hybrid

| 번호 | 완료 시각(KST) | 상세 내용 | 목적 | 결과 | 결정 |
|---|---|---|---|---|---|
| EXP-057 | 2026-08-05 00:37 | P2-V9 pose hybrid와 기존 V9C root expert를 validation blend | p2 pose/classification과 V9C root 강점 결합 | MPJPE 16.31cm, root 32.29cm, danger MPJPE 47.99cm, danger recall 95.71% | 현재 seen 권장 모델로 채택 |
| EXP-056 | 2026-08-05 00:25 | 동결 p2 위에 V9 rotation/root/logit residual 학습 및 독립 gate | p2 성능을 보존하며 낙상 trajectory 개선 | pose strength 0.35, class/risk residual 0; MPJPE 16.31cm, danger endpoint 65.17cm | pose residual 채택, 자체 root residual 기각 |
| EXP-055 | 2026-08-05 00:05 | p2 clean split 재현 후 LR 2e-4 fine-tune | 팀원 encoder 강점을 현재 코드·GT에서 재현 | val/test MPJPE 14.98/16.46cm, test class/risk 93.3/96.7% | V10 coarse checkpoint로 채택 |

### EXP-057: V10 P2-V9 dual hybrid

- 데이터는 ajh/mhw E01-E03과 lmh E01만 사용했다. train/validation/test는
  `1266/329/329`이며 test는 어떤 gate 선택에도 사용하지 않았다.
- p2의 amplitude/sanitized-phase 전처리, absence subtraction, shared LinkEncoder, FiLM,
  concat fusion, 4.2초 TCN과 class/risk logits를 유지했다.
- V9 residual은 p2 temporal feature, amplitude 차분, circular phase 차분, coarse motion을
  입력받아 bone rotation trajectory를 보정했다. validation pose strength는 `0.35`다.
- 자체 root residual은 validation 최소 개선폭을 못 넘어 `0`으로 기각했다. 기존 V9C root
  expert와의 blend는 validation에서 `0.50`이 선택되어 test root를 `36.25 -> 32.29cm`,
  danger MPJPE를 `53.63 -> 47.99cm`로 개선했다.
- class residual은 `0`, risk residual은 `0`이므로 p2 logits를 유지한다. validation-only
  danger bias `2.95`로 test danger recall은 `63/70 -> 67/70`, safe-to-danger는
  `2/175 -> 4/175`가 됐다.
- dual encoder의 계산량 증가는 남은 비용이다. 다음 실험은 V9C root expert를 p2 encoder에
  distillation하는 단일-backbone student다.

## 최신 기록: V9C clean-split multi-task

| 번호 | 완료 시각(KST) | 상세 내용 | 목적 | 결과 | 결정 |
|---|---|---|---|---|---|
| EXP-054 | 2026-08-04 23:26 | V9C temporal prior와 validation-only danger logit calibration | 낙상 recall을 높이면서 pose 안정화 여부 확인 | MPJPE 20.41cm, danger MPJPE 51.14cm, danger recall 66/70=94.29%, safe->danger 7/175=4.00% | 위험 분류 기준 모델로 채택, pose prior의 미미한 개선은 과대해석하지 않음 |
| EXP-053 | 2026-08-04 23:10 | V9 encoder에 pose/17-class/3-risk head를 연결하고 `single_split_lmh_e01`로 학습 | 팀원 모델 없이 V9 자체의 multi-task 성능 평가 | 17-class accuracy 87.84%, macro F1 84.89%; raw danger recall 62/70=88.57% | validation danger-recall gate와 logit calibration 추가 |

### EXP-054: V9C clean-split multi-task

- 데이터는 ajh/mhw E01-E03과 lmh E01만 사용했다. 전체 trial 수는 train/validation/test
  `1266/329/329`, pose GT trial 수는 `1210/315/315`이다.
- V9 temporal feature에 pose, 17-class, safe/warning/danger risk head를 연결했다.
- validation에서 temporal-prior strength `0.75`와 danger-logit bias `+1.75`를 선택했다.
  test는 두 값을 고르는 데 사용하지 않았다.
- risk calibration 전후 danger recall은 `62/70 -> 66/70`, risk macro F1은
  `93.33% -> 94.43%`다. safe-to-danger 오경보는 `5/175 -> 7/175`로 증가했다.
- temporal prior는 danger MPJPE를 `51.16 -> 51.14cm`로 0.02cm 낮췄지만 전체 MPJPE는
  `20.38 -> 20.41cm`로 0.03cm 악화했다. 따라서 pose 성능 향상으로 주장하지 않는다.

## 최신 기록: 종합 감사와 10안 결정

| 번호 | 완료 시각(KST) | 목적 | 방법 | 핵심 결과 | 결정 |
|---|---|---|---|---|---|
| EXP-052 | 2026-08-04 18:10 | 3시간 코드·데이터·연구 종합 감사 확정 | EXP-001~051, 40 JSON, 48 tests, 최신 연구 메커니즘과 gate 통합 | 승격 모델 없음; P0→V10 seen→joint-shift 순서 확정 | 새 학습 보류, 최종 실행안과 구현 명세를 기준 문서로 고정 |
| EXP-051 | 2026-08-04 18:06 | subject shift와 installation shift 분리 가능성 감사 | cache index field와 9 compound domain/geometry manifest 검사 | physical installation field 0, manifest 0; LOSO가 사람+설치를 동시 holdout | protocol 이름 정정, factorial 재수집 필요 |
| EXP-050 | 2026-08-04 18:00 | TX identity/order 의존성 측정 | V9A에서 CSI+link mask를 6개 순열로 함께 변환 | worst local/root/danger +6.24/+17.24/+13.82cm | board identity+geometry hash 필수, unordered set은 geometry 조건부 |
| EXP-049 | 2026-08-04 17:54 | subcarrier dropout의 mask 의미 검증 | zero band를 PerLinkNorm 전후로 측정 | mask 입력 없음, normalized band가 모든 frame 동일하고 temporal std 0 | mean impute+post-norm zero+subcarrier token |
| EXP-048 | 2026-08-04 17:49 | PerLinkNorm의 train/empty-room 계약 검증 | historical 20-batch train fit과 108 absence refit 후 val 분포 비교 | absence/train sigma ratio median 0.199; val amplitude std 1.14→14.76, p99 4.85→68.53 | train standardizer 고정, empty-room adapter와 분리 |
| EXP-047 | 2026-08-04 17:42 | V9 calibration 선택 재현성 확인 | 저장 후보를 현재 `0.8~1.15` gate와 score로 재선택 | V9A stored 0.15, current source 0.0; V9A/V9B 사이 gate 1.20→1.15 drift | V9 수치 historical 고정, V10은 resolved policy+source hash 저장 |
| EXP-046 | 2026-08-04 17:36 | robust sampler와 역빈도 CE·GroupDRO의 중복 보정 감사 | 4개 robust split에서 replacement sampler 100 epoch 모의실험 | epoch 고유 trial 60%, 중복 draw 40%; danger 17.6→29.4%, risk CE 질량 51~53% | sampler/CE/DRO를 한 번에 쓰지 않고 독립 ablation, 기본은 순회 sampler |
| EXP-045 | 2026-08-04 17:34 | domain invariance와 trial pose 목표 충돌 감사 | robust loss graph, 4 run history/metrics 분석 | SupCon 전 epoch 활성, action yja 4.18%·LOSO 5.93%로 chance 수준 | semantic invariant와 kinematic equivariant latent 분리 |
| EXP-044 | 2026-08-04 17:30 | 코드에만 있는 V3/kinematic 경로의 학습 여부 확인 | 51 checkpoint arch/state key 전수 검사 | arch=v3 0, kinematic state 0 | 검증 모델로 간주 금지, P0 후 from-scratch ablation |
| EXP-043 | 2026-08-04 17:27 | geometry-aware root 경로의 실제 활성 여부 | 51 checkpoint와 repository manifest 전수 검사 | nonzero geometry 0, available=true 0, configured path 0 | per-installation bundle로 재설계, 현 absolute root는 geometry-free로 표기 |
| EXP-042 | 2026-08-04 17:20 | V10 rotation target의 실제 가용성 확인 | split index와 `targets.py` GT 공통 키 감사 | ajh/lmh/mhw 2,366개는 joints-only, yja 263개만 full SMPL | 전원 SMPL 재추출 전에는 bone direction+FK, SO(3) loss 금지 |
| EXP-041 | 2026-08-04 17:15 | subcarrier 구조의 실효성 확인 | V9A에 reverse/roll/fixed permutation/frequency mean 반사실 입력 | permutation local/root +7.54/+17.12cm, mean danger absolute +14.52cm | frequency token 유지, 조기 pooling 지연, Doppler는 병렬 ablation |
| EXP-040 | 2026-08-04 17:10 | 진행 시점 변동과 nonlinear 정렬 상한 | 누적 이동거리 progress와 GT-oracle monotonic DTW | danger p10 p05~p95 15~135 frame, DTW danger local/absolute gain 2.37/2.40cm | dynamic-only progress 채택, time warp는 보조로 제한 |
| EXP-039 | 2026-08-04 17:07 | root 오차의 좌표 원점/궤적 성분 분해 | ajh+mhw global/site action template와 oracle initial translation | site가 root 6.85cm 개선하지만 시작점 oracle은 0.5~0.8cm뿐 | canonical displacement 우선, 새 설치 absolute는 geometry 조건부 |
| EXP-038 | 2026-08-04 17:03 | GVHMR 체형 변동의 pose 하한 측정 | exact GT bone direction에 고정 skeleton을 FK retarget | 동일 사람 bone CV 2.85~3.17%, LOSO shape-only local 1.29~1.67cm | raw/canonical skeleton 분리, optional trial-static shape head |
| EXP-037 | 2026-08-04 17:05 | amp/phase float16 cache 정밀도 검증 | 64 trial의 float32 resampling을 재실행해 632만 cache 값과 비교 | amplitude 상대 p99 0.0445%, phase 절대 p99 0.000405rad, mask mismatch 0 | float16 유지, `csi_iq` 이름과 문서만 수정 |
| EXP-036 | 2026-08-04 16:57 | hidden weighting과 mask semantics 감사 | risk/subject/time-method별 기대 gradient mass, GT-valid와 observation-valid 비교 | danger loss mass 70.8%, lmh 24.9% vs raw 33.3%; no-link GT frame는 train 0.018% | quality 중복 제거, target/observation mask API 분리 |
| EXP-035 | 2026-08-04 16:51 | offline/streaming 경계와 runtime 확인 | RTX 5060 Ti batch-1/24 timing, 미래 문맥 정적 감사 | batch-1 47.4ms/142MB지만 10.13초 양방향 문맥, streaming 불가 | offline reconstruction과 causal alert protocol 분리 |
| EXP-034 | 2026-08-04 16:46 | 전체 감사를 실행 가능한 V10 순서로 고정 | 데이터→cache→encoder→decoder→평가→unseen 의존성 정리 | 승격 모델 없음, P0 복구 후 global motion bank+calibration adapter+CSI residual 재학습 | [`final_code_audit_and_v10_execution_plan.md`](final_code_audit_and_v10_execution_plan.md)를 최종 기준으로 채택 |
| EXP-033 | 2026-08-04 16:39 | 정지 자세 collapse를 trial 다양성으로 측정 | V7/V9A의 temporal RMS와 같은 site/action 내 residual energy/cosine 비교 | V9A danger local RMS 6.63cm vs GT 28.25cm, danger trial residual cosine 0.050 | MPJPE 외 CSI-specific residual gate 신설 |
| EXP-032 | 2026-08-04 16:31 | calibration 의존성과 필요한 길이 확인 | V9A를 baseline none과 빈방 10/30/60/120초로 반복 평가 | 0초 local/root 31.32/54.25cm, 10초 20.60/31.65cm, 120초 20.60/31.61cm | 10초 runtime calibration API 구현, 조건별 별도 보고 |
| EXP-031 | 2026-08-04 16:24 | V9A/V9C와 smoothing의 실제 기여 검증 | raw/smoothed stage 비교와 trial bootstrap 10,000회 | V9A gain +0.682cm CI [0.600,0.766], V9C prior gain -0.072cm CI [-0.085,-0.059] | V9C prior 기각, V9A는 수정 전 비교용 |
| EXP-030 | 2026-08-04 16:16 | split·중복·checkpoint 재현성 감사 | hash duplicate, LOSO coverage, 51 checkpoint metadata 전수 검사 | exact duplicate 0, 현 LOSO held-out 17.1%만 평가, data/cache/split/git fingerprint 0/51 | full LOSO 재정의, provenance 의무화 |
| EXP-029 | 2026-08-04 16:08 | RF augmentation 표현 계약 검증 | 64 train trial 641만 값에서 증강 전후 채널 분포 비교 | phase std 31.4배, 55.0%가 abs(phase)>pi, amplitude 음수 발생 | 현 motion encoder 폐기, corrected augmentation으로 전면 재학습 |
| EXP-028 | 2026-08-04 15:58 | 잘못된 GT가 평가를 어떻게 왜곡했는지 측정 | contaminated 50 vs clean 355 dev_test의 V9 metrics 분리 | contaminated absolute 33.97cm vs 나머지 42.65cm, 전체 absolute 약 1.07cm 낙관 | 기존 점수 승격 금지 |
| EXP-027 | 2026-08-04 15:49 | pose GT 전체 품질과 frame index 감사 | 3,155 frame index, cached 2,629 pose의 orientation/jump/height 검사 | frame index 이상 0, patch 후에도 재추출 대상 16개 내외 | GT QC gate를 cache 이전 필수 단계로 이동 |
| EXP-026 | 2026-08-04 15:36 | orientation 수정본의 실제 반영 여부 확인 | Desktop patch, ZIP, D current, cache의 SHA-256·pose 좌표 비교 | patch 295 중 D와 동일 0, cache는 D old와 295/295 동일, test 50 오염 | 모든 기존 모델/점수를 pre-repair historical로 격하 |
| EXP-025 | 2026-08-04 16:04 | timestamp가 loss까지 보존되는지 감사 | 2,366 trial의 실제 frame Δt와 fixed-30Hz derivative 비교 | exact single-frame scale p05/med/p95 0.450/0.930/1.410, 3.312초 gap 발견 | cache에 time/Δt/gap 저장, actual-time loss로 교체 |
| EXP-024 | 2026-08-04 15:55 | CSI embedding retrieval의 추가 가치 확인 | 동일 site/action train trajectory k-NN, k는 validation에서 선택 | soft top-5 test local/root 13.54/29.55cm, danger absolute 44.66cm | 평균 prototype을 보존하고 confidence-gated retrieval을 ablation |
| EXP-023 | 2026-08-04 15:39 | 1~9안의 병목을 하나의 다음 설계로 통합 | 코드 실행 그래프, 데이터 contract, 반증 실험, 최신 연구 메커니즘을 종합 | `soft site×action prototype + monotonic progress + CSI trial residual` 초안 | cascade 중단은 유지, site prototype의 production 채택은 EXP-034에서 철회 |
| EXP-022 | 2026-08-04 15:32 | 강한 seen baseline 확인 | train GT site×action prototype을 GT/CSI hard/CSI soft class로 test | CSI soft prototype local 13.92cm, root 30.81cm, danger absolute 44.27cm로 9C를 크게 상회 | 수정 전 diagnostic baseline, production decoder 채택은 철회 |
| EXP-021 | 2026-08-04 15:30 | danger 분류 병목 확인 | risk/action별 accuracy, confidence, ECE 감사 | 전체 87.65%지만 danger 62.22%, D03 27.8% | hard class 금지, soft mixture와 hierarchy head 사용 |
| EXP-020 | 2026-08-04 15:28 | subgroup failure mode 확인 | subject/environment/risk/action/timestamp quality별 local/root/absolute/speed 평가 | danger local speed 0.089 vs GT 0.223m/s, D03/D04 local pose 31cm대 | local articulation과 root를 분리해 개선 |
| EXP-019 | 2026-08-04 15:21 | V9C가 trial별 CSI를 얼마나 쓰는지 확인 | same-site/class shuffle, reverse, shift, time mean, channel/link ablation | 같은 site/class trial shuffle은 local pose +0.22cm뿐, time mean은 +5.47cm | CSI는 쓰지만 local trial residual이 약함 |

원시 결과는 `work_v2/reports/v9_*_audit.json`, 고정 요약은
[`results/comprehensive_audit_v10.json`](results/comprehensive_audit_v10.json), 전체 진단과 다음
실험 gate는 [`comprehensive_diagnosis_and_plan_v10.md`](comprehensive_diagnosis_and_plan_v10.md)에 있다.

### EXP-052: final three-hour synthesis

- EXP-001~051의 코드 흐름, 51 checkpoint, source/cache/split, GT/timestamp, calibration, sampling,
  selection, counterfactual, runtime, 최신 연구 메커니즘을 하나의 dependency graph로 통합했다.
- 기계 판독 결과는 40 JSON이며 JSON parse, Markdown relative links, audit script compile,
  `git diff --check`, repository unit test `48/48`을 통과했다.
- 최종 판단은 승격 모델 없음이다. V9A는 pre-GT-repair historical comparator, V9C prior와 현 frozen
  cascade/robust encoder는 기각한다.
- 실행 순서는 GT patch/QC/target schema → cache v4/fingerprint → corrected 3-seed baseline → global
  motion bank+soft action+progress → CSI local/root residual → time/frequency ablation → uncertainty/OOF prior
  → participant+installation joint shift다.
- 새 학습을 실행하지 않은 이유는 P0 hard gate가 실패한 상태이기 때문이다. 잘못된 GT와 stale cache로
  새 점수를 만드는 것은 진전이 아니며, P0를 통과하기 전의 학습 결과는 비교표에 넣지 않는다.
- 기준 문서는 [`final_code_audit_and_v10_execution_plan.md`](final_code_audit_and_v10_execution_plan.md),
  파일/API 계약은 [`v10_file_level_implementation_spec.md`](v10_file_level_implementation_spec.md)다.

### EXP-051: subject-installation factorization audit

- source index의 subject는 ajh/lmh/mhw, environment code는 E01/E02/E03이며 model domain은 9개
  `subject_environment` compound key다.
- `installation_id`, `room_id`, `physical_site_id`, `geometry_id`, TX/RX position, camera extrinsic을
  검사했지만 index에 존재하는 필드는 0개이고 repository geometry manifest도 0개다.
- 같은 E01 코드가 사람 간 같은 물리 장소라는 증거가 없으므로 한 사람 LOSO는 그 사람과 세 설치를
  동시에 holdout한다. 이를 pure subject generalization으로 해석할 수 없다.
- 결론: 현 full LOSO는 `leave_one_participant_plus_installations_out`, within-subject LOEO는
  `seen_participant_unseen_installation`으로 표기한다. 세 participant fold는 진단이지 population estimate가
  아니다. 다음 수집은 같은 설치에 여러 사람, 같은 사람이 여러 설치에 참여하는 factorial design과
  installation/geometry ID를 필요로 한다.

### EXP-050: link identity and ordering counterfactual

- V9A seen dev test에서 site-baseline 적용 뒤 CSI와 `link_mask`를 함께 6개 TX 순열로 바꿨다. GT와
  나머지 입력은 그대로라 단순 board identity/order 반사실이다.
- TX2/TX3 swap만으로 local/root/danger absolute가 `20.60/31.61/51.15`에서
  `25.32/43.75/60.03cm`로 악화했다.
- worst `TX2,TX3,TX1`은 `26.84/48.85/64.98cm`, 정상 대비 `+6.24/+17.24/+13.82cm`다.
  가장 작은 비정상 순열도 local/root/danger가 `+2.62/+6.37/+2.16cm` 나빠졌다.
- 결론: shared link encoder가 permutation invariance를 주지 않는다. CSV row order에 의존하지 말고
  board serial, TX identity, antenna/position/orientation, installation hash로 link를 bind한다. 순서 불변
  set fusion을 쓸 경우에도 geometry/identity embedding으로 각 원소를 조건화하고 permutation unit test를 둔다.

### EXP-049: subcarrier-mask semantics

- `_augment_rf()`는 한 link의 4~16 subcarrier 두 채널을 0으로 만들지만 model input에는 frame×link
  `link_mask`만 있고 subcarrier mask가 없다.
- zeroing은 `PerLinkNorm` 전에 적용되므로 normalized 값은 0이 아니라 `-mu/sigma`다. TX1 live
  subcarrier 48:60 실측에서 원래 band abs mean은 amplitude/phase `0.575/0.737`, zeroed band는
  `0.322/0.350`이었고 모든 frame에 같은 값이라 temporal std 중앙이 정확히 `0`이었다.
- 크기 outlier는 아니지만 실제 packet loss와 다른 완벽한 rectangular constant signature다. 모델은
  이를 결손 견고성 대신 augmentation 식별자로 사용할 수 있다.
- 결론: 결손 band는 train mean으로 impute해 normalization 뒤 0이 되게 하고, post-norm mask를 다시
  적용하며 frequency encoder에 subcarrier-valid token을 전달한다. 이 세 경로를 끈 단독 ablation도 둔다.

### EXP-048: normalization and empty-room calibration contract

- `TrainConfig.norm_source`는 선언돼 있지만 trainer에서 읽히지 않고 CLI 옵션도 없다. 실제 fit은
  shuffled train의 처음 20 batch, 320 trial로 고정된다.
- `PerLinkNorm.fit()` 주석은 배포 때 빈방 10초로 다시 맞출 수 있다고 하지만, 모델은 사람 동작이 섞인
  train 분포로 학습됐다. 게다가 `SiteBaseline(sub)`가 이미 설치별 빈방 평균을 뺀 뒤다.
- train-fit과 source 9개 site의 absence 108 trial fit을 비교하면 subcarrier/channel별
  `sigma_absence/sigma_train` 중앙은 `0.199`, p05~p95 `0.057~0.786`이다. 살아 있는 link만 본
  site별 absence sigma도 최대/최소 `4.80x`다.
- 같은 validation 입력은 train norm에서 amplitude mean/std/p99-abs `0.08/1.14/4.85`지만 empty-room
  refit 후 `4.88/14.76/68.53`으로 폭증했다. 이것은 calibration이 아니라 encoder 입력 분포 파괴다.
- 결론: V10의 model standardizer는 train에서 전 데이터로 fit한 뒤 immutable하게 checkpoint에
  고정한다. empty-room CSI는 별도 installation adapter의 입력으로만 쓰며, 학습 때 동일 adapter 경로를
  거친다. `sub`, `sub_z`, PerLinkNorm, runtime adapter를 서로 다른 이름과 hash로 관리한다.

### EXP-047: selection-policy reproducibility audit

- V9A/V9B `results.json`의 calibration 후보를 현재 `train_seen_v4_trajectory.py` score와
  speed gate `0.8~1.15`로 다시 선택했다. score 수식 자체는 저장값과 일치했다.
- V9A 저장 결과는 speed ratio `1.186`인 pose strength `0.15`를 `feasible=true`로 선택했다. 이는
  실행 당시 상한이 `1.20`이었음을 뜻한다. 현재 source에서는 strength `0.05`도 `1.15061`로 탈락해
  strength `0.0`이 선택된다.
- V9B 저장 결과에서는 같은 `1.15` gate가 이미 적용돼 strength `0.0`이 선택됐다. 즉 V9A와 V9B
  사이 source policy가 바뀌었지만 두 result 모두 git/source fingerprint가 없다.
- 결론: V9A의 20.60cm는 당시 policy로 생성된 historical artifact이며 현재 source 재현값이 아니다.
  V10 result/checkpoint는 resolved selection formula, thresholds, smoother, metric version, source hash를
  함께 저장하고 policy 변경 시 새 experiment ID를 강제한다.

### EXP-046: robust sampler and reweighting overlap

- 네 robust split에서 `CrossDomainBatchSampler(batch=16)`를 100 epoch 모의실험했다. 모든 17개
  class가 pairable하고, yja holdout은 class마다 9 domain, LOSO는 6 domain을 가진다.
- sampler는 dataset을 순회하지 않고 class를 균등 복원추출한다. 한 epoch의 고유 trial은 평균
  `59.9~60.2%`, 중복 draw는 `40.2%`였다. 100 epoch 누적 trial 노출 횟수는 최소 `35~37`, 최대
  `206`으로 `5.6~5.9x` 차이였다.
- yja holdout의 raw risk 분포 safe/warning/danger `54.8/27.6/17.6%`가 sampler 뒤
  `52.9/17.7/29.4%`가 됐다. raw count로 계산한 inverse-risk CE까지 곱하면 보조 CE 질량은
  `29.4/19.6/51.0%`가 된다. LOSO danger 질량은 `52.8%`다.
- `GroupDRO(domain×risk)`와 별도의 danger/quality weighting까지 동시에 있으므로 기존 robust run은
  한 기법의 검증이 아니다. 특히 class-balanced sampler 뒤 raw inverse-frequency weight를 다시 쓰면
  희소 class가 이중 보정된다.
- 결론: V10 기본은 without-replacement epoch traversal과 metric 로그다. class balance, risk CE weight,
  GroupDRO는 한 번에 하나씩 추가하고, 같은 총 optimizer step/unique-trial exposure로 비교한다. sampler가
  필요하면 epoch 전체 coverage를 보장하는 weighted permutation 또는 batch composition constraint를 쓴다.

### EXP-045: domain-objective conflict audit

- robust SupCon positive는 `same action + different domain`이고 domain adversarial과 같은 pooled shared
  embedding을 쓴다. pose decoder도 같은 encoder의 temporal feature를 받는다.
- 4개 robust run 모두 `lambda_supcon=0.03`, `lambda_domain=0.03`, GRL `0.2`, 잘못된 RF augmentation,
  GroupDRO를 동시에 사용했다. train SupCon은 모든 epoch에 `2.50~2.74`로 활성됐다.
- historical yja action accuracy는 `4.18%`, LOSO-subsampled 평균은 `5.93%`로 17-class chance
  `5.88%` 수준이다. risk도 각각 `48.29/37.78%`였다.
- 잘못된 augmentation과 다중 objective 때문에 SupCon 단독 인과는 주장하지 않는다.
- 결론: action/risk용 domain-invariant semantic token과 calibration-conditioned trial kinematic token을
  분리한다. domain adversarial/SupCon은 semantic에만 적용하고 pose 쪽은 자기 CSI-pose pair와 다른
  같은-action trial을 구분하는 matching objective를 쓴다.

### EXP-044: implemented-versus-trained architecture audit

- 51개 checkpoint의 config/state를 전수 검사했다.
- arch 분포는 graphformer 8, impact graphformer 9, robust graphformer 8, latent flow 2,
  metadata 없는 cascade 24개다.
- `arch=v3` checkpoint와 `kinematic`/`bone_direction` state key가 있는 checkpoint는 모두 0개다.
- 결론: `V3PoseNet`, `DualViewFrequencyTokenizer`, `KinematicBoneDecoder`, geometry branch는 코드가
  존재할 뿐 성능이 검증되지 않았다. V10에서 코드를 재사용할 수는 있지만 corrected GT에서 baseline과
  같은 조건으로 처음부터 학습·비교한다.

### EXP-043: geometry contract audit

- 51개 `.pt`를 모두 열었고 load error는 없었다.
- geometry 관련 state/config entry가 있는 checkpoint는 19개지만 nonzero `board_geometry`와
  `geometry_available=True`는 0개, configured `geometry_path`도 0개다.
- repository 안의 geometry/layout/extrinsic manifest도 0개다.
- 현 `V3PoseNet`은 geometry unavailable일 때 zero vector를 biased MLP에 통과시키고, 파일이 있더라도
  모델 생성 시 하나의 tensor를 모든 site와 batch에 공유한다. camera extrinsic 입력은 없다.
- 결론: 현 결과는 geometry-aware가 아니다. V10은 availability-gated per-installation bundle과 hash를
  batch/checkpoint에 넣고 canonical root와 absolute transform을 분리한다.

### EXP-042: rotation target schema audit

- ajh 789, lmh 788, mhw 789 pose trial은 모두 `gvhmr_joints_v1`이며 공통 키는
  `joints_world/transl/frame_index`뿐이다.
- yja E02 pose 263개만 `gvhmr_smpl_full_v1`이다.
- joint position은 뼈 축 방향은 정하지만 그 축 주위 twist는 정하지 못하므로 full local rotation을
  유일하게 역산할 수 없다.
- 결론: 네 사람 모두 같은 GVHMR 버전의 SMPL body pose를 재추출하면 6D/SO(3)+FK를 사용한다. 그렇지
  않으면 unit bone direction+canonical FK를 쓰고 geodesic rotation loss와 twist 주장을 제거한다.

### EXP-041: frequency topology counterfactual

- 수정 전 V9A normal local/root/danger absolute는 `20.60/31.61/51.15cm`다.
- subcarrier reverse는 `23.34/38.21/57.97cm`, 10칸 roll은 `23.40/37.37/54.84cm`다.
- 고정 permutation은 `28.14/48.73/62.21cm`, frequency mean 반복은 `29.19/49.68/65.68cm`다.
- 결론: 현재 모델은 주파수 순서와 국소 구조를 실제로 쓴다. V10은 frequency branch를 제거하지 않고
  mean+max pooling을 늦추며 Doppler branch를 병렬 ablation한다.

### EXP-040: monotonic progress and oracle time-warp

- danger 누적 이동거리 p10 frame은 중앙 `66.5`, p05/p95 `15/135`로 시작·진행 시점 변동이 크다.
- GT를 보는 비배포 monotonic DTW는 ajh+mhw site/action prototype의 danger local을
  `15.05 -> 12.68cm`, absolute를 `32.38 -> 29.99cm`로 개선했다.
- 전체 평균 warp는 `9.86 frame`, p95는 `18.49 frame`이다.
- S02/S03/S04의 총 이동량은 중앙 `0.08~0.13m`라 progress가 실제 동작보다 GT jitter를 나타낸다.
- 결론: progress는 이동량 `>=0.5m` 동적 trial에만 적용한다. 시간 정렬은 유효하지만 주 병목을 해결할
  크기는 아니며, target pose를 보는 자유 DTW는 학습·평가에 쓰지 않는다.

### EXP-039: coordinate frame and root decomposition

- stale orientation GT를 피하려고 ajh+mhw 270 test trial만 사용했다.
- global action root template는 `35.97cm`, site/action은 `29.13cm`다.
- test 첫 root를 oracle로 맞춰도 각각 `35.14/28.61cm`로 `0.83/0.52cm`만 개선됐다.
- 같은 사람·행동의 환경별 initial root dispersion은 `3.73cm`, 전체 trajectory dispersion은 `17.54cm`다.
- 결론: 현재 root 병목은 단순 원점보다 이동 경로와 진행을 못 읽는 문제다. 다만 unseen 설치의 절대
  좌표·방향은 board/camera geometry 없이는 식별할 수 없으므로 canonical displacement를 주 지표로 둔다.

### EXP-038: body shape target audit

- 같은 사람의 train trial 사이 bone length CV 중앙값은 ajh/lmh/mhw `2.85/3.15/3.17%`다.
- bone length 범위 p95는 사람별 `5.49~7.65cm`, 최대 `5.78~7.90cm`다.
- GT bone direction을 완벽히 알아도 자기 train 평균 skeleton이면 seen local `1.10~1.18cm`가 남는다.
- 다른 두 사람 평균 skeleton을 쓰는 LOSO shape-only 하한은 local `1.29~1.67cm`, distal
  `1.73~2.66cm`다.
- 결론: GVHMR joint position은 articulation과 trial-varying shape noise를 섞는다. raw metric GT와
  canonical skeleton GT를 분리하고 rotation/direction을 주 타깃으로 학습한다.

### EXP-037: cache quantization audit

- 64 trial의 CSI를 float32로 다시 parse/resample하고 기존 float16 cache 632만 값과 비교했다.
- amplitude absolute p99 error는 `0.0588`, relative p99는 `0.0445%`다.
- sanitized phase absolute p99 error는 `0.000405rad`, maximum은 `0.00299rad`다.
- link mask mismatch는 0건이다.
- 결론: float16은 충분하다. `csi_iq`를 `csi_features`로 바꾸고 현재 amp/phase 표현에 맞는 주석과
  quantization regression threshold를 추가한다.

### EXP-036: hidden weighting and mask semantics

- danger는 train pose의 17.35%지만 sampler 4배, loss 2배, quality 중복으로 기대 loss mass 70.8%다.
- lmh는 raw 33.3%에서 기대 loss mass 24.9%로 줄고 mhw는 44.2%로 늘어난다.
- `uniform_30fps` compatibility 518 train trial은 partial-scaled timestamp이며 quality가 sampler와
  loss에 두 번 들어가 median effective weight가 낮다.
- GT-valid인데 세 link가 모두 없는 frame은 train 0.018%, dev_test 0.0058%라 현 성능 주 병목은 아니다.
- 결론: class/quality balance는 sampler 또는 loss 한 곳에서만 적용하고 실제 epoch mass를 로그로 남긴다.
  cache/model API의 target-valid와 observation-valid는 의미상 분리한다.

### EXP-035: runtime and streaming contract

- 수정 전 V9A는 RTX 5060 Ti batch-1에서 평균 `47.4ms`, p95 `52.7ms`, peak allocation `142MB`다.
- batch 24는 trial당 평균 `20.0ms`지만 peak allocation 약 `1.95GB`다.
- 입력은 304 frame=`10.13초`이고 symmetric temporal convolution, bidirectional Transformer,
  centered 5-frame smoothing을 사용한다.
- 결론: 계산 지연은 작지만 알고리즘상 10초 미래 문맥을 요구한다. 현재 모델은 offline reconstruction이며
  real-time alert가 필요하면 causal sliding-window 모델을 별도 평가한다.

### EXP-034: final integrated audit and V10 execution plan

- 현재 승격 가능한 checkpoint는 없다. V9A/V9C와 prototype 결과는 모두 수정 전 역사적 기준선이다.
- 실행 순서를 GT 복구 → cache v4/fingerprint → corrected baseline 재학습 → global action motion bank →
  monotonic progress → CSI rotation/root residual → encoder ablation → prior → unseen으로 고정했다.
- site×action prototype은 seen diagnostic으로만 유지하고 V10 production decoder는 global action bank와
  empty-room calibration adapter로 분해한다.
- 최종 문서: [`final_code_audit_and_v10_execution_plan.md`](final_code_audit_and_v10_execution_plan.md).

### EXP-033: motion diversity and collapse audit

- V9A danger local temporal RMS는 `6.63cm`, GT는 `28.25cm`다. 평균 크기 기준 약 23%다.
- 같은 site/action danger trial 사이 local 차이는 예측 `4.42cm`, GT `11.80cm`다.
- 예측 trial residual과 GT residual cosine은 danger local `0.050`, root `0.106`, absolute `0.102`다.
- 결론: 모델은 CSI를 쓰지만 trial마다 어떻게 넘어졌는지보다 평균 궤적을 출력한다. shuffle penalty,
  temporal RMS, residual cosine을 V10의 필수 gate로 추가한다.

### EXP-032: calibration dependency and duration

- baseline 없음은 V9A local/root `31.32/54.25cm`, danger absolute `69.38cm`다.
- random 10초 빈방 calibration 3회 평균은 `20.60/31.65/50.97cm`다.
- 120초 전체 baseline은 `20.60/31.61/51.15cm`로 10초와 실질적으로 같다.
- 현재 `calibrated_model.pt`는 environment adaptation이 아니라 validation-selected residual strength다.
- 결론: 10초 baseline은 유망하지만 runtime API와 installation-keyed bundle로 구현하고 no-cal/10/30/120초를
  별도 protocol로 보고한다.

### EXP-031: cascade, prior, smoothing bootstrap

- V9A는 V7 local을 평균 `0.682cm` 개선했고 95% CI는 `[0.600,0.766]cm`다.
- V9C prior는 V9A를 평균 `0.072cm` 악화했고 CI는 `[-0.085,-0.059]cm`다.
- danger에서도 prior gain은 `-0.036cm`, CI `[-0.057,-0.015]cm`다.
- centered 5-frame smoothing의 local gain은 `0.007cm`뿐이지만 speed correlation은 `0.069` 증가한다.
- 결론: V9C prior를 기각한다. raw/causal/offline-smoothed metric을 분리한다.

### EXP-030: split, duplicate, and provenance audit

- 2,749 cached trial의 CSI/GT exact hash duplicate는 0건이고 train/val/test exact copy도 없다.
- 현 LOSO는 held-out subject의 pose 788/789개 중 135개, 약 17.1%만 평가한다.
- 정식 full LOSO는 source train+test로 학습, source val로 선택, held-out subject 전체를 평가한다.
- 51개 `.pt` 중 split/cache/data fingerprint와 git commit을 기록한 것은 모두 0개다. source checkpoint를
  참조하는 6개도 path만 저장하고 SHA-256은 없다.
- 결론: checkpoint와 cache에 source lineage를 강제하고 현재 yja E02도 `unseen_dev_test`로 이름을 바꾼다.

### EXP-029: RF augmentation contract audit

- cache representation은 amplitude+sanitized phase인데 `_augment_rf()`는 두 채널을 complex I/Q로 회전한다.
- 64 train trial의 유효 값 641만 개에서 원 phase std `0.504`, 증강 phase std `15.856`이었다.
- 증강 phase의 55.0%가 `abs(phase)>pi`, 22.5%가 `abs(phase)>10`이고 amplitude 채널도 음수가 된다.
- motion-first/motion-pose/seen-residual pretraining이 이 경로를 사용했다.
- 결론: 현재 frozen motion encoder를 재사용하지 않고 representation-aware augmentation으로 처음부터 학습한다.

### EXP-028: metric contamination from stale orientation GT

- dev_test 405개 중 50개가 stale lmh orientation GT다.
- V9C absolute error는 contaminated 50에서 `33.97cm`, 나머지 355에서 `42.65cm`다.
- 같은 잘못된 좌표계를 train과 test가 공유해 전체 absolute 결과가 약 `1.07cm` 낙관적으로 보였다.
- 결론: 기존 성능은 corrected GT와 비교할 수 없고 논문 결과로 승격하지 않는다.

### EXP-027: full GT quality and frame index audit

- 3,155 pose 파일의 frame index는 모두 0부터 연속이며 joints/video frame 수와 맞았다.
- cached 2,629 pose에서 orientation warning 321, pose step `>0.2m` 15, absolute height `>2.5m` 3건이다.
- orientation patch를 적용하면 lmh E03 jump 2건은 사라진다. 이후 lmh E01, mhw E02, yja E02의
  추적 flip과 height 이상을 재추출하거나 제외해야 한다.
- 결론: GT QC를 cache 이전의 hard gate로 만든다.

### EXP-026: corrected GT lineage audit

- Desktop corrected GT와 `GT수정본.zip`은 동일하지만 D dataset current는 backup-before와 동일했다.
- patch 295개 중 D current와 같은 것은 0개, cache는 D current와 295/295 같고 patch와 0/295 같다.
- old→corrected target 이동은 local 평균 `46.05cm`, root `47.49cm`, absolute `68.10cm`다.
- 영향은 train/val/dev_test `195/50/50`개다.
- patch에서도 `lmh_E02_S01_t001` 한 건은 잘못 뒤집혀 반대 회전으로 재추출해야 한다.
- 결론: 모든 기존 checkpoint와 점수를 `pre-GT-repair historical`로 격하한다.

### EXP-025: actual timebase audit

- `frame_times()`는 CSI resampling에 사용되지만 cache schema에 frame time이 없어 이후 모델과 loss가 다시 모든 간격을 `1/30초`로 취급한다.
- exact timestamp 1,578 trial에서 fixed-30Hz single-frame speed/true speed scale은 p05/median/p95 `0.450/0.930/1.410`이다.
- exact frame transition의 14.5%는 0.75 미만, 23.4%는 1.25 초과다.
- 5-frame scale은 `0.846/1.032/1.218`로 완화되지만 동일하지 않다.
- train의 `ajh_E02_S04_t006`은 timestamp 마지막 구간에 `3.312초` gap이 있으며 index-based sequence에서는 바로 이웃으로 처리된다.
- 결론: timestamp 존재 여부만 weight로 쓰는 것으로 부족하다. frame time/Δt/discontinuity를 cache와 batch에 보존하고 derivative 및 positional encoding을 seconds 기준으로 바꾼다.

### EXP-024: CSI embedding trajectory retrieval

- frozen motion-first embedding으로 동일 site/action train trajectory를 cosine 검색했다.
- `k=1/3/5` 중 validation의 local+root+danger absolute 조합으로 선택했으며 GT/hard/soft class 모두 `k=5`였다.
- GT class top-5 test는 local/root/absolute `11.29/26.92/30.77cm`, danger absolute `34.96cm`다.
- CSI soft class top-5는 `13.54/29.55/34.28cm`, danger absolute `44.66cm`다.
- soft 평균 prototype 대비 overall은 `13.92/30.81/35.64 -> 13.54/29.55/34.28cm`로 개선했지만 danger absolute는 `44.27 -> 44.66cm`로 소폭 악화했다.
- 결론: retrieval은 유효한 CSI-conditioned prior지만 danger에서 무조건 우월하지 않다. 평균 prototype을 초기값으로 유지하고 confidence-gated retrieval만 ablation한다.

### EXP-019: CSI counterfactual audit

- 정상 입력은 local pose `20.68cm`, root `31.61cm`, danger absolute `51.14cm`다.
- 같은 site/action의 다른 trial CSI를 넣으면 각각 `20.90cm`, `34.34cm`, `52.93cm`다.
- 시간 평균 CSI는 `26.15cm`, `45.26cm`, `66.04cm`로 악화된다.
- 시간 역순과 block 역순에서 speed correlation이 거의 0이 되므로 temporal order는 사용한다.
- amplitude only, phase only, single-link는 모두 악화되어 두 채널과 세 링크의 정보가 유효하다.
- 결론: 9C는 CSI를 무시하지 않지만 같은 site/action 내부의 local pose 차이는 대부분 복원하지 못한다.

### EXP-020: subgroup and metric audit

- danger predicted local speed는 `0.089m/s`, GT는 `0.223m/s`로 관절 동작을 과소 복원한다.
- 기존 trial-average speed ratio는 정지 action의 작은 분모 때문에 S02/S03/S04에서 4~5배가 되어 danger under-motion을 가린다.
- D03 local pose `31.72cm`, D04 `31.49cm`, D02 absolute `59.63cm`가 주요 failure mode다.
- lmh approximate timestamp test는 local pose `22.11cm`, complete timestamp인 ajh+mhw는 `19.96cm`지만 subject가 섞인 비교이므로 인과로 단정하지 않는다.
- 결론: 기존 speed ratio를 model-selection에서 내리고 pooled local/root/absolute motion metrics로 교체한다.

### EXP-021: action classification audit

- overall action accuracy `87.65%`, ECE `3.49%`다.
- safe `96.14%`, warning `92.59%`, danger `62.22%`다.
- D01 `50.0%`, D02 `72.2%`, D03 `27.8%`, D04 `72.2%`, D05 `88.9%`다.
- 결론: danger trajectory decoder에 hard predicted class를 넣지 않고 soft distribution을 사용하며 D03를 별도 gate로 둔다.

### EXP-022: train-only motion prototype baseline

- train GT를 normalized frame 축에서 평균해 class, subject×class, environment×class, site×class prototype을 만들었다.
- GT site×class는 local `11.68cm`, root `28.31cm`, absolute `32.31cm`다.
- CSI hard site×class는 local `14.14cm`, CSI soft mixture는 `13.92cm`다.
- CSI soft mixture의 danger local/root/absolute는 `23.55/36.92/44.27cm`로 9C의 `28.91/~31.61/51.14cm`보다 pose가 낫다.
- 결론: site ID와 반복 choreography를 이용한 강한 seen baseline이다. unseen 성능으로 주장하지 않고, 10안이 반드시 넘어야 할 lower bound로 채택한다.

### EXP-023: comprehensive code audit and V10 design

- `amp_phase` contract인데 일부 pretraining에서 I/Q rotation augmentation을 적용한 오류를 확인했다.
- 9C는 1.10M baseline에서 4.51M까지 8개 frozen stage를 누적해 upstream 오류를 공동 수정할 수 없다.
- V9 prior의 observed-mask train/inference mismatch, 비현실적 corruption, local-only 보정, padding endpoint 문제를 확인했다.
- lmh 788 trial이 `uniform_30fps` compatibility이며 exact/approx 결과를 분리해야 한다.
- current 405 test는 반복 개발로 final holdout가 아니므로 `dev_test`로 취급한다.
- 결론: 10안은 corrected raw+Doppler encoder, CSI soft action, monotonic progress, site-action prototype, rotation/root residual을 end-to-end로 학습한다. seen shuffle gate 통과 전에는 unseen adaptation과 diffusion prior를 시작하지 않는다.

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
| EXP-018 | 2026-08-04 14:56 | GT-only temporal denoising prior 9C | MPJPE 20.68cm, danger 51.14cm, speed 1.163 | 당시 채택, EXP-031 bootstrap과 EXP-026 lineage 감사 후 기각 |

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
- 당시 판단: 이전 seen 모델보다 모든 위치 지표가 소폭 개선되고 speed gate도 통과해
  당시 권장 seen 모델로 채택했다. root 개선은 0.03cm에 불과해 별도 병목이었다.

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
- 당시 판단: 9C를 당시 권장 seen 모델로 채택했다. 이후 EXP-031 bootstrap에서 기각했다. 속도 안정화에는 성공했지만 danger
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
