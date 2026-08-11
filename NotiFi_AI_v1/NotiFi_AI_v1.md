# NotiFi AI v1 기술 및 운영 문서

## 1. 모델 정의

NotiFi AI v1은 **세 개의 Wi-Fi CSI 링크만으로 사람의 행동과 위험도를 분류하고,
3D 관절 움직임을 추정하는 1차 배포 모델**이다.

최종 입력과 출력은 다음과 같다.

```text
입력
  CSI: [T, 3 links, 114 subcarriers, 2 channels]
  link_mask: [T, 3 links]
  설치별 calibration profile

출력
  action: 17개 행동 확률과 최종 행동
  risk: safe / warning / danger 확률과 최종 위험도
  pose_rel: [304, 22, 3] 골반 기준 3D 관절 궤적(m)
  root: [304, 3] 보조 전역 위치 출력(m)
  quality: 링크 커버리지와 신뢰도
```

모델의 기본 시간 계약은 30 Hz, 최대 304 frame이다. 약 10.13초보다 짧은 trial은
0과 `False` mask로 채우고, 긴 trial은 304 frame에서 자른다.

### 1.1 전체 파이프라인

```text
ESP CSI packet
  -> 필드/길이/송신기 검사
  -> TX1, TX2, TX3 분리
  -> 30 Hz 공통 시간축 보간 + packet mask
  -> guard/DC subcarrier 제거: 128 -> 114
  -> amplitude + sanitized phase 표현
  -> 설치 calibration: empty-room mean 제거
  -> 고정 CSI backbone
       |-> 행동 분류 head: 17 classes
       |-> 위험 분류 head: 3 classes
       `-> 시간별 CSI motion feature
             -> coarse pose와 motion descriptor
             -> train-only 3D motion bank 후보 검색
             -> learned candidate reranking
             -> 전신/부위별 속도 profile 비교
             -> CSI motion energy 기반 시간 보정
             -> risk-adaptive motion 혼합
             -> 골반 기준 SMPL-22 pose
```

### 1.2 CSI 전처리

ESP 원시 payload는 128개 subcarrier의 I/Q 256개 값이다. 파서는 손상된 행,
잘못된 `sender_id`, 잘못된 payload 길이, invalid first word를 제거한다.

학습과 배포는 동일하게 다음을 사용한다.

- 링크 순서: `TX1`, `TX2`, `TX3`
- 물리 배치 방향: `RX=North`, `TX1=South`, `TX2=West`, `TX3=East`
- 유효 subcarrier: guard/DC 14개를 제외한 114개
- 채널 0: amplitude
- 채널 1: 공통 위상 오프셋과 선형 성분을 줄인 sanitized phase
- 보간 간격: 30 Hz
- packet 허용 간격: 100 ms
- packet이 없는 시점과 불량 링크: 값 0, `link_mask=False`

거리와 설치 높이는 완전 고정값으로 가정하지 않지만, 보드의 논리 순서와 방위는
고정해야 한다. 방향 순서가 바뀌면 학습 때의 링크 의미가 바뀌므로 calibration만으로
안정적으로 복구하기 어렵다.

### 1.3 행동·위험 분류 branch

CSI backbone은 링크와 subcarrier의 변화, 시간별 움직임과 장기 문맥을 feature로 만든다.
고정된 분류 branch는 이 feature를 사용해 행동 17개와 위험도 3개를 각각 출력한다.

위험도는 행동 ID를 단순 변환한 값이 아니라 독립 risk logits를 사용한다. 따라서
행동을 일부 틀리더라도 danger 신호를 별도로 검출할 수 있다. 최종 API는 class ID뿐
아니라 전체 확률을 반환하므로, 제품에서는 danger threshold를 운영 정책에 맞춰
조정할 수 있다.

### 1.4 3D pose branch

3D pose는 관절 좌표를 처음부터 완전 생성하는 방식이 아니라 **CSI 조건부 motion
retrieval** 방식이다.

1. CSI feature에서 coarse pose, 행동 logits, 위험 logits, motion activity를 예측한다.
2. temporal selector가 CSI 전체 구간을 38개 bin으로 요약해 motion embedding을 만든다.
3. coarse pose와 predicted action으로 학습 motion bank에서 후보 20개를 고른다.
4. learned reranker가 CSI embedding, 후보 embedding, 행동, 위험도를 함께 비교한다.
5. 전신 속도와 머리·몸통·양팔·양다리 속도 profile로 후보를 다시 평가한다.
6. 상위 후보들을 혼합한 뒤 CSI의 누적 motion energy에 맞춰 동작 속도를 단조 보정한다.
7. danger 확률과 불확실도에 따라 coarse pose와 retrieval pose의 혼합 강도를 조정한다.
8. predicted action 상위 3개 안에서 profile ranker가 동작을 다시 선택한다.
9. 저주파 motion correction을 45% 반영하고 골반 좌표를 0으로 맞춘다.

motion bank는 **학습 split의 pose만** 포함한다. test pose, query GT, query 행동 정답,
원본 영상은 검색에 사용하지 않는다.

### 1.5 Calibration

Calibration은 신규 환경의 CSI를 모델이 학습 때 보던 입력 분포에 가깝게 만드는 설치
절차다. 현재 버전은 backbone 전체를 역전파로 fine-tuning하지 않는다.

첫 단계는 empty-room baseline 보정이다.

```text
calibrated_csi[t, link, subcarrier, channel]
  = raw_csi[t, link, subcarrier, channel]
  - mean_absence[link, subcarrier, channel]
```

absence trial의 유효 packet만 모아 링크·subcarrier·채널별 평균을 계산한다. 이 방식은
학습 때 적용한 site baseline subtraction과 동일하다. calibration에서 충분히 관측되지
않은 링크는 raw 값으로 통과시키지 않고 비활성화한다.

두 번째 단계는 support-only logit bias 보정이다. 설치 안내에 따라 수집한 기본 행동의
정답 ID와 frozen model logits를 비교해 작은 output bias만 학습한다.

- display action bias
- retrieval action bias
- display risk bias
- retrieval risk bias

bias는 강한 L2 규제와 `[-0.75, 0.75]` 제한을 사용한다. support 몇 개만으로 backbone이
특정 사용자에게 과적합되는 것을 막기 위한 설계다. 이 bias는 화면 분류뿐 아니라 pose
검색의 행동 후보에도 반영된다.

Calibration support와 실제 query는 반드시 분리한다. query의 GT pose, 행동 정답,
위험 정답 또는 영상을 이용해 profile을 바꾸는 것은 금지한다.

## 2. Seen 기준 기능과 성능

### 2.1 평가 조건

Seen 평가는 `ajh E01-E03`, `mhw E01-E03`, `lmh E01`에서 사람과 환경이 train/val/test에
모두 존재하되 trial은 분리된 고정 split을 사용했다. `lmh E02/E03`과 `yja`는 이 수치에
포함하지 않았다.

- 행동·위험 test: 329 trial, absence 포함
- pose test: 315 trial, absence 제외
- pose bank: train split 1,210 trial만 사용
- 모델·가중치 선택: validation만 사용
- 입력 전처리: 불량 링크 mask + 해당 site의 absence 평균 제거

### 2.2 분류 성능

| 지표 | 결과 |
|---|---:|
| 17개 행동 accuracy | 94.22% |
| 17개 행동 macro-F1 | 92.54% |
| 3개 위험도 accuracy | 97.26% |
| 3개 위험도 macro-F1 | 96.55% |
| danger recall | 91.43% (64/70) |
| safe를 danger로 오경보 | 1.14% (2/175) |

Accuracy는 전체 정답 비율이고, macro-F1은 각 class를 동일한 비중으로 평가한다.
Danger recall은 실제 danger 70개 중 64개를 danger로 찾았다는 뜻이다.

### 2.3 Pose 성능

| 지표 | 결과 |
|---|---:|
| 전체 MPJPE | 12.89 cm |
| distal MPJPE | 18.67 cm |
| 움직임이 큰 구간 MPJPE | 16.42 cm |
| danger pose MPJPE | 19.83 cm |
| danger distal MPJPE | 29.25 cm |
| danger endpoint MPJPE | 25.05 cm |
| 전체 speed correlation | 0.514 |
| danger speed correlation | 0.579 |

MPJPE는 대응하는 관절의 평균 3D 거리 오차다. Distal은 머리, 손목, 발목처럼 몸통에서
먼 관절을 강조한다. 현재 pose는 낙상 유형과 대략적인 3D 움직임을 시각화하는 1차
모델로 사용할 수 있지만, 29.25 cm의 danger distal 오차를 고려하면 정확한 부상 부위나
최초 충돌 관절을 의료 수준으로 확정할 수는 없다.

### 2.4 탐지 가능한 행동

| ID | 위험도 | 행동 |
|---:|---|---|
| 0 | safe | 걷기 (`walking`) |
| 1 | safe | 가만히 서기 (`standing_still`) |
| 2 | safe | 가만히 앉기 (`sitting_still`) |
| 3 | safe | 가만히 눕기 (`lying_still`) |
| 4 | safe | 누웠다가 일어서기 (`lie_to_stand`) |
| 5 | safe | 정상적으로 눕기 (`stand_to_lie_normal`) |
| 6 | safe | 사람 없음 (`absence`) |
| 7 | safe | 앉았다 일어서기 (`sit_to_stand`) |
| 8 | safe | 서 있다 앉기 (`stand_to_sit`) |
| 9 | warning | 불안정 보행 (`unstable_walking`) |
| 10 | warning | 비틀거린 뒤 회복 (`stumble_recover`) |
| 11 | warning | 침대 이탈 실패 (`bed_exit_failed`) |
| 12 | danger | 서 있다 낙상 (`fall_from_standing`) |
| 13 | danger | 걷다가 낙상 (`fall_while_walking`) |
| 14 | danger | 침대 이탈 중 낙상 (`bed_exit_fall`) |
| 15 | danger | 침대에서 낙상 (`bed_fall`) |
| 16 | danger | 의자 이탈 중 낙상 (`chair_exit_fall`) |

## 3. 신규 사용자 및 ESP 설치 절차

### 3.1 하드웨어 배치

네 개 보드는 행동 영역을 둘러싸도록 설치하며 모델의 고정 위치 계약은
`RX=North`, `TX1=South`, `TX2=West`, `TX3=East`이다. 현장 도면과 ESP 이름을 연결할
때 이 계약을 기준으로 등록하고, 설치 앱에서 실제 방 좌표와 논리 ID를 한 번 더 확인한다.

높이와 거리는 학습 설치와 완전히 같을 필요는 없지만, 행동 영역을 세 TX가 둘러싸고
안테나가 trial 중 회전하지 않아야 한다. 설치 후 가구를 크게 옮기거나 보드 위치를
바꾸면 calibration을 다시 수행한다.

### 3.2 장치 등록

`examples/device.example.json`에 RX/TX의 고유 ID 또는 MAC 매핑을 기록한다.

```powershell
notifi-ai --registry runtime\devices register `
  --config examples\device.example.json
```

등록 시 다음을 검사한다.

- `device_id`가 안전한 파일명인지
- RX/TX ID 네 개가 비어 있지 않고 서로 다른지
- 방향 순서가 학습 계약과 같은지

### 3.3 ESP 수집 데이터 계약

dataset 호환 CSV는 최소한 다음 필드를 포함한다.

```text
pc_elapsed_s, sender_id, csi_data, csi_len, first_word_invalid, rssi
```

`sender_id`는 반드시 `TX1`, `TX2`, `TX3` 중 하나여야 한다. `csi_data`는 256개 I/Q
정수 payload이며 `csi_len=256`이어야 한다. 각 TX packet은 같은 RX PC의 monotonic
시간축에 기록해야 한다.

### 3.4 Calibration 수집 권장량

1. 사람이 없는 상태를 10초씩 12회 수집한다.
2. 기본 행동 8종을 각 2회 수집한다.
3. support trial의 action/risk ID는 설치 안내 앱이 알고 있는 계획 동작으로 기록한다.
4. 낙상 support는 필수가 아니다. 추가한다면 매트와 보조 인력을 갖춘 통제 환경에서
   danger 5종을 각 1회만 선택적으로 수집한다.

기본 support 8종은 걷기, 서기, 앉기, 눕기, 누웠다 일어서기, 정상적으로 눕기,
앉았다 일어서기, 서 있다 앉기다.

`examples/calibration_manifest.example.json` 형식으로 CSV 경로와 support ID를 적고
다음 명령을 실행한다.

```powershell
notifi-ai --device cuda --registry runtime\devices calibrate `
  --device-id home-001 `
  --manifest examples\calibration_manifest.example.json
```

완료되면 다음 파일이 생성된다.

```text
runtime/devices/home-001/device.json
runtime/devices/home-001/calibration.pt
```

대규모 backend에서는 CSV 대신 `calibration.npz`를 API에 보낼 수 있다.

### 3.5 운영 전 점검

- calibration에서 최소 2개 이상 링크가 안정적으로 살아 있는지 확인한다.
- `link_coverage`가 낮은 보드는 케이블, 전원, 안테나, sender ID를 재확인한다.
- GPU 서버 시작 후 한 번 warm-up한다.
- `quality.low_quality=true`인 query는 자동 경보의 단독 근거로 사용하지 않는다.
- 보드 이동, 안테나 회전, 큰 가구 배치 변경 후에는 재-calibration한다.

## 4. 구현된 API와 실행 파이프라인

### 4.1 Python API

```python
from notifi_ai import NotiFiAIv1
from notifi_ai.registry import DeviceRegistry

model = NotiFiAIv1(device="cuda")
registry = DeviceRegistry("runtime/devices")
profile = registry.load_calibration("home-001")

prediction = model.predict(csi, link_mask, profile)
print(prediction.action_label)
print(prediction.risk_label)
prediction.save_npz("outputs/prediction.npz")
```

원시 dataset 호환 CSV는 `model.predict_csv(path, profile)`로 바로 처리할 수 있다.
서비스 시작 직후 `model.warmup()`을 한 번 실행하면 CUDA cold start를 실제 요청 전에
소모할 수 있다.

### 4.2 CLI

| 명령 | 기능 |
|---|---|
| `describe` | 모델, label, artifact metadata 출력 |
| `register` | RX/TX 설치 정보 등록 |
| `calibrate` | absence와 support로 profile 생성 |
| `predict` | CSV 또는 NPZ 한 trial 추론 |
| `serve` | FastAPI 서버 실행 |

모든 `--device`, `--registry`, `--artifacts` 옵션은 subcommand보다 앞에 둔다.

### 4.3 HTTP API

API 의존성을 설치한다.

```powershell
pip install -e ".[api]"
notifi-ai --device cuda --registry runtime\devices serve --host 0.0.0.0 --port 8000
```

구현된 endpoint는 다음과 같다.

| Method | Endpoint | 기능 |
|---|---|---|
| GET | `/health` | 모델 로드와 label metadata 확인 |
| GET | `/v1/devices` | 등록 장치 목록 |
| POST | `/v1/devices/register` | 장치 JSON 등록 |
| POST | `/v1/devices/{id}/calibrate` | calibration NPZ 업로드 |
| POST | `/v1/devices/{id}/predict` | query NPZ 업로드와 추론 |

`predict`의 `include_pose=false` 기본값은 JSON 응답 크기를 줄인다. 3D pose가 필요하면
`include_pose=true`를 사용한다.

현재 HTTP API는 batch trial endpoint다. ESP serial을 장시간 직접 수신하는 daemon,
WebSocket streaming, 사용자 인증, TLS, rate limit은 아직 포함하지 않았다.

### 4.4 NPZ 계약

Calibration NPZ:

```text
absence_csi    [N, 304, 3, 114, 2] float32
absence_mask   [N, 304, 3]          bool
support_csi    [M, 304, 3, 114, 2] float32  optional
support_mask   [M, 304, 3]          bool     optional
support_action [M]                   int64    optional
support_risk   [M]                   int64    optional
```

Query NPZ:

```text
csi       [304, 3, 114, 2] float32
link_mask [304, 3]         bool
```

Output NPZ:

```text
pose_rel          [304, 22, 3] float32
root              [304, 3]     float32
frame_valid       [304]         bool
action_probability[17]          float32
risk_probability  [3]           float32
action_id         scalar        int64
risk_id           scalar        int64
```

### 4.5 추론 시간

현재 개발 PC의 CUDA warmed smoke test에서는 10초 trial 하나가 약 0.21~0.23초였다.
이는 정식 hardware benchmark가 아니며 GPU, PyTorch, 동시 요청 수에 따라 달라진다.
첫 CUDA 요청은 graph와 kernel 초기화 때문에 수십 초가 걸릴 수 있으므로 서버 시작 시
warm-up이 필요하다. CPU 추론은 가능하지만 실시간 운영 기준으로 별도 측정해야 한다.

## 5. 주요 코드와 artifact

```text
NotiFi_AI_v1/
|-- notifi_ai/
|   |-- model.py          통합 추론 진입점
|   |-- retrieval.py      motion selector, reranker, profile, retiming
|   |-- calibration.py    absence mean과 support logit bias
|   |-- csi_parser.py     ESP CSV/IQ parser
|   |-- preprocessing.py  resampling, mask, 고정 304-frame 계약
|   |-- schemas.py        장치, support, prediction schema
|   |-- registry.py       장치와 calibration profile 저장
|   |-- io.py             NPZ와 calibration manifest 입출력
|   |-- cli.py            명령행 인터페이스
|   `-- api.py            FastAPI endpoint
|-- artifacts/
|   |-- notifi_ai_v1_core.ts
|   |-- notifi_ai_v1_features.ts
|   |-- notifi_ai_v1_core_cpu.ts
|   |-- notifi_ai_v1_features_cpu.ts
|   |-- notifi_ai_v1_retrieval.pt
|   `-- manifest.json
|-- examples/
|-- scripts/
|-- tests/
|-- README.md
`-- NotiFi_AI_v1.md
```

`notifi_ai_v1_core.ts`는 행동·위험·coarse pose·root를 내는 frozen TorchScript다.
`notifi_ai_v1_features.ts`는 retrieval에 필요한 시간 feature와 motion descriptor를 낸다.
`*_cpu.ts` 두 파일은 같은 graph의 CPU 실행용 artifact다. 런타임이 device에 맞춰
CUDA 또는 CPU 파일을 자동 선택한다.
`notifi_ai_v1_retrieval.pt`는 train-only motion bank와 selector/reranker/profile 가중치를
하나로 묶은 파일이다. 사용자가 앞 단계 체크포인트를 순서대로 다시 실행할 필요가 없다.

`manifest.json`은 다섯 artifact의 byte 크기와 SHA-256을 저장한다.

```powershell
python scripts\verify_release.py
python scripts\verify_release.py --smoke --device cpu
```

위 명령은 hash를 확인하고 CPU에서 모든 public runtime component를 실제로 로드한다.

## 6. 검증과 재현

```powershell
python -m compileall -q notifi_ai scripts tests
python -m unittest discover -s tests -p "test_*.py"
python scripts\verify_release.py
```

통합 패키지는 고정 seen test 329개를 다시 추론해 분류 confusion matrix가 기존 고정
결과와 일치하는지 확인했다. 독립 배포 retrieval과 기존 연구 retrieval의 pose 출력도
동일 입력에서 최대 절대 차이 0.0으로 확인했다.

상세 수치는 `docs/results/notifi_ai_v1_seen.json`에 기계 판독 가능한 형태로 저장한다.

## 7. 현재 한계와 제품 적용 주의사항

1. **Seen 성능은 unseen 보장이 아니다.** 위 수치는 학습에 등장한 사람·환경의 새 trial
   결과다. Calibration 코드는 구현했지만 신규 가정 전체에서 seen 수준을 보장하는
   대규모 외부 검증은 아직 완료되지 않았다.
2. **Calibration은 현재 full fine-tuning이 아니다.** Empty-room mean 제거와 bounded
   logit bias를 사용한다. 적은 support로 backbone을 망가뜨릴 위험은 낮지만, 체형과
   multipath 차이를 모두 제거할 수는 없다.
3. **Motion bank의 표현 범위가 한계다.** Bank에 없는 낙상 궤적은 가까운 학습 motion의
   혼합과 시간 보정으로 표현한다. 완전히 새로운 관절 움직임을 생성하지는 않는다.
4. **정확한 충돌·부상 판정 모델이 아니다.** Danger distal MPJPE가 29.25 cm이므로
   어느 부위가 먼저 닿았는지를 임상적으로 확정해서는 안 된다.
5. **절대 위치보다 상대 자세가 우선이다.** `root`는 설치 geometry의 영향을 크게 받는다.
   제품 시각화와 분석은 `pose_rel`을 기본으로 사용한다.
6. **한 사람을 전제로 한다.** 다중 사용자 분리와 누가 넘어졌는지 식별하는 기능은 없다.
7. **현재 inference는 window 단위다.** 완전한 packet streaming state machine과 조기
   낙상 경보 latency 평가는 후속 과제다.
8. **API 보안은 개발용이다.** 외부 공개 전 인증, TLS, encryption at rest, audit log,
   request size limit과 rate limit을 추가해야 한다.

## 8. 다음 개발 우선순위

1. 사용자와 환경을 완전히 분리한 calibration validation split을 만든다.
2. 신규 가정마다 support 수를 0/4/8/16개로 바꿔 성능 곡선을 측정한다.
3. retrieval pose에 CSI 조건부 continuous residual decoder를 추가하되, bank 복사 대비
   실제 개선과 과적합을 별도로 검증한다.
4. floor proximity/contact head를 명시적으로 학습하고 부위별 calibration을 평가한다.
5. ESP 실시간 수집 daemon, rolling 10초 window, danger threshold와 alert cooldown을
   통합한다.
6. GPU/CPU/edge device별 latency, memory, 동시 요청 수와 cold start를 정식 측정한다.

NotiFi AI v1은 현재 데이터에서 검증된 seen 분류와 retrieval pose를 하나의 실행 가능한
제품 인터페이스로 묶은 기준 모델이다. 이후 calibration이나 continuous pose generator를
개선할 때도 이 버전의 고정 split, leakage 금지 조건, artifact hash와 지표를 비교 기준으로
유지한다.
