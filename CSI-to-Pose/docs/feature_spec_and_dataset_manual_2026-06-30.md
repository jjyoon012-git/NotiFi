# NotiFi CSI-to-Pose 기능명세서 및 데이터셋 수집 메뉴얼

Date: 2026-06-30  
Purpose: CSI-to-Pose 실험 기능 정의 및 학습용 데이터셋 수집 기준 정리

## 1. CSI-to-Pose 기능명세서

1. CSI-to-Pose는 WiFi CSI 신호만으로 사람의 자세와 움직임 흐름을 3D skeleton proxy로 복원하는 실험 기능이다.
2. 이 기능의 목표는 실제 영상을 생성하는 것이 아니라, 낙상/보행/무활동 판단에 필요한 관절 위치 변화를 추정하는 것이다.
3. 학습 단계에서는 CSI와 동기화된 영상을 함께 수집하고, 영상에서 추출한 pose를 teacher GT로 사용한다.
4. 실제 사용 단계에서는 카메라 없이 CSI 신호만 입력받아 skeleton sequence를 예측한다.
5. 예측 대상은 머리, 어깨, 팔꿈치, 손목, 골반, 무릎, 발목으로 구성된 13-point skeleton이다.
6. 출력 결과는 시간에 따른 skeleton 움직임, 자세 변화, 낙상 시작/종료 구간, 움직임 후 정지 여부를 설명하는 데 사용한다.
7. 개인 체형 차이를 반영하기 위해 첫 설정 단계에서 body template 또는 SMPL-X 기반 체형 정보를 선택적으로 생성한다.
8. CSI-to-Pose 결과는 단독 판정 모델이 아니라 safe/alert 모델의 설명 가능성과 세부 상황 해석을 보조한다.
9. 우선 목표는 standing/sitting/lying/walking/unstable_walking/fall-inactive 흐름을 구분 가능한 수준으로 복원하는 것이다.
10. 최종 목표는 “왜 alert가 떴는지”를 skeleton 흐름으로 보여주어 보호자와 사용자에게 이해 가능한 위험 설명을 제공하는 것이다.

## 2. 데이터셋 수집 목적

이번 데이터셋은 CSI-to-Pose 모델을 학습시키기 위한 것이다.

수집 목표:

```text
CSI CSV + Video GT
-> pose teacher로 13-point skeleton 추출
-> CSI student model 학습
-> 추론 시 CSI만으로 skeleton proxy 복원
```

중요 원칙:

- 영상은 학습용 GT 생성을 위해서만 사용한다.
- 실제 서비스 추론 단계에서는 영상 없이 CSI만 사용한다.
- 영상 원본은 민감 데이터이므로 외부 공유를 제한한다.
- 팀 공유/논문/발표에는 가능하면 skeleton, plot, overlay, 익명화된 결과를 사용한다.

## 3. 수집 장비 및 공간 세팅

| 항목 | 기준 |
| --- | --- |
| CSI 보드 | Seeed Studio XIAO ESP32-C6 기반 sender 1개, receiver 1개 |
| 펌웨어 | ESP-CSI `csi_send`, `csi_recv` |
| 송수신기 거리 | 1.5m 권장 |
| 허용 거리 | 1.2m-2.0m, 단 한 세션 중 위치 고정 |
| 보드 높이 | 70-100cm |
| 안테나 방향 | 두 보드 모두 세로 방향 고정 |
| 사람 위치 | sender-receiver 사이 또는 바로 근처 |
| 카메라 위치 | 정면 또는 대각 정면 |
| 영상 범위 | 머리부터 발목까지 최대한 포함 |
| 조명 | MediaPipe가 관절을 잡을 수 있을 정도로 밝게 |
| 안전 공간 | 침대, 매트, 이불 등 충격 완화 공간 확보 |

권장 배치:

```text
receiver --- 75cm --- 사람 행동 영역 --- 75cm --- sender
```

침대/의자 관련 라벨은 침대나 의자가 sender-receiver 경로 사이 또는 바로 근처에 오도록 둔다.

## 4. 한 Trial 수집 흐름

한 trial은 CSI와 video를 동시에 저장해야 한다.

```text
1. sender/receiver 위치 고정
2. receiver port 확인
3. CSI_DATA가 들어오는지 10초 테스트
4. 카메라 프레임 확인
5. label 선택
6. 3초 카운트다운
7. CSI + video 동시 수집
8. 수집 종료 후 CSV/MP4 저장 확인
9. pose extraction 실행
10. overlay 또는 plot으로 관절 추출 품질 확인
```

실패 trial 기준:

- CSI_DATA frame이 0개인 경우
- 영상이 저장되지 않은 경우
- 사람 몸이 거의 보이지 않는 경우
- pose detection rate가 90% 미만인 경우
- skeleton이 심하게 튀거나 뒤집히는 경우
- label과 다른 행동이 섞인 경우

실패 trial은 학습 데이터에서 제외하고 재수집한다.

## 5. 파일명 규칙

권장 파일명:

```text
{subject}_{label}_t001.csv
{subject}_{label}_t001.mp4
{subject}_{label}_t001_proxy13.csv
{subject}_{label}_t001_overlay.mp4
{subject}_{label}_t001_derived_features.png
```

예시:

```text
yja_unstable_walking_t001.csv
yja_unstable_walking_t001.mp4
yja_unstable_walking_t001_proxy13.csv
yja_unstable_walking_t001_overlay.mp4
```

권장 저장 구조:

```text
csi_to_pose/
  data/{risk}/{domain}/{label}/{subject}/
  videos/{risk}/{domain}/{label}/{subject}/
  pose_gt/proxy_13/{label}/{subject}/
  pose_overlays/{label}/{subject}/
  pose_plots/{label}/{subject}/
  body_templates/{subject}/
```

## 6. Skeleton GT 정의

CSI-to-Pose의 기본 예측 대상은 13-point skeleton proxy이다.

| Proxy Joint | 구성 | 역할 |
| --- | --- | --- |
| `head` | nose 또는 눈/귀 평균 | 높이 변화, 서기/눕기, 낙상 시 머리 위치 |
| `left_shoulder` | 왼쪽 어깨 | 상체 기울기, 낙상 방향 |
| `right_shoulder` | 오른쪽 어깨 | 상체 기울기, 낙상 방향 |
| `left_elbow` | 왼쪽 팔꿈치 | 손으로 짚기, 일어나기 시도, 경련 |
| `right_elbow` | 오른쪽 팔꿈치 | 손으로 짚기, 일어나기 시도, 경련 |
| `left_wrist` | 왼쪽 손목 | 손 움직임, 지지 동작 |
| `right_wrist` | 오른쪽 손목 | 손 움직임, 지지 동작 |
| `left_hip` | 왼쪽 골반 | 몸 중심, 앉기/서기/눕기 |
| `right_hip` | 오른쪽 골반 | 몸 중심, 앉기/서기/눕기 |
| `left_knee` | 왼쪽 무릎 | 보행, 앉기/서기 전환 |
| `right_knee` | 오른쪽 무릎 | 보행, 앉기/서기 전환 |
| `left_ankle` | 왼쪽 발목 | 보행, 발 걸림 |
| `right_ankle` | 오른쪽 발목 | 보행, 발 걸림 |

한 프레임 표현:

```text
13 joints x (x, y, z) = 39D pose vector
13 joints x (x, y, z, visibility) = 52D pose vector
```

발목이 일부 잘리면 학습 품질이 낮아질 수 있다.  
최소 기준은 머리, 어깨, 골반, 무릎이 안정적으로 잡히는 것이다.

## 7. Derived Feature

pose GT에서 아래 feature를 함께 계산한다.

| Feature | 설명 |
| --- | --- |
| `body_height` | visible joint 기반 신체 높이 |
| `head_height` | 머리 높이 |
| `hip_height` | 골반 높이 |
| `torso_angle` | 어깨-골반 중심축 기울기 |
| `motion_velocity` | 전체 joint 평균 속도 |
| `joint_acceleration` | 전체 joint 가속도 |
| `wrist_motion` | 손목 움직임 크기 |
| `knee_motion` | 무릎 움직임 크기 |
| `ankle_motion` | 발목 움직임 크기 |
| `fall_height_drop` | 머리/골반 높이 급락량 |
| `post_motion_static_score` | 움직임 후 정지 정도 |

이 feature들은 skeleton 복원 품질 확인과 downstream 위험 해석에 사용한다.

## 8. 수집 라벨 및 개수

라벨당 10회는 복원 성능 확인에 부족했다.  
시간이 빠듯한 v1 기준으로 아래처럼 나눈다.

```text
핵심 라벨: 25회
보조 라벨: 15회
위험 낙상 라벨: 안전을 위해 10-15회
```

### 8.1 Priority A: 필수 수집

먼저 이 세트만 수집해도 CSI-to-Pose v1 학습과 검증이 가능하다.

총 275 trials.

| label | 초/trial | 횟수 | 수집 목적 |
| --- | ---: | ---: | --- |
| `standing_still` | 20초 | 25 | 서 있는 기본 skeleton |
| `sitting_still` | 20초 | 25 | 앉은 자세 skeleton |
| `lying_still` | 20초 | 25 | 누운 자세 skeleton |
| `walking` | 20초 | 25 | 정상 보행 skeleton |
| `unstable_walking` | 20초 | 25 | 불안정 보행 흐름 복원 |
| `sit_to_stand` | 10초 | 25 | 앉기에서 서기로 전환 |
| `stand_to_sit` | 10초 | 25 | 서기에서 앉기로 전환 |
| `lie_to_stand` | 10초 | 25 | 누운 상태에서 일어남 |
| `stand_to_lie_normal` | 10초 | 25 | 선 상태에서 정상적으로 누움 |
| `bed_exit_failed` | 10초 | 25 | 침대에서 일어나려다 실패 |
| `post_fall_inactive` | 20초 | 25 | 낙상 후 무활동 |

예상 순수 수집 시간:

```text
20초 라벨 6개 x 25 = 3000초
10초 라벨 5개 x 25 = 1250초
총 4250초 = 약 71분
```

준비/저장/재촬영 포함 실제 소요는 약 2시간 내외로 예상한다.

### 8.2 Priority B: 성능 개선용

총 105 trials.

| label | 초/trial | 횟수 | 수집 목적 |
| --- | ---: | ---: | --- |
| `hand_move` | 20초 | 15 | 손/팔 움직임 분리 |
| `bed_sitting_to_stand_fall` | 10초 | 15 | 침대 앉은 상태에서 일어서다 낙상 |
| `bed_lying_to_stand_fall` | 10초 | 15 | 침대 누운 상태에서 일어나려다 낙상 |
| `chair_sitting_to_stand_fall` | 10초 | 15 | 의자에서 일어서다 낙상 |
| `chair_stand_to_sit_fall` | 10초 | 15 | 의자에 앉으려다 낙상 |
| `lying_convulsive_like_movement` | 10초 | 15 | 누운 상태 경련 의심 움직임 |
| `normal_breathing_visible` | 20초 | 15 | 호흡 시 상체 미세 움직임 참고 |

### 8.3 Priority C: 시간이 남을 때

총 50 trials.

| label | 초/trial | 횟수 | 수집 목적 |
| --- | ---: | ---: | --- |
| `walking_trip_fall` | 10초 | 10 | 보행 중 발 걸림 낙상 |
| `walking_turn_fall` | 10초 | 10 | 방향 전환 중 낙상 |
| `lying_fast_breath` | 20초 | 10 | 빠른 호흡 skeleton 참고 |
| `lying_slow_breath` | 20초 | 10 | 느린 호흡 skeleton 참고 |
| `lying_irregular_breath` | 20초 | 10 | 불규칙 호흡 skeleton 참고 |

호흡 라벨은 skeleton만으로는 미세하므로 CSI breathing stream과 별도 분석하는 것이 좋다.

## 9. 최소 수집안

시간이 매우 부족하면 아래 10개 라벨만 먼저 수집한다.

| label | 초/trial | 횟수 |
| --- | ---: | ---: |
| `standing_still` | 20초 | 25 |
| `sitting_still` | 20초 | 25 |
| `lying_still` | 20초 | 25 |
| `walking` | 20초 | 25 |
| `unstable_walking` | 20초 | 25 |
| `sit_to_stand` | 10초 | 25 |
| `stand_to_sit` | 10초 | 25 |
| `lie_to_stand` | 10초 | 25 |
| `bed_exit_failed` | 10초 | 25 |
| `post_fall_inactive` | 20초 | 25 |

최소 합계:

```text
250 trials
```

이 최소 세트는 자세, 보행, 불안정 보행, 자세 전환, 일어나기 실패, 낙상 후 무활동을 모두 포함한다.

## 10. 라벨별 행동 가이드

| label | 행동 가이드 |
| --- | --- |
| `standing_still` | 정면을 보고 서서 팔과 다리를 거의 움직이지 않는다. |
| `sitting_still` | 의자나 침대에 앉아 상체를 세우고 거의 움직이지 않는다. |
| `lying_still` | 침대/매트에 누워 몸을 거의 움직이지 않는다. |
| `walking` | sender-receiver 사이를 평소 속도로 자연스럽게 왕복한다. |
| `unstable_walking` | 실제로 넘어지지 않고, 보폭을 불규칙하게 하거나 좌우로 살짝 흔들리며 걷는다. |
| `sit_to_stand` | 앉은 상태에서 자연스럽게 일어난다. |
| `stand_to_sit` | 선 상태에서 자연스럽게 앉는다. |
| `lie_to_stand` | 누운 상태에서 상체를 일으키고 안전하게 일어난다. |
| `stand_to_lie_normal` | 선 상태에서 침대/매트에 정상적으로 눕는다. |
| `bed_exit_failed` | 누운 상태에서 일어나려다 상체만 들거나 팔로 지탱한 뒤 다시 눕는다. |
| `post_fall_inactive` | 매트/침대 근처 바닥에 누운 상태로 20초 동안 거의 움직이지 않는다. |
| `hand_move` | 큰 이동 없이 손, 팔, 상체 일부만 움직인다. |
| `lying_convulsive_like_movement` | 누운 상태에서 팔/다리/상체를 짧고 불규칙하게 움직인다. 무리하지 않는다. |

## 11. 품질 확인 기준

각 trial 수집 후 아래를 확인한다.

| 품질 항목 | 목표 |
| --- | --- |
| CSI_DATA 저장 | 0 frame이면 실패 |
| CSI 수집률 | 중간에 길게 끊기면 재수집 |
| video 저장 | mp4 파일이 정상 재생되어야 함 |
| pose detected rate | 90% 이상 권장 |
| head/shoulder/hip | 반드시 안정적으로 잡혀야 함 |
| knee | 가능하면 안정적으로 잡혀야 함 |
| ankle | 가능하면 포함, 잘리면 메모 |
| overlay | skeleton이 심하게 튀거나 뒤집히면 재수집 |
| label consistency | 다른 행동이 섞이면 제외 |

## 12. 안전 수칙

낙상 라벨은 실제로 세게 넘어지지 않는다.

- 매트, 침대, 이불 등 충격 완화 공간에서만 수행한다.
- 빠르게 쓰러지지 말고 천천히 무너지는 방식으로 수집한다.
- 혼자 수집할 경우 보행 낙상은 Priority C로 미룬다.
- 어지러움, 통증, 호흡 불편이 있으면 즉시 중단한다.
- 위험한 동작은 횟수를 줄이고 post-fall inactive 중심으로 대체해도 된다.

## 13. 최종 산출물

라벨별로 아래 파일이 있어야 한다.

```text
CSI CSV
Video MP4
Proxy13 CSV
Pose overlay MP4
Derived feature plot PNG
수집 메모
```

최종 검증 질문:

```text
CSI만 입력했을 때 skeleton sequence가 자세/보행/전환/낙상 후 무활동 흐름을 설명할 수 있는가?
```
