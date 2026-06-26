# NotiFi CSI-to-Pose

This folder documents the CSI-to-Pose experiment for NotiFi.

NotiFi CSI-to-Pose는 WiFi CSI 신호만으로 노인 행동 흐름을 직접 분류하는 것을 넘어, 영상 기반 pose teacher가 만든 skeleton representation을 CSI-only student model이 예측할 수 있는지 확인하는 실험입니다.

![CSI-to-Pose overview](assets/csi-to-pose-overview.png)

---

## 1. Experiment Goal

본 실험의 목적은 WiFi CSI 데이터만으로 다음 정보를 복원하거나 추정할 수 있는지 확인하는 것입니다.

- 사람이 서 있는지, 앉아 있는지, 누워 있는지
- 정상 보행인지, 불안정 보행인지
- 자세 전환이 언제 발생했는지
- 낙상 흐름이 언제 시작되고 끝났는지
- 낙상 후 무활동 상태가 나타나는지
- CSI만으로 3D skeleton proxy를 예측할 수 있는지

본 실험은 실제 영상 이미지를 복원하는 것이 아닙니다. 영상은 학습 단계에서만 사용하며, 영상에서 추출한 pose landmark를 ground truth로 사용합니다.

```text
Training:
CSI + Video
→ Video Pose Teacher
→ Skeleton GT
→ CSI Student learns CSI → Skeleton Proxy

Inference:
CSI only
→ Trained CSI Student
→ Predicted 3D Skeleton Proxy
```

---

## 2. Experiment Scope

이번 실험은 다음 순서로 진행합니다.

1. MediaPipe가 영상에서 skeleton landmark와 derived feature를 잘 추출하는지 확인
2. 기존 NotiFi CSI 수집 매뉴얼을 기반으로 CSI + video paired dataset 수집
3. 먼저 warning 라벨인 `unstable_walking`을 pilot으로 수집
4. `unstable_walking` 영상에서 pose GT와 derived feature 추출
5. CSI와 pose GT를 동기화
6. CSI-only model로 skeleton proxy 복원 가능성 확인

첫 pilot label:

```text
risk: warning
domain: gait
detail_label: unstable_walking
duration: 20s
ambient: quiet / aircon / tv / music
```

---

## 3. Dataset Unit

본 실험에서 샘플 1개는 CSI와 영상이 짝을 이루는 paired clip입니다.

```text
1 paired clip = CSI CSV 1개 + 같은 trial의 video MP4 1개
```

예시:

```text
yja_unstable_walking_t001.csv
yja_unstable_walking_t001.mp4
```

CSI와 영상은 반드시 같은 `subject`, `label`, `trial` 값을 가져야 합니다.

---

## 4. Data Collection Setup

기존 NotiFi 데이터셋 수집 매뉴얼을 기준으로 하되, 이번 실험에서는 영상 동기화를 추가합니다.

| Item | Rule |
|---|---|
| Sender / receiver distance | 1.5m |
| Person position | sender와 receiver 사이 중앙 |
| Board height | 바닥 기준 70~100cm로 고정 |
| Antenna direction | 두 보드 안테나가 서로 마주보게 고정 |
| Camera | 사람 전신, 침대/의자, 보행 경로가 보이게 고정 |
| Sync action | trial 시작 직전 손을 크게 들거나 박수 1회 |
| Ambient | `quiet`, `aircon`, `tv`, `music` 중 하나 기록 |

카메라 1대만 사용할 경우, 사람 전신과 주요 행동 공간이 모두 보이는 대각선 위치에 고정합니다. 가능하면 정면 카메라와 측면 카메라 2대를 사용하는 것이 좋습니다.

---

## 5. Skeleton Proxy Joint Definition

본 실험에서는 CSI 신호로부터 사람의 전체 영상을 직접 복원하는 것이 아니라, 노인 행동 감지에 필요한 핵심 신체 관절을 단순화한 13-point skeleton proxy를 예측 대상으로 설정합니다.

영상 데이터에서는 pose estimation model을 이용해 전신 관절 정보를 추출하고, 이 중 노인 행동 감지에 중요한 머리, 어깨, 팔, 골반, 무릎, 발목 중심의 관절을 선택하여 skeleton proxy를 구성합니다.

| Proxy Joint | 구성 | 역할 |
|---|---|---|
| `head` | nose 또는 양쪽 눈/귀 좌표의 평균 | 신체 높이 변화, 서기/눕기 구분, 낙상 시 머리 위치 변화 확인 |
| `left_shoulder` | 왼쪽 어깨 | 상체 기울기, 방향 전환, 낙상 방향 파악 |
| `right_shoulder` | 오른쪽 어깨 | 상체 기울기, 방향 전환, 낙상 방향 파악 |
| `left_elbow` | 왼쪽 팔꿈치 | 손으로 짚는 동작, 일어나려는 시도, 경련성 움직임 확인 |
| `right_elbow` | 오른쪽 팔꿈치 | 손으로 짚는 동작, 일어나려는 시도, 경련성 움직임 확인 |
| `left_wrist` | 왼쪽 손목 | 손 움직임, 지지 동작, 비정상적 팔 움직임 확인 |
| `right_wrist` | 오른쪽 손목 | 손 움직임, 지지 동작, 비정상적 팔 움직임 확인 |
| `left_hip` | 왼쪽 골반 | 몸 중심, 앉기/서기/눕기 구분 |
| `right_hip` | 오른쪽 골반 | 몸 중심, 앉기/서기/눕기 구분 |
| `left_knee` | 왼쪽 무릎 | 앉기/서기 전환, 보행, 균형 변화 확인 |
| `right_knee` | 오른쪽 무릎 | 앉기/서기 전환, 보행, 균형 변화 확인 |
| `left_ankle` | 왼쪽 발목 | 보행, 발 걸림, 불안정 보행 확인 |
| `right_ankle` | 오른쪽 발목 | 보행, 발 걸림, 불안정 보행 확인 |

각 proxy joint는 프레임 단위로 3차원 좌표를 가집니다.

```text
joint = (x, y, z)
13 joints × 3 coordinates = 39-dimensional pose vector
```

visibility 정보를 함께 사용할 경우 다음과 같이 확장할 수 있습니다.

```text
13 joints × (x, y, z, visibility) = 52-dimensional pose vector
```

---

## 6. Derived Features

관절 좌표만으로 학습하지 않고, 노인 행동 감지에 직접적인 단서가 되는 derived feature를 함께 계산합니다.

| Derived Feature | Meaning |
|---|---|
| `body_center` | 어깨 중심과 골반 중심을 이용한 몸 전체 중심 |
| `shoulder_center` | 양쪽 어깨의 중앙점 |
| `hip_center` | 양쪽 골반의 중앙점 |
| `body_height` | 머리부터 발목까지의 높이 |
| `torso_angle` | 어깨 중심과 골반 중심을 잇는 상체 축의 기울기 |
| `motion_velocity` | 몸 중심의 시간당 이동량 |
| `height_velocity` | 몸 높이의 시간당 변화량 |
| `wrist_motion` | 손목 움직임 크기 |
| `ankle_motion` | 발목 움직임 크기 |

낙상 및 위험 행동에서는 특히 다음 패턴을 중요하게 봅니다.

```text
body_height 급감
torso_angle 급변
body_center 급이동
이후 motion_velocity 감소 또는 0에 가까움
```

---

## 7. Model Plan

본 실험은 teacher-student 구조로 설계합니다.

| Component | Role |
|---|---|
| Video Pose Teacher | 영상에서 13-point skeleton proxy와 derived feature 생성 |
| CSI Student Model | CSI sequence만 보고 skeleton proxy와 event state 예측 |

초기 student model 후보:

1. TCN
2. 1D-CNN + GRU
3. Transformer Encoder

초기 실험에서는 데이터 수가 많지 않으므로 TCN을 우선 고려합니다.

```text
CSI sequence
→ TCN Encoder
→ Pose Head
→ Event Head
→ Direction Head
```

초기 출력:

```text
body_center
body_height
torso_angle
motion_velocity
13-point skeleton proxy
event_state
```

---

## 8. Pilot: unstable_walking

가장 먼저 `unstable_walking`을 수집하고 복원 결과를 확인합니다.

이 라벨을 먼저 선택한 이유:

- 정상 보행과 다르게 body center의 좌우 흔들림이 나타남
- ankle, knee, hip 움직임이 뚜렷함
- fall보다 안전하게 반복 수집 가능함
- skeleton proxy와 derived feature가 잘 나오는지 확인하기 좋음

권장 pilot 수집:

| Label | Duration | Repeat | Ambient |
|---|---:|---:|---|
| `unstable_walking` | 20s | 10 | quiet 3, aircon 2, tv 2, music 3 |

수집 note 예시:

```text
csi_to_pose_pilot_v1, camera_front, walking_path_near_bed, unstable_walking
```

---

## 9. Evaluation Plan

먼저 `unstable_walking` pilot에서 다음을 확인합니다.

1. 영상에서 MediaPipe landmark가 안정적으로 추출되는가
2. 13-point skeleton proxy가 프레임별로 저장되는가
3. `body_center`, `body_height`, `torso_angle`, `motion_velocity`가 계산되는가
4. CSI와 영상의 trial 번호와 시간이 맞는가
5. 불안정 보행 구간에서 skeleton proxy와 derived feature가 흔들림을 보여주는가

성공 기준:

```text
CSV에 CSI_DATA가 존재함
영상에 사람 전신이 보임
MediaPipe landmark 누락이 심하지 않음
unstable_walking에서 body_center와 ankle/knee movement 변화가 관찰됨
CSI-only student model의 입력/출력 형태를 만들 수 있음
```

---

## 10. Notes

본 실험은 privacy-preserving sensing을 목표로 합니다. 영상은 학습용 ground truth 생성을 위해서만 사용하며, 실제 배포 단계에서는 사용하지 않습니다.

```text
Training only: CSI + Video
Deployment: CSI only
```

이 실험은 단순한 safe / warning / danger classification보다 더 해석 가능한 중간 표현을 만드는 것을 목표로 합니다.

```text
CSI → skeleton proxy → behavior / event interpretation
```
