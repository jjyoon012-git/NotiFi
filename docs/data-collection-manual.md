# NotiFi CSI Data Collection Manual

Date: 2026-06-16  
Project: NotiFi  
Purpose: ESP32-C6 WiFi CSI 데이터 수집 조건을 통일하기 위한 매뉴얼

---

## 1. Manual Goal (매뉴얼 목적)

NotiFi는 카메라나 웨어러블 없이 WiFi CSI 신호를 이용해 노인의 낙상, 무활동, 호흡 이상 가능성을 감지하는 프로젝트이다.

WiFi CSI는 사람의 움직임뿐 아니라 보드 위치, 안테나 방향, 거리, 방 구조, 주변 물체에도 영향을 많이 받는다. 따라서 3명이 데이터를 수집하더라도, 가능한 한 사람만 다르고 나머지 조건은 같게 맞추는 것이 중요하다.

이 매뉴얼의 목표는 다음과 같다.

- 같은 보드와 같은 펌웨어를 사용한다.
- sender / receiver 위치를 고정한다.
- 보드 간 거리, 높이, 안테나 방향을 수치로 통일한다.
- 라벨별 행동 방식을 자세히 정의한다.
- 파일명과 실험 메모 형식을 통일한다.

---

## 2. Label Stages (라벨 단계)

```text
Stage 1 = Experimental validation labels (실험 검증용 라벨)
Stage 2 = Service state labels (서비스 상태 판단용 라벨)
Stage 3 = Event labels (순간 이벤트 라벨)
Stage 4 = Breathing abnormality labels (호흡 이상 라벨)
```

이번 1차 수집에서는 Stage 1을 먼저 진행한다. Stage 1은 모델 학습이 목적이 아니라, 우리 장비와 환경에서 CSI 값이 사람 위치와 움직임에 따라 유의미하게 달라지는지 빠르게 확인하기 위한 단계이다.

---

## 3. Hardware Setup (하드웨어 구성)

```text
board_model: Seeed Studio XIAO ESP32-C6 based modified board
sender_firmware: esp-csi/examples/get-started/csi_send
receiver_firmware: esp-csi/examples/get-started/csi_recv
target: esp32c6
baud_rate: 921600
collection_board: receiver
```

수집 규칙:

- 모든 수집자는 같은 보드 모델을 사용한다.
- sender 보드에는 `csi_send` 펌웨어를 올린다.
- receiver 보드에는 `csi_recv` 펌웨어를 올린다.
- PC에 연결해서 CSV를 저장하는 보드는 receiver이다.
- 실험 중 sender와 receiver 역할을 바꾸지 않는다.
- Arduino 방식은 사용하지 않고 ESP-IDF 기준으로 진행한다.

---

## 4. Sender / Receiver Placement (송신/수신 보드 배치)

기본 배치:

```text
[Sender] ---- 1.5 m ---- [Receiver + PC]
              center
          participant
```

고정값:

```text
sender_receiver_distance: 1.50 m
distance_tolerance: +/- 2 cm
board_height: 80 cm from floor
height_tolerance: +/- 2 cm
participant_position: center between sender and receiver
participant_center_distance_from_sender: 75 cm
participant_center_distance_from_receiver: 75 cm
```

배치 규칙:

- sender와 receiver 사이 거리는 1.50 m로 맞춘다.
- 거리는 보드 몸체 끝이 아니라, 가능하면 안테나 중심 또는 보드 중심 기준으로 잰다.
- sender와 receiver 높이는 둘 다 바닥에서 80 cm로 맞춘다.
- 높이는 보드 바닥이 아니라 안테나 중심 또는 보드 중심 기준으로 기록한다.
- sender와 receiver는 서로 마주보게 둔다.
- 사람은 sender와 receiver 사이 중앙에 선다, 앉는다, 또는 눕는다.
- 바닥에 sender 위치, receiver 위치, participant center 위치를 테이프로 표시한다.

---

## 5. Antenna Protocol (안테나 프로토콜)

### 5.1 Antenna Type (안테나 종류)

수집 전에 안테나 종류를 반드시 기록한다.

```text
antenna_type: external_UFL_IPEX or internal_PCB
antenna_model: same model if possible
external_antenna_gpio_enabled: yes or no
```

권장:

- 외장 U.FL/IPEX 안테나 개조품이면 모든 보드에서 외장 안테나를 사용한다.
- 일부는 외장, 일부는 내장으로 섞지 않는다.
- 외장 안테나 모델이 다르면 CSI 값이 달라질 수 있으므로 가능하면 같은 안테나를 사용한다.
- 안테나 모델이 다르면 메모에 반드시 적는다.

### 5.2 XIAO ESP32-C6 External Antenna Setting (외장 안테나 설정)

XIAO ESP32-C6 개조품에서 외장 안테나 사용이 필요한 경우, Wi-Fi 초기화 전에 아래 설정이 적용되어야 한다.

```text
GPIO3: LOW
GPIO14: HIGH
```

ESP-IDF C 코드 예시:

```c
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static void xiao_esp32c6_use_external_antenna(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << GPIO_NUM_3) | (1ULL << GPIO_NUM_14),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    gpio_config(&io_conf);

    gpio_set_level(GPIO_NUM_3, 0);
    vTaskDelay(pdMS_TO_TICKS(10));

    gpio_set_level(GPIO_NUM_14, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
}
```

사용 위치:

```c
void app_main(void)
{
    xiao_esp32c6_use_external_antenna();

    // 기존 Wi-Fi / ESP-NOW 초기화 코드
}
```

주의:

- 이 코드는 외장 안테나 개조품일 때만 적용한다.
- 적용 여부는 모든 보드에서 같게 맞춘다.
- 적용했다면 실험 메모에 `external_antenna_gpio_enabled: yes`로 기록한다.

### 5.3 Antenna Direction (안테나 방향)

안테나 방향은 아래 값으로 통일한다.

```text
antenna_orientation: vertical
antenna_axis_angle_from_floor: 90 degrees
angle_tolerance: +/- 10 degrees
antenna_tip_direction: upward
antenna_rotation: same direction for sender and receiver
```

실제 배치:

- 외장 안테나 끝이 천장 방향을 향하도록 세운다.
- 안테나 축이 바닥과 90도에 가깝게 되도록 한다.
- 허용 오차는 약 +/- 10도 이내로 한다.
- sender와 receiver 안테나를 같은 방식으로 세운다.
- 실험 중 안테나를 만지거나 돌리지 않는다.
- U.FL/IPEX 케이블이 움직이지 않도록 테이프로 가볍게 고정한다.

### 5.4 Antenna Clearance (안테나 주변 간격)

가능하면 아래 간격을 지킨다.

```text
antenna_to_metal_object_min_distance: 30 cm
antenna_to_laptop_min_distance: 30 cm
antenna_to_wall_min_distance: 30 cm
antenna_to_hand_before_trial_min_distance: 20 cm
antenna_height: 80 cm from floor
```

규칙:

- 안테나 주변 30 cm 이내에 금속 물체를 두지 않는다.
- 노트북, 보조배터리, 금속 책상 다리, 큰 전자기기는 안테나에서 최소 30 cm 이상 떨어뜨린다.
- 실험자는 녹화 또는 수집 시작 후 안테나를 만지지 않는다.
- 보드가 흔들리지 않도록 테이프나 고정대를 사용한다.
- 안테나 방향이 중간에 바뀌면 해당 trial은 재수집한다.

---

## 6. Environment Protocol (환경 조건)

기본값:

```text
location_id: room_A, room_B, ...
door_state: fixed
other_people: none
fan_or_aircon: fixed
large_objects: unchanged
```

환경 규칙:

- 가능하면 같은 장소에서 수집한다.
- 장소가 다르면 `location_id`를 반드시 다르게 기록한다.
- 실험 중 다른 사람이 sender와 receiver 주변을 지나가지 않는다.
- 문 열림/닫힘 상태를 고정한다.
- 선풍기, 에어컨, 큰 움직이는 물체는 가능하면 끈다.
- 책상, 침대, 의자, 큰 금속 물체 위치를 중간에 바꾸지 않는다.

---

## 7. Participant Protocol (참가자 규칙)

고정값:

```text
participant_id: P01, P02, P03
default_facing_direction: facing receiver
start_countdown: 5 seconds
rest_between_labels: 10 seconds
```

참가자 규칙:

- 참가자는 자신의 고정 ID를 사용한다.
- 예: `P01`, `P02`, `P03`
- 기본적으로 receiver 방향을 바라본다.
- 행동 시작 전 5초 카운트다운을 둔다.
- 라벨 사이에는 10초 준비 시간을 둔다.
- 실험 중 불필요한 말하기, 웃기, 몸 흔들기, 휴대폰 만지기를 하지 않는다.
- 휴대폰을 주머니에 넣을지 말지는 모든 trial에서 통일하고 메모에 적는다.

---

## 8. Stage 1 Label Set (1단계 라벨)

1차 수집 라벨:

```text
empty (사람 없음)
sitting_still (앉아서 가만히 있음)
standing_still (서서 가만히 있음)
lying_still (누워서 가만히 있음)
hand_move (손 움직임)
walk (걷기)
```

권장 수집 순서:

```text
1. empty
2. sitting_still
3. standing_still
4. lying_still
5. hand_move
6. walk
```

공통 수집 시간:

```text
duration_per_label: 60 seconds
trial_count_first_round: trial01
```

---

## 9. Detailed Action Guide (라벨별 행동 가이드)

### 9.1 empty (사람 없음)

목적:

- 사람이 없는 상태의 기준 CSI를 수집한다.

행동 규칙:

- sender와 receiver 사이에 아무도 없어야 한다.
- 참가자와 실험자는 측정 구간에서 최소 2 m 이상 떨어진다.
- 실험 중 sender와 receiver 사이를 지나가지 않는다.
- 의자, 매트 등 다음 실험에 필요한 물건은 미리 제자리에 두고, 수집 중 움직이지 않는다.

수집 조건:

```text
duration: 60 seconds
participant_location: outside sensing area
minimum_distance_from_boards: 2 m
movement_near_boards: none
```

재수집 조건:

- 수집 중 사람이 지나감.
- 보드나 안테나를 건드림.
- 주변 물체 위치가 바뀜.

### 9.2 sitting_still (앉아서 가만히 있음)

목적:

- 앉은 자세에서 움직임이 거의 없는 CSI 패턴을 확인한다.

행동 규칙:

- 의자를 participant center 위치에 둔다.
- 참가자는 의자에 앉고 receiver 방향을 바라본다.
- 등은 의자에 자연스럽게 기대거나 곧게 세우되, 모든 참가자가 같은 방식으로 한다.
- 손은 양쪽 무릎 위에 둔다.
- 발은 바닥에 붙인다.
- 고개, 팔, 다리, 몸통을 최대한 움직이지 않는다.

수집 조건:

```text
duration: 60 seconds
chair_position: participant center
facing_direction: receiver
hand_position: both hands on knees
foot_position: both feet on floor
movement: as still as possible
```

재수집 조건:

- 중간에 손을 움직임.
- 다리를 꼼지락거리거나 자세를 크게 바꿈.
- 웃거나 말하면서 몸이 흔들림.

### 9.3 standing_still (서서 가만히 있음)

목적:

- 선 자세에서 움직임이 거의 없는 CSI 패턴을 확인한다.

행동 규칙:

- 참가자는 participant center 위치에 선다.
- receiver 방향을 바라본다.
- 양발은 어깨너비 정도로 벌린다.
- 양팔은 몸 옆에 자연스럽게 내린다.
- 시선은 정면을 향하고, 고개를 돌리지 않는다.
- 60초 동안 자세를 유지한다.

수집 조건:

```text
duration: 60 seconds
position: participant center
facing_direction: receiver
feet_width: shoulder width
arm_position: naturally down
movement: as still as possible
```

재수집 조건:

- 발 위치를 크게 바꿈.
- 팔을 들어 올림.
- 몸을 좌우로 흔듦.
- 중간에 걷거나 방향을 바꿈.

### 9.4 lying_still (누워서 가만히 있음)

목적:

- 누운 상태의 CSI 패턴을 확인한다.
- 추후 수면, 무활동, 낙상 후 상태 분석과 연결될 수 있다.

행동 규칙:

- 매트나 침대를 participant center 위치에 둔다.
- 참가자는 sender와 receiver 사이 중앙에 눕는다.
- 기본 자세는 천장을 보고 눕는 자세로 통일한다.
- 머리 방향은 sender 쪽 또는 receiver 쪽 중 하나로 정해 고정한다.
- 권장 기본값은 머리 방향을 sender 쪽으로 둔다.
- 팔은 몸 옆에 자연스럽게 둔다.
- 다리는 편하게 펴고, 수집 중 움직이지 않는다.

수집 조건:

```text
duration: 60 seconds
lying_position: supine
head_direction: toward sender
arm_position: beside body
leg_position: straight and relaxed
movement: as still as possible
```

재수집 조건:

- 자세를 고쳐 누움.
- 팔이나 다리를 움직임.
- 머리 방향이 정해진 기준과 다름.
- 매트 위치가 중앙에서 크게 벗어남.

### 9.5 hand_move (손 움직임)

목적:

- 몸 전체 이동 없이 작은 움직임이 CSI에 반영되는지 확인한다.

행동 규칙:

- 참가자는 participant center 위치에 선다.
- receiver 방향을 바라본다.
- 몸통과 다리는 최대한 고정한다.
- 오른손만 사용한다.
- 오른손을 가슴 높이까지 올린다.
- 손을 좌우로 천천히 반복해서 흔든다.
- 팔 전체를 크게 휘두르기보다 손과 전완 중심으로 움직인다.

권장 움직임 수치:

```text
duration: 60 seconds
body_position: standing at participant center
moving_hand: right hand
hand_height: chest height
movement_direction: left-right
movement_width: about 30 cm
movement_speed: about 1 cycle per second
body_movement: minimized
```

1 cycle 의미:

```text
center -> left -> center -> right -> center = 1 cycle
```

재수집 조건:

- 왼손을 사용함.
- 몸 전체가 같이 흔들림.
- 걷거나 발 위치를 바꿈.
- 손 움직임 폭이 너무 작거나 너무 큼.

### 9.6 walk (걷기)

목적:

- 큰 움직임이 CSI에 뚜렷하게 반영되는지 확인한다.

행동 규칙:

- sender와 receiver 사이의 직선 경로를 사용한다.
- 참가자는 sender 쪽 30 cm 앞 지점에서 시작한다.
- receiver 쪽 30 cm 앞 지점까지 천천히 걷고 다시 돌아온다.
- 뛰지 않는다.
- 팔은 자연스럽게 흔들되 과장하지 않는다.
- 가능한 일정한 속도를 유지한다.

권장 움직임 수치:

```text
duration: 60 seconds
walking_path: sender-receiver line
start_point: 30 cm from sender
turn_point: 30 cm from receiver
walking_distance_one_way: about 90 cm
walking_speed: slow and steady
step_style: normal walking, no running
```

배치 예시:

```text
[Sender] -- 30 cm -- walk start ---- center ---- walk turn -- 30 cm -- [Receiver]
```

재수집 조건:

- 뛰거나 너무 빠르게 움직임.
- 보드 바깥쪽으로 크게 벗어남.
- 중간에 멈춰서 다른 행동을 함.
- sender 또는 receiver에 너무 가까이 다가가서 보드를 건드림.

---

## 10. Collection Procedure (수집 절차)

각 라벨마다 아래 순서로 진행한다.

```text
1. 보드 위치와 안테나 방향 확인
2. receiver가 PC에 연결되어 있는지 확인
3. serial_to_csv.py 실행
4. 5초 카운트다운
5. 정해진 행동을 60초 동안 수행
6. CSV 저장 확인
7. 10초 휴식
8. 다음 라벨 진행
```

예시 명령어:

```bash
python serial_to_csv.py --port COM_PORT --baud 921600 --duration 60 --label empty --outdir data/raw
python serial_to_csv.py --port COM_PORT --baud 921600 --duration 60 --label sitting_still --outdir data/raw
python serial_to_csv.py --port COM_PORT --baud 921600 --duration 60 --label standing_still --outdir data/raw
python serial_to_csv.py --port COM_PORT --baud 921600 --duration 60 --label lying_still --outdir data/raw
python serial_to_csv.py --port COM_PORT --baud 921600 --duration 60 --label hand_move --outdir data/raw
python serial_to_csv.py --port COM_PORT --baud 921600 --duration 60 --label walk --outdir data/raw
```

주의:

- `COM_PORT`는 실제 receiver 포트로 바꾼다.
- macOS에서는 `/dev/cu.usbmodem101` 같은 포트가 될 수 있다.
- Windows에서는 `COM3`, `COM4` 같은 포트가 될 수 있다.

---

## 11. File Naming Rule (파일명 규칙)

파일명 형식:

```text
YYYY-MM-DD_PARTICIPANT_LABEL_TRIAL.csv
```

예시:

```text
2026-06-16_P01_empty_trial01.csv
2026-06-16_P01_sitting_still_trial01.csv
2026-06-16_P01_standing_still_trial01.csv
2026-06-16_P01_lying_still_trial01.csv
2026-06-16_P01_hand_move_trial01.csv
2026-06-16_P01_walk_trial01.csv
```

규칙:

- 참가자 ID는 `P01`, `P02`, `P03`처럼 고정한다.
- 라벨명은 영어 라벨을 그대로 사용한다.
- 파일명에는 공백과 한글을 넣지 않는다.
- 같은 라벨을 여러 번 수집하면 `trial02`, `trial03`으로 증가시킨다.

---

## 12. Experiment Memo Template (실험 메모 양식)

각 CSV와 함께 아래 메모를 남긴다.

```text
date:
participant_id:
label:
trial:
duration:

board_model:
sender_firmware:
receiver_firmware:
target:
baud_rate:

antenna_type:
antenna_model:
external_antenna_gpio_enabled:
antenna_orientation:
antenna_axis_angle_from_floor:
antenna_tip_direction:
antenna_to_metal_object_min_distance:

sender_receiver_distance:
board_height:
participant_position:
facing_direction:

location_id:
door_state:
other_people:
fan_or_aircon:
wifi_channel:

phone_in_pocket:
clothing_note:
note:
```

작성 예시:

```text
date: 2026-06-16
participant_id: P01
label: hand_move
trial: trial01
duration: 60s

board_model: XIAO ESP32-C6 modified
sender_firmware: esp-csi csi_send
receiver_firmware: esp-csi csi_recv
target: esp32c6
baud_rate: 921600

antenna_type: external_UFL_IPEX
antenna_model: same external antenna
external_antenna_gpio_enabled: yes
antenna_orientation: vertical
antenna_axis_angle_from_floor: 90 degrees
antenna_tip_direction: upward
antenna_to_metal_object_min_distance: 30 cm

sender_receiver_distance: 1.50 m
board_height: 80 cm
participant_position: center
facing_direction: receiver

location_id: room_A
door_state: closed
other_people: none
fan_or_aircon: off
wifi_channel: 11

phone_in_pocket: no
clothing_note: normal clothes
note: right hand moved left-right at chest height
```

---

## 13. Success Check (수집 성공 기준)

수집 후 확인할 것:

```text
CSI_DATA lines exist: yes
collection_duration: about 60 seconds
RSSI values exist: yes
CSI array length mostly stable: yes
serial disconnected during collection: no
label action followed correctly: yes
```

재수집해야 하는 경우:

- CSV에 `CSI_DATA`가 거의 없다.
- 수집 중 USB 또는 시리얼 포트가 끊겼다.
- 안테나 방향이 중간에 바뀌었다.
- 보드 위치가 움직였다.
- 다른 사람이 측정 구간을 지나갔다.
- 라벨 행동을 잘못 수행했다.
- 60초보다 훨씬 짧게 저장되었다.

---

## 14. Quick Summary (요약)

반드시 통일할 것:

```text
1. 보드 모델
2. 펌웨어
3. sender / receiver 역할과 위치
4. 보드 간 거리: 1.50 m
5. 보드 높이: 80 cm
6. 안테나 방향: vertical, 90 degrees from floor, tip upward
7. 안테나 주변 간격: metal/laptop/wall from antenna >= 30 cm if possible
8. 사람 위치: sender와 receiver 사이 중앙
9. 바라보는 방향: receiver 방향
10. 라벨별 행동 방식
11. 수집 시간: 각 라벨 60초
12. 파일명
13. 실험 메모
```

1차 목표:

```text
Stage 1 라벨을 60초씩 수집하고, CSI_DATA가 실제로 저장되며 라벨별 통계가 달라지는지 확인한다.
```
