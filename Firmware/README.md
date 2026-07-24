# NotiFi Firmware

ESP32-C6 기반 NotiFi CSI 데이터 수집용 펌웨어입니다. 기본 구성은 **3TX + 1RX**입니다.

- TX 3개: CSI 생성을 위해 ESP-NOW 패킷 송신
- RX 1개: TX 3개의 CSI를 받아 시리얼로 `CSI_DATA` 출력
- 대상 보드: Seeed Studio XIAO ESP32-C6 기반 보드
- 안테나: 외장 U.FL/IPEX 안테나 사용 기준
- ESP-IDF target: `esp32c6`

## 핵심 설정

| 항목 | 값 |
| --- | --- |
| TX1 MAC | `1a:00:00:00:00:00` |
| TX2 MAC | `1a:00:00:00:00:01` |
| TX3 MAC | `1a:00:00:00:00:02` |
| RX 허용 MAC | 위 TX 3개만 허용 |
| TX 송신 속도 | TX당 `30 pkt/s` |
| Wi-Fi 채널 | `11` |
| 시리얼 baud | `921600` |
| 외장 안테나 | GPIO3 `LOW`, GPIO14 `HIGH` |

## 폴더 구조

```text
NotiFi/Firmware/
├── firmware/
│   ├── csi_recv/main/app_main.c   # RX용 수정 코드
│   └── csi_send/main/app_main.c   # TX용 수정 코드
├── scripts/
│   ├── set_tx_mac.py              # TX1/TX2/TX3 MAC 끝자리 변경
│   └── check_tx_links.py          # 3개 TX 수신량 확인
└── requirements.txt
```

## 1. 준비

### macOS

```bash
cd ~/Desktop/NotiFi
git clone https://github.com/espressif/esp-csi.git

cp Firmware/firmware/csi_recv/main/app_main.c \
  esp-csi/examples/get-started/csi_recv/main/app_main.c
cp Firmware/firmware/csi_send/main/app_main.c \
  esp-csi/examples/get-started/csi_send/main/app_main.c

/opt/anaconda3/bin/python3 -m pip install -r Firmware/requirements.txt
```

ESP-IDF 환경을 켭니다.

```bash
source ~/esp/esp-idf/export.sh
```

포트 확인:

```bash
find /dev -maxdepth 1 \( -name 'cu.usbmodem*' -o -name 'cu.usbserial*' \) -print
```

### Windows PowerShell

```powershell
cd C:\Users\<USER>\NotiFI
git clone https://github.com/espressif/esp-csi.git

Copy-Item .\Firmware\firmware\csi_recv\main\app_main.c `
  .\esp-csi\examples\get-started\csi_recv\main\app_main.c -Force
Copy-Item .\Firmware\firmware\csi_send\main\app_main.c `
  .\esp-csi\examples\get-started\csi_send\main\app_main.c -Force

python -m pip install -r .\Firmware\requirements.txt
```

ESP-IDF 환경을 켭니다.

```powershell
cd C:\Users\<USER>\esp\esp-idf
.\export.bat
```

포트 확인:

```powershell
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Description
```

## 2. RX 굽기

RX 보드만 노트북에 연결하고 실행합니다.

### macOS

```bash
cd ~/Desktop/NotiFi/esp-csi/examples/get-started/csi_recv
idf.py set-target esp32c6
idf.py -p /dev/cu.usbmodemXXX -b 921600 build flash monitor
```

### Windows PowerShell

```powershell
cd C:\Users\<USER>\NotiFI\esp-csi\examples\get-started\csi_recv
idf.py set-target esp32c6
idf.py -p COM_PORT -b 921600 build flash monitor
```

`monitor` 종료는 `Ctrl + ]`입니다.

## 3. TX 굽기

TX는 **TX3 → TX2 → TX1** 순서로 굽습니다. 보드는 반드시 하나씩만 연결합니다.

### TX3

```bash
python Firmware/scripts/set_tx_mac.py --esp-csi esp-csi --tx tx3
cd esp-csi/examples/get-started/csi_send
idf.py set-target esp32c6
idf.py -p PORT -b 921600 build flash
```

### TX2

```bash
python Firmware/scripts/set_tx_mac.py --esp-csi esp-csi --tx tx2
cd esp-csi/examples/get-started/csi_send
idf.py -p PORT -b 921600 build flash
```

### TX1

```bash
python Firmware/scripts/set_tx_mac.py --esp-csi esp-csi --tx tx1
cd esp-csi/examples/get-started/csi_send
idf.py -p PORT -b 921600 build flash
```

Windows에서는 `PORT`를 `COM4` 같은 값으로, macOS에서는 `/dev/cu.usbmodemXXX`로 바꿉니다.

## 4. 최종 링크 확인

TX1/TX2/TX3는 보조배터리로 켜고, RX만 노트북에 USB로 연결합니다. `idf.py monitor`가 켜져 있으면 먼저 종료합니다.

### macOS

```bash
cd ~/Desktop/NotiFi
/opt/anaconda3/bin/python3 Firmware/scripts/check_tx_links.py \
  /dev/cu.usbmodemXXX --sec 10 --rate 30 --baud 921600
```

### Windows PowerShell

```powershell
cd C:\Users\<USER>\NotiFI
python .\Firmware\scripts\check_tx_links.py COM_PORT --sec 10 --rate 30 --baud 921600
```

정상 기준은 10초 동안 각 TX가 대략 `240`개 이상 들어오는 것입니다. 환경이 좋으면 TX당 `270~300`개 정도가 보입니다.

```text
[OK ] TX1 1a:00:00:00:00:00: 286
[OK ] TX2 1a:00:00:00:00:01: 281
[OK ] TX3 1a:00:00:00:00:02: 289
```

## 5. 문제 해결

| 증상 | 확인 |
| --- | --- |
| 포트가 안 뜸 | 데이터 케이블인지 확인, 보드 하나만 연결, macOS는 `/dev/cu.*`, Windows는 장치 관리자 확인 |
| flash 실패 | BOOT 버튼 누른 채 재시도, 포트 점유 중인 monitor 종료 |
| 특정 TX만 0개 | 해당 TX MAC 끝자리 중복/오류 가능성, `set_tx_mac.py` 후 다시 build flash |
| 특정 TX만 수신량 낮음 | 외장 안테나 체결, 안테나 방향, 보조배터리 전원 확인 |
| 3개가 모두 낮음 | Wi-Fi 채널 혼잡 가능성, 4개 보드 채널이 모두 같은지 확인 |
| CSI_DATA가 없음 | RX가 맞는지 확인, TX 전원 확인, RX monitor/수집 프로그램 포트 충돌 확인 |

## 6. 수집 배치 요약

- RX: 노트북 옆, 높이 약 80cm, USB 연결
- TX1: RX와 정면으로 마주 보게, RX와 1.5m 거리, 높이 약 80cm
- TX2: 측면 벽, 높이 약 140~150cm
- TX3: 반대쪽 대각/구석, 높이 약 30~40cm
- 사람/행동 영역: TX1-RX 직선의 중간 지점
- 외장 안테나: 4개 모두 u.FL/IPEX 커넥터 체결, 안테나 면이 행동 영역을 향하게 배치
