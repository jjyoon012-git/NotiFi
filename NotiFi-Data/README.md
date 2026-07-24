# NotiFi Dataset Collection v2.0

NotiFi의 3TX+1RX CSI, RGB 영상, 13-point pose teacher GT를 동기화해 수집하는 현장용 도구다. 기준은 `NotiFi 데이터셋 수집 계획서 v2.0 (2026-07-22)`이며, 이전 데이터셋의 배경음·가전 상태별 분기와 과거 라벨은 사용하지 않는다.

## 수집 목표

| Risk | 라벨 수 | 1인·1환경 | 1인·3환경 | 4인 전체 |
| --- | ---: | ---: | ---: | ---: |
| SAFE | 9 | 150 | 450 | 1,800 |
| WARNING | 3 | 75 | 225 | 900 |
| DANGER | 5 | 50 | 150 | 600 |
| 합계 | 17 | 275 | 825 | 3,300 |

수집자는 `ajh`, `lmh`, `mhw`, `yja` 네 명이며, 네 명 모두 동일한 세 물리적 환경 `E01`, `E02`, `E03`에서 전체 라벨을 수집한다. 모든 원본 trial은 10초다.

## 1. 설치

Windows PowerShell:

```powershell
git clone https://github.com/NotiFi2026/NotiFi-Data.git
cd NotiFi-Data
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/generate_sounds.py
```

Python 3.11을 권장한다. Windows의 카메라와 COM 포트를 사용하므로 VS Code도 같은 `.venv` 인터프리터를 선택해야 한다.

COM 포트 확인:

```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
```

## 2. 3TX+1RX 준비

- TX1 MAC: `1a:00:00:00:00:00`
- TX2 MAC: `1a:00:00:00:00:01`
- TX3 MAC: `1a:00:00:00:00:02`
- TX별 송신률: `30 pkt/s`
- 네 보드 채널: 동일한 `11`
- RX 시리얼: 기본 `921600 baud`

펌웨어 수정과 플래시는 [3TX+1RX 펌웨어 가이드](firmware/README.md)를 먼저 완료한다. 본 수집 전 세 TX를 모두 켜고 RX만 PC에 연결한 뒤 아래 검사를 통과해야 한다.

```powershell
python scripts/check_tx_links.py COM_PORT --rate 30
```

10초 기준 TX1/TX2/TX3가 각각 270 frames 이상이어야 한다. `idf.py monitor`가 RX 포트를 사용 중이면 `Ctrl+]`로 종료한다.

## 3. 장비 배치

| 장비 | 기준 |
| --- | --- |
| RX | 활동 중심 정면, 높이 0.80 m, USB로 노트북 연결 |
| TX1 | RX 맞은편, RX와 1.50 m, 높이 0.80 m |
| TX2 | 활동 중심 측면, 높이 1.45 m |
| TX3 | 활동 중심 대각 바닥 측, 높이 0.35 m |
| 안테나 | 네 보드 모두 세로 편파, 케이블 위치 고정 |
| 카메라 | 활동 중심을 향하며 전신·매트·의자·침대가 모두 보이도록 고정 |

세 환경에서 같은 로컬 좌표와 가구 배치를 사용한다. TX2와 TX3의 정확한 수평 좌표는 파일럿에서 확정한 실측값을 세 환경에 동일하게 복제하고 setup 사진에 남긴다. session 중 위치가 2 cm 또는 각도가 3도 이상 바뀌면 수집을 멈추고 새 `device_config_id`를 사용한다.

카메라 확인:

```powershell
python scripts/preview_camera.py --camera 0
```

## 4. 세션 설정

환경이나 날짜가 바뀔 때마다 새 session을 만든다. placeholder를 실제 값으로 바꾼다.

```powershell
python scripts/create_session.py `
  --subject SUBJECT_ID `
  --environment E01 `
  --session YYYYMMDD_AM01 `
  --port COM_PORT `
  --camera 0 `
  --firmware-commit FIRMWARE_COMMIT `
  --device-config K0_P0 `
  --channel 11 `
  --packet-rate 30
```

로컬 설정은 `.notifi_session.json`에 저장되며 Git에는 올라가지 않는다.

환경별 권장 session 배치는 다음과 같다. 한 session의 DANGER는 마지막에 수행하고 10회 종료 후 10분 이상 쉰다.

| Session | 비낙상 라벨 | DANGER | 합계 |
| --- | --- | --- | ---: |
| A | `stand_to_lie_normal` 30 + `unstable_walking` 20 | `fall_from_standing` 10 | 60 |
| B | `stumble_recover` 30 + `lying_still` 18 | `fall_while_walking` 10 | 58 |
| C | `bed_exit_failed` 25 + `walking` 24 | `fall_collapse` 10 | 59 |
| D | `lie_to_stand` 18 + `standing_still` 12 + `sitting_still` 12 | `bed_fall` 10 | 52 |
| E | `absence` 12 + `sit_to_stand` 12 + `stand_to_sit` 12 | `chair_fall` 10 | 46 |

이 다섯 session을 `E01`, `E02`, `E03`에서 반복하면 개인별 825회가 된다.

## 5. 수집

예시:

```powershell
python scripts/collect_dataset.py --label walking --repeat 24
```

DANGER 예시:

```powershell
python scripts/collect_dataset.py `
  --label fall_from_standing `
  --repeat 10 `
  --safety-confirmed
```

기본 동작은 다음과 같다.

1. 첫 trial 전 5초 카운트다운
2. sound1과 동시에 CSI·영상 기록 시작
3. 동적 라벨은 2.5/3.0/3.5초를 순환하며 action cue 재생
4. 10초 후 sound2와 함께 trial 종료
5. TX1/TX2/TX3별 CSI amplitude 시각화 PNG 자동 저장
6. 다음 trial까지 2초 휴식
7. 전체 반복 완료 후 sound3 재생

정적 라벨은 action cue가 없다. DANGER는 5회 후 자동으로 3분 휴식하며, 10회 완료 후 다음 DANGER 세트까지 10분 이상 쉬어야 한다.

수집 파일:

```text
collection_data/v2/
  SUBJECT/E01/SESSION/risk/label/source_trial_uid/
    *_csi.csv
    *_csi_visualization.png
    *_video.mp4
    *_video_timestamps.csv
    *_meta.json
    checksums.sha256
  manifests/trials.csv
```

CSI CSV에는 PC monotonic timestamp, sender MAC/ID, sequence number, firmware timestamp, RSSI, CSI 배열을 보존한다. `*_csi_visualization.png`에는 TX1/TX2/TX3의 평균 CSI amplitude가 같은 trial 기준으로 저장된다. metadata에는 cue, variant, 장비 배치, 자동 QC, CSI 시각화 생성 여부, 실제 이벤트 주석 필드를 저장한다.

## 6. 이벤트 주석

수집 영상에서 실제 행동 시작, 충돌, 행동 종료 시각을 확인한 뒤 기록한다. 충돌이 없는 SAFE/WARNING은 `--impact`를 생략한다.

```powershell
python scripts/annotate_trial.py PATH_TO_META_JSON `
  --actual-onset 2.9 `
  --impact 4.1 `
  --action-end 4.6 `
  --manual-qc ACCEPT
```

planned cue와 actual onset 차이가 0.5초를 넘거나 동작 정의를 위반하면 `REJECT`로 기록하고 재수집한다.
정적 라벨은 `--actual-onset`, `--action-end`를 생략하면 자동으로 `0.0`, `10.0`이 기록된다. DANGER는 `--impact`가 필수다.

## 7. 13-point pose teacher GT

한 trial:

```powershell
python scripts/extract_pose13.py PATH_TO_TRIAL_FOLDER --overlay
```

한 session 전체:

```powershell
python scripts/extract_pose13.py --root collection_data/v2/SUBJECT/E01/SESSION --overlay
```

출력은 머리, 양쪽 어깨·팔꿈치·손목·골반·무릎·발목의 world-coordinate 13-point pose와 validity mask다. `absence`는 reconstruction에서 자동 제외한다. pose valid ratio 95% 미만은 reconstruction 재수집 대상이며, CSI-영상 최근접 timestamp residual p95가 50 ms를 넘으면 REVIEW, 100 ms를 넘으면 재수집한다.

## 8. 진행률과 최종 QC

```powershell
python scripts/show_progress.py --subject SUBJECT_ID --environment E01
python scripts/validate_collection.py `
  --write-report collection_data/v2/qc_report.json
```

세부 행동·금지 행동·재수집 기준은 [전체 수집 매뉴얼](docs/collection_manual.md)을 따른다.
첨부 문서 안에 남아 있던 이전 숫자와 삭제 라벨의 처리 원칙은 [v2.0 구현 기준 메모](docs/implementation_notes.md)에 기록했다.

팀원별 명령:

- [ajh.md](ajh.md)
- [lmh.md](lmh.md)
- [mhw.md](mhw.md)
- [yja.md](yja.md)

## 안전

DANGER는 건강한 성인의 통제 하강 시뮬레이션으로만 수행한다. 실제 고령자에게 낙상 동작을 시키지 않는다. 머리·목 보호, 이중 매트, 안전요원, 고정 의자, 낮은 침대가 준비되지 않으면 `--safety-confirmed`를 사용하지 않는다. 통증, 어지럼증, 메스꺼움, 두통, 손목 또는 무릎 불편감이 있으면 즉시 중단한다.
