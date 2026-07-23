# NotiFi v2.0 수집 명령서 - lmh

- 피험자 ID: `lmh`
- 수집 환경: `E01`, `E02`, `E03`
- 환경별 목표: `275` trials
- 개인 전체 목표: `825` trials
- 모든 trial: 10초
- 기본 흐름: sound1(시작) → action cue(동적 라벨만) → sound2(종료) → 2초 휴식
- 모든 라벨을 각 환경에서 아래 횟수만큼 반복한다.

## 1. 환경별 세션 시작

PowerShell에서 아래 placeholder를 실제 값으로 바꾼다.

```powershell
cd C:\PATH\TO\NotiFi-Data
.\.venv\Scripts\Activate.ps1
python scripts/create_session.py --subject lmh --environment E01 --session YYYYMMDD_AM01 --port COM_PORT --camera 0 --firmware-commit FIRMWARE_COMMIT --device-config K0_P0
python scripts/check_tx_links.py COM_PORT --rate 30
python scripts/preview_camera.py --camera 0
```

`E02`, `E03`에서는 `--environment`와 새 `--session` 값만 바꿔 같은 절차를 반복한다.
같은 환경에서도 60회 또는 35분에 도달하면 새 `--session`을 만든다. 진행률과 trial 번호는 manifest 기준으로 이어진다.

권장 5-session 묶음(각 환경에서 반복):

| Session | 비낙상 라벨 | DANGER | 합계 |
| --- | --- | --- | ---: |
| A | stand_to_lie_normal 30 + unstable_walking 20 | fall_from_standing 10 | 60 |
| B | stumble_recover 30 + lying_still 18 | fall_while_walking 10 | 58 |
| C | bed_exit_failed 25 + walking 24 | fall_collapse 10 | 59 |
| D | lie_to_stand 18 + standing_still 12 + sitting_still 12 | bed_fall 10 | 52 |
| E | absence 12 + sit_to_stand 12 + stand_to_sit 12 | chair_fall 10 | 46 |

각 Session의 DANGER 10회는 마지막에 수행하고, 완료 후 10분 이상 쉰 다음 새 session을 시작한다.

## 2. 라벨별 수집 명령

### SAFE

#### S01 `walking` - 직선 걷기 (24회)

시작 자세: 1.5 m 보행 경로의 시작선 A에서 종료선 B를 바라보고 선다. 양팔은 몸통 옆에 자연스럽게 둔다.

```powershell
python scripts/collect_dataset.py --label walking --repeat 24
```

#### S02 `standing_still` - 정지 서기 (12회)

시작 자세: 활동 중심 C에서 카메라를 정면으로 본다. 발뒤꿈치 간격 20 cm, 팔은 몸통 옆, 손바닥은 허벅지 쪽이다.

```powershell
python scripts/collect_dataset.py --label standing_still --repeat 12
```

#### S03 `sitting_still` - 의자 정지 앉기 (12회)

시작 자세: 고정 의자 좌판 중앙에 앉는다. 양발은 바닥 발 표시 위, 무릎 90도, 손바닥은 허벅지 위, 등은 등받이에 가볍게 둔다.

```powershell
python scripts/collect_dataset.py --label sitting_still --repeat 12
```

#### S04 `lying_still` - 침대 정지 눕기 (18회)

시작 자세: 침대 중앙에 바로 눕는다. 머리는 카메라에서 먼 쪽, 발은 카메라 쪽이다. 팔은 몸통에서 10 cm, 다리는 곧게 편다.

```powershell
python scripts/collect_dataset.py --label lying_still --repeat 18
```

#### S05 `lie_to_stand` - 침대에서 정상적으로 일어나기 (18회)

시작 자세: S04와 동일한 바로 누운 자세. 지정된 한쪽 침대 이탈 방향을 모든 trial에서 유지한다.

```powershell
python scripts/collect_dataset.py --label lie_to_stand --repeat 18
```

#### S06 `stand_to_lie_normal` - 침대에 정상적으로 눕기 (30회)

시작 자세: 침대 긴 변 중앙 앞 발 표시에서 침대를 향해 선다. 허벅지와 침대 가장자리 간격은 10 cm다.

```powershell
python scripts/collect_dataset.py --label stand_to_lie_normal --repeat 30
```

#### S07 `absence` - 부재 (12회)

시작 자세: 피험자, 운영자, 안전요원 모두 카메라와 활동 영역 밖으로 완전히 이동한다. 문은 세션 기준 상태로 고정한다.

```powershell
python scripts/collect_dataset.py --label absence --repeat 12
```

#### S08 `sit_to_stand` - 의자에서 정상적으로 일어서기 (12회)

시작 자세: S03과 동일한 착석 자세. 양발은 무릎 바로 아래 발 표시 위에 둔다.

```powershell
python scripts/collect_dataset.py --label sit_to_stand --repeat 12
```

#### S09 `stand_to_sit` - 의자에 정상적으로 앉기 (12회)

시작 자세: 의자 앞 40 cm 발 표시에서 등을 의자에 향하고 선다.

```powershell
python scripts/collect_dataset.py --label stand_to_sit --repeat 12
```

### WARNING

#### W01 `unstable_walking` - 지속적 불안정 보행 (20회)

시작 자세: 보행 경로 시작선 A에서 종료선 B를 바라보고 선다.

```powershell
python scripts/collect_dataset.py --label unstable_walking --repeat 20
```

#### W02 `stumble_recover` - 발을 헛디딘 뒤 회복 (30회)

시작 자세: 보행 경로 시작선 A에서 종료선 B를 바라보고 선다. 실제 장애물은 두지 않고 C 지점에 평면 표식만 둔다.

```powershell
python scripts/collect_dataset.py --label stumble_recover --repeat 30
```

#### W03 `bed_exit_failed` - 침대에서 일어서려다 실패 후 다시 앉기 (25회)

시작 자세: S04와 동일한 바로 누운 자세. 이탈 측은 S05 및 D04와 동일하다.

```powershell
python scripts/collect_dataset.py --label bed_exit_failed --repeat 25
```

### DANGER

#### D01 `fall_from_standing` - 서 있는 상태에서 측면 낙상 (10회)

시작 자세: 이중 매트 중앙 C에서 선다. 지정 낙상 측에 안전요원이 대기하고 머리 쪽 매트는 이중으로 겹친다.

```powershell
python scripts/collect_dataset.py --label fall_from_standing --repeat 10 --safety-confirmed
```

#### D02 `fall_while_walking` - 걷다가 발을 헛디뎌 낙상 (10회)

시작 자세: 보행 경로 시작선 A에서 종료선 B를 바라보고 선다. C는 매트 중앙이며 실제 장애물은 두지 않는다.

```powershell
python scripts/collect_dataset.py --label fall_while_walking --repeat 10 --safety-confirmed
```

#### D03 `fall_collapse` - 다리에 힘이 풀리는 주저앉기형 낙상 (10회)

시작 자세: 이중 매트 중앙 C에서 양발을 어깨너비로 두고 선다.

```powershell
python scripts/collect_dataset.py --label fall_collapse --repeat 10 --safety-confirmed
```

#### D04 `bed_fall` - 침대에서 일어난 직후 옆으로 낙상 (10회)

시작 자세: S04와 같은 바로 누운 자세. 침대 이탈 측 바닥 전체를 이중 매트로 덮고 안전요원이 지정 측에 선다.

```powershell
python scripts/collect_dataset.py --label bed_fall --repeat 10 --safety-confirmed
```

#### D05 `chair_fall` - 의자에 앉으려다 좌판을 놓쳐 낙상 (10회)

시작 자세: S09와 같은 위치에서 등을 의자에 향하고 선다. 지정 낙상 측 의자 옆을 이중 매트로 덮고 의자를 고정한다.

```powershell
python scripts/collect_dataset.py --label chair_fall --repeat 10 --safety-confirmed
```

## 3. 진행률 및 세션 후 처리

```powershell
python scripts/show_progress.py --subject lmh --environment E01
python scripts/extract_pose13.py --root collection_data/v2/lmh/E01/SESSION_ID --overlay
python scripts/validate_collection.py --write-report collection_data/v2/qc_report.json
```

세부 동작 순서와 재수집 기준은 [수집 매뉴얼](docs/collection_manual.md)을 따른다.
DANGER는 건강한 성인의 통제 하강으로만 수행하며 안전요원과 이중 매트가 없으면 실행하지 않는다.
