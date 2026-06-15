# NotiFi

NotiFi는 카메라나 웨어러블 없이 Wi-Fi CSI(Channel State Information)를 활용해 노인의 낙상, 호흡 이상, 무활동 가능성을 탐지하는 것을 목표로 하는 프로젝트입니다.

현재 단계는 모델 학습이 아니라, ESP32-C6 보드에서 CSI 데이터가 실제로 수집되는지 검증하는 초기 실험입니다.

## 현재 진행 단계

- Seeed Studio XIAO ESP32-C6 기반 보드 사용
- ESP-IDF 기반 개발
- ESP-CSI get-started 예제 사용
- 보드 2개 구성
  - `csi_send`: CSI 생성을 위한 송신 보드
  - `csi_recv`: CSI 데이터를 수신하고 `CSI_DATA` 로그 출력

## 작업 로그

| 날짜 | 내용 |
|---|---|
| [2026.06.16](logs/2026.06.16.md) | ESP-IDF 환경 복구, ESP32-C6 타깃 빌드, `csi_send`/`csi_recv` 플래시, `CSI_DATA` 수신 확인, 시각화 실행 |

## 최신 확인 상태

2026.06.16 기준:

- ESP-IDF v5.5.4 환경 확인
- `csi_send` 빌드 및 플래시 완료
- `csi_recv` 빌드 및 플래시 완료
- 수신 보드에서 `CSI_DATA` 로그 연속 출력 확인
- CSI 배열 길이 `256` 확인
- RSSI는 대략 `-85 ~ -92 dBm` 범위에서 확인

## 다음 작업

- `CSI_DATA`를 CSV로 저장
- `idle`, `hand_move`, `walk` 60초씩 수집
- RSSI, CSI 배열 길이, CSI mean absolute value 비교
- 움직임에 따른 CSI 변화가 재현되는지 확인

## 참고

현재 성공 기준은 낙상 감지나 호흡 추정이 아니라, **CSI 데이터가 실제로 들어오고 저장 가능한 상태인지 확인하는 것**입니다.
