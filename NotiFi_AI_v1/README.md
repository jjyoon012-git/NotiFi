# NotiFi AI v1

NotiFi AI v1은 세 개의 Wi-Fi CSI 링크만으로 17개 행동과 3단계 위험도를 분류하고,
학습 데이터의 3D 동작 사전에서 CSI와 가장 가까운 움직임을 검색·시간 보정하여
SMPL 22관절 궤적을 출력하는 1차 배포 모델입니다.

모델 구조, 성능, 신규 설치, calibration, API, 코드 구성과 한계는
[NotiFi_AI_v1.md](NotiFi_AI_v1.md)에 정리되어 있습니다.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python scripts\verify_release.py
```

장치를 등록하고 ESP 수집 CSV로 calibration합니다.

```powershell
notifi-ai --registry runtime\devices register --config examples\device.example.json
notifi-ai --device cuda --registry runtime\devices calibrate `
  --device-id home-001 `
  --manifest examples\calibration_manifest.example.json
```

한 trial을 추론합니다.

```powershell
notifi-ai --device cuda --registry runtime\devices predict `
  --device-id home-001 `
  --csv data\query\csi.csv `
  --output outputs\prediction.npz `
  --json outputs\prediction.json
```

HTTP API는 `pip install -e ".[api]"` 후 다음과 같이 실행합니다.

```powershell
notifi-ai --device cuda --registry runtime\devices serve --host 0.0.0.0 --port 8000
```

## Verification

```powershell
python -m compileall -q notifi_ai scripts tests
python -m unittest discover -s tests -p "test_*.py"
python scripts\verify_release.py
```

모델 추론에는 영상, query 정답 행동, query GT pose를 입력하지 않습니다. 원시 CSI를
사용할 때는 등록된 장치의 calibration profile이 필수입니다.
