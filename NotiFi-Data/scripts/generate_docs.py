from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.labels import LABEL_SPECS, SUBJECTS, total_repeats


def command_for(spec) -> str:
    command = (
        f"python scripts/collect_dataset.py "
        f"--label {spec.label} --repeat {spec.repeats_per_environment}"
    )
    if spec.risk == "DANGER":
        command += " --safety-confirmed"
    return command


def team_doc(subject: str) -> str:
    lines = [
        f"# NotiFi v2.0 수집 명령서 - {subject}",
        "",
        f"- 피험자 ID: `{subject}`",
        "- 수집 환경: `E01`, `E02`, `E03`",
        f"- 환경별 목표: `{total_repeats()}` trials",
        f"- 개인 전체 목표: `{total_repeats() * 3}` trials",
        "- 모든 trial: 10초",
        "- 기본 흐름: sound1(시작) → action cue(동적 라벨만) → sound2(종료) → 2초 휴식",
        "- 모든 라벨을 각 환경에서 아래 횟수만큼 반복한다.",
        "",
        "## 1. 환경별 세션 시작",
        "",
        "PowerShell에서 아래 placeholder를 실제 값으로 바꾼다.",
        "",
        "```powershell",
        "cd C:\\PATH\\TO\\NotiFi-Data",
        ".\\.venv\\Scripts\\Activate.ps1",
        (
            f'python scripts/create_session.py --subject {subject} '
            "--environment E01 --session YYYYMMDD_AM01 --port COM_PORT "
            "--camera 0 --firmware-commit FIRMWARE_COMMIT --device-config K0_P0"
            ),
            "python scripts/check_tx_links.py COM_PORT --rate 30",
            "python scripts/check_camera_source.py",
            "python scripts/preview_camera.py --camera 0",
            "```",
            "",
            "macOS에서 iPhone/iPad/Continuity Camera가 감지되면 카메라 확인과 수집이 중단된다. 노트북 내장 카메라만 남긴 뒤 다시 실행한다.",
            "",
            "`E02`, `E03`에서는 `--environment`와 새 `--session` 값만 바꿔 같은 절차를 반복한다.",
        "같은 환경에서도 60회 또는 35분에 도달하면 새 `--session`을 만든다. 진행률과 trial 번호는 manifest 기준으로 이어진다.",
        "",
        "권장 5-session 묶음(각 환경에서 반복):",
        "",
        "| Session | 비낙상 라벨 | DANGER | 합계 |",
        "| --- | --- | --- | ---: |",
        "| A | stand_to_lie_normal 30 + unstable_walking 20 | fall_from_standing 10 | 60 |",
        "| B | stumble_recover 30 + lying_still 18 | fall_while_walking 10 | 58 |",
        "| C | bed_exit_failed 25 + walking 24 | fall_collapse 10 | 59 |",
        "| D | lie_to_stand 18 + standing_still 12 + sitting_still 12 | bed_fall 10 | 52 |",
        "| E | absence 12 + sit_to_stand 12 + stand_to_sit 12 | chair_fall 10 | 46 |",
        "",
        "각 Session의 DANGER 10회는 마지막에 수행하고, 완료 후 10분 이상 쉰 다음 새 session을 시작한다.",
        "",
        "## 2. 라벨별 수집 명령",
        "",
    ]
    current_risk = None
    for spec in LABEL_SPECS:
        if spec.risk != current_risk:
            current_risk = spec.risk
            lines.extend((f"### {current_risk}", ""))
        lines.extend(
            (
                f"#### {spec.id} `{spec.label}` - {spec.korean} ({spec.repeats_per_environment}회)",
                "",
                f"시작 자세: {spec.start_pose}",
                "",
                "```powershell",
                command_for(spec),
                "```",
                "",
            )
        )
    lines.extend(
        (
            "## 3. 진행률 및 세션 후 처리",
            "",
            "```powershell",
            f"python scripts/show_progress.py --subject {subject} --environment E01",
            (
                "python scripts/extract_pose13.py --root "
                f"collection_data/v2/{subject}/E01/SESSION_ID --overlay"
            ),
            "python scripts/validate_collection.py --write-report collection_data/v2/qc_report.json",
            "```",
            "",
            "세부 동작 순서와 재수집 기준은 [수집 매뉴얼](docs/collection_manual.md)을 따른다.",
            "DANGER는 건강한 성인의 통제 하강으로만 수행하며 안전요원과 이중 매트가 없으면 실행하지 않는다.",
            "",
        )
    )
    return "\n".join(lines)


def manual_doc() -> str:
    lines = [
        "# NotiFi v2.0 데이터셋 수집 매뉴얼",
        "",
        "기준 문서: `NotiFi 데이터셋 수집 계획서 v2.0 (2026-07-22)`",
        "",
        "## 공통 규격",
        "",
        "- 4명(`ajh`, `lmh`, `mhw`, `yja`) 모두 동일한 3개 환경(`E01`, `E02`, `E03`)에서 수집한다.",
        "- 환경별 275회, 개인별 825회, 전체 3,300 accepted trials가 기본 목표다.",
        "- 모든 trial은 10초다.",
        "- 동적 행동 cue는 trial 번호에 따라 2.5초, 3.0초, 3.5초를 순환한다.",
        "- 한 trial에는 행동을 정확히 한 번만 수행한다.",
        "- core 수집에서는 배경 조건을 별도 변수로 나누지 않는다.",
        "- 원본 CSI, 영상, 프레임 timestamp, metadata, checksum을 같은 trial 폴더에 저장한다.",
        "- CSI와 영상은 같은 PC monotonic clock과 같은 `trial_start_monotonic_ns` 기준으로 동시에 기록한다.",
        "- 수집 후 actual onset, impact, action end와 수동 QC를 기록한다.",
        "",
        "## 장비 배치",
        "",
        "- RX: 활동 중심 정면, 높이 0.80 m, USB로 노트북에 연결",
        "- TX1: RX 맞은편, RX와 1.50 m, 높이 0.80 m",
        "- TX2: 활동 중심 측면, 높이 1.45 m",
        "- TX3: 활동 중심 대각 바닥 측, 높이 0.35 m",
        "- 네 보드 모두 세로 편파, 외장 안테나와 케이블 방향을 고정",
        "- 카메라: 활동 중심을 향하고 전신, 매트, 의자, 침대가 모두 프레임 안에 들어오도록 고정",
        "- 카메라는 노트북 내장 카메라만 사용한다. iPhone/iPad/Continuity Camera가 감지되면 수집을 시작하지 않는다.",
        "- 세 환경에서 동일한 로컬 좌표와 가구 배치를 재현하고 setup 사진과 실측값을 남긴다.",
        "- session 중 보드 위치 2 cm 또는 각도 3도 이상 변하면 중단하고 새 device_config_id를 사용한다.",
        "",
        "## 소리와 시간",
        "",
        "- sound1: 10초 trial의 실제 기록 시작",
        "- action cue: 동적 행동을 한 번 수행하는 시점. 정적 라벨에는 울리지 않음",
        "- sound2: trial 기록 종료",
        "- 기본 반복 간 휴식: 2초",
        "- sound3: 명령에 포함된 모든 반복 완료",
        "- DANGER 5회 후 3분, 10회 후 10분 이상 휴식",
        "",
        "## 라벨별 상세 매뉴얼",
        "",
    ]
    for spec in LABEL_SPECS:
        lines.extend(
            (
                f"### {spec.id} `{spec.label}` - {spec.korean}",
                "",
                f"- Risk: `{spec.risk}`",
                f"- Context: `{spec.context}`",
                f"- 환경별 반복: `{spec.repeats_per_environment}`회",
                f"- 시작 자세: {spec.start_pose}",
                f"- 종료 자세: {spec.end_pose}",
                f"- 허용 variant: {', '.join(spec.variants) if spec.variants else '고정'}",
                "",
                "| 시간 | 수행 내용 |",
                "| --- | --- |",
            )
        )
        for phase, action in spec.timeline:
            lines.append(f"| `{phase}` | {action} |")
        lines.extend(
            (
                "",
                f"- 금지 행동: {spec.prohibited}",
                f"- 재수집 기준: {spec.recollect}",
                "",
            )
        )
    lines.extend(
        (
            "## 자동 QC",
            "",
            "- CSI 파일이 없거나 0 byte면 재수집한다.",
            "- TX1/TX2/TX3 중 한 링크라도 0 frame이면 재수집한다.",
            "- TX별 수신량이 낮아도 0 frame만 아니면 자동 실패로 처리하지 않는다. 수집 품질은 `*_csi_visualization.png`에서 확인한다.",
            "- 영상이 없거나 손상되면 재수집한다.",
            "- iPhone/iPad/Continuity Camera가 감지되면 수집을 시작하지 않는다.",
            "- 전신이 잘리거나 pose valid frame ratio가 95% 미만이면 reconstruction 대상에서 제외한다.",
            "- CSI-영상 sync residual p95가 50 ms를 넘으면 REVIEW, 100 ms를 넘으면 재수집한다.",
            "- planned cue와 actual onset 차이가 0.5초를 넘으면 재수집한다.",
            "- 정의와 다른 outcome이면 재수집한다.",
            "- 잘못 수집된 trial은 해당 trial 폴더를 삭제한 뒤 같은 명령어를 다시 실행한다. 삭제된 trial 번호는 manifest에 남아 있어도 다시 사용된다.",
            "",
            "## 안전",
            "",
            "- 실제 고령자는 DANGER 동작을 수행하지 않는다.",
            "- DANGER는 건강한 성인의 통제 하강 시뮬레이션으로만 수행한다.",
            "- 머리와 목 보호, 이중 매트, 안전요원, 고정 의자, 낮은 침대가 필수다.",
            "- 통증, 어지럼증, 메스꺼움, 두통, 손목 또는 무릎 불편감이 있으면 즉시 중단한다.",
            "- session당 accepted 최대 60회, DANGER 최대 10회, 최대 35분을 지킨다.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    config_dir = ROOT / "config"
    docs_dir = ROOT / "docs"
    config_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    payload = []
    for spec in LABEL_SPECS:
        item = asdict(spec)
        item["timeline"] = [list(value) for value in spec.timeline]
        item["variants"] = list(spec.variants)
        payload.append(item)
    (config_dir / "labels_v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (docs_dir / "collection_manual.md").write_text(manual_doc(), encoding="utf-8")
    for subject in SUBJECTS:
        (ROOT / f"{subject}.md").write_text(team_doc(subject), encoding="utf-8")
    print("[OK] labels_v2.json, collection_manual.md, and four team guides generated.")


if __name__ == "__main__":
    main()
