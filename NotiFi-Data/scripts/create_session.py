from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notifi_collection.labels import ENVIRONMENTS, SUBJECTS
from notifi_collection.session import DEFAULT_CONFIG_PATH, SessionConfig, save_session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the local NotiFi v2.0 session configuration."
    )
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--environment", required=True, choices=ENVIRONMENTS)
    parser.add_argument("--session", required=True, help="Example: 20260723_AM01")
    parser.add_argument("--port", required=True, help="RX serial port, e.g. COM4")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--firmware-commit", required=True)
    parser.add_argument("--device-config", required=True)
    parser.add_argument("--channel", type=int, default=11)
    parser.add_argument("--packet-rate", type=int, default=30)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    config = SessionConfig(
        subject=args.subject,
        environment=args.environment,
        session_id=args.session,
        port=args.port,
        camera_index=args.camera,
        firmware_commit=args.firmware_commit,
        device_config_id=args.device_config,
        channel=args.channel,
        packet_rate=args.packet_rate,
        baud=args.baud,
        camera_width=args.width,
        camera_height=args.height,
        camera_fps=args.fps,
    )
    save_session(config, args.out)
    print(f"[OK] session config saved: {args.out.resolve()}")
    print(
        f"subject={config.subject} environment={config.environment} "
        f"session={config.session_id} port={config.port} camera={config.camera_index}"
    )


if __name__ == "__main__":
    main()
