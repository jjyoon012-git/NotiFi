"""Command-line interface for registration, calibration, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import (
    load_calibration_manifest,
    load_calibration_npz,
    load_query_npz,
)
from .model import NotiFiAIv1
from .registry import DeviceRegistry
from .schemas import DeviceConfig


def _model(args) -> NotiFiAIv1:
    return NotiFiAIv1(args.artifacts, args.device)


def main() -> int:
    parser = argparse.ArgumentParser(prog="notifi-ai")
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--registry", type=Path, default=Path("runtime/devices"))
    commands = parser.add_subparsers(dest="command", required=True)

    describe = commands.add_parser("describe")

    register = commands.add_parser("register")
    register.add_argument("--config", type=Path, required=True)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--device-id", required=True)
    calibration_input = calibrate.add_mutually_exclusive_group(required=True)
    calibration_input.add_argument("--input", type=Path)
    calibration_input.add_argument("--manifest", type=Path)

    predict = commands.add_parser("predict")
    predict.add_argument("--device-id", required=True)
    prediction_input = predict.add_mutually_exclusive_group(required=True)
    prediction_input.add_argument("--input", type=Path)
    prediction_input.add_argument("--csv", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--json", type=Path, default=None)

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    registry = DeviceRegistry(args.registry)
    if args.command == "register":
        config = DeviceConfig(
            **json.loads(args.config.read_text(encoding="utf-8"))
        )
        print(registry.register(config))
        return 0
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("install with: pip install -e .[api]") from exc
        from .api import create_app

        uvicorn.run(
            create_app(args.artifacts, args.registry, args.device),
            host=args.host,
            port=args.port,
        )
        return 0

    model = _model(args)
    if args.command == "describe":
        print(model.describe_json())
        return 0
    if args.command == "calibrate":
        registry.load_device(args.device_id)
        if args.manifest is not None:
            absence, support = load_calibration_manifest(args.manifest)
        else:
            absence, support = load_calibration_npz(args.input)
        profile = model.fit_calibration(args.device_id, absence, support)
        print(json.dumps(profile.summary(), indent=2, ensure_ascii=False))
        registry.save_calibration(profile)
        return 0
    if args.command == "predict":
        profile = registry.load_calibration(args.device_id)
        if args.csv is not None:
            result = model.predict_csv(args.csv, profile)
        else:
            csi, mask = load_query_npz(args.input)
            result = model.predict(csi, mask, profile)
        result.save_npz(args.output)
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(result.to_dict(False), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(result.to_dict(False), indent=2, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
