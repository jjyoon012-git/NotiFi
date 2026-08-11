"""FastAPI service for device registration, calibration, and inference."""

from __future__ import annotations

import threading
from pathlib import Path

from .io import load_calibration_npz, load_query_npz
from .model import NotiFiAIv1
from .registry import DeviceRegistry
from .schemas import DeviceConfig


def create_app(
    artifact_dir: str | Path | None = None,
    registry_root: str | Path = "runtime/devices",
    device: str | None = None,
):
    try:
        from fastapi import FastAPI, File, HTTPException
    except ImportError as exc:
        raise RuntimeError("install with: pip install -e .[api]") from exc

    app = FastAPI(title="NotiFi AI v1", version="1.0.0")
    model = NotiFiAIv1(artifact_dir, device)
    registry = DeviceRegistry(registry_root)
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", **model.describe()}

    @app.get("/v1/devices")
    def list_devices():
        return {"devices": registry.list_devices()}

    @app.post("/v1/devices/register")
    def register_device(payload: dict):
        try:
            config = DeviceConfig(**payload)
            path = registry.register(config)
            return {"device_id": config.device_id, "path": str(path)}
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/devices/{device_id}/calibrate")
    async def calibrate(device_id: str, file: bytes = File(...)):
        try:
            registry.load_device(device_id)
            absence, support = load_calibration_npz(file)
            with lock:
                profile = model.fit_calibration(device_id, absence, support)
            path = registry.save_calibration(profile)
            return {"path": str(path), **profile.summary()}
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/devices/{device_id}/predict")
    async def predict(
        device_id: str,
        file: bytes = File(...),
        include_pose: bool = False,
    ):
        try:
            profile = registry.load_calibration(device_id)
            csi, mask = load_query_npz(file)
            with lock:
                result = model.predict(csi, mask, profile)
            return result.to_dict(include_pose=include_pose)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
