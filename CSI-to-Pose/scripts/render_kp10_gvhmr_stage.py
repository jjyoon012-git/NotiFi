"""Render KP10 payloads as clean stick views and official GVHMR SMPL scenes.

Run this inside the WSL GVHMR environment. The mesh mode uses GVHMR's
PyTorch3D Renderer, checkerboard floor, static perspective camera, and lights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


WIDTH, HEIGHT = 1280, 720
HALF = WIDTH // 2
HEADER = 72
FONT = cv2.FONT_HERSHEY_SIMPLEX
EDGES = (
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20),
    (19, 21),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gvhmr-root", type=Path, default=Path.home() / "GVHMR")
    parser.add_argument(
        "--body-models", type=Path, default=None,
        help="GVHMR body_models root; defaults below --gvhmr-root",
    )
    parser.add_argument(
        "--smpl-model", type=Path, default=None,
        help="deprecated compatibility argument; topology comes from smplx",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--match", default="", help="render payload names containing this text")
    parser.add_argument("--only", choices=("stickman", "gvhmr", "both"), default="both")
    return parser.parse_args()


class PanelProjector:
    def __init__(self, trajectory: np.ndarray, x_offset: int):
        points = self.raw(trajectory.reshape(-1, 3))
        low = np.percentile(points, 0.5, axis=0)
        high = np.percentile(points, 99.5, axis=0)
        center = (low + high) / 2
        span = np.maximum(high - low, (0.8, 1.3)) * 1.25
        self.center = center
        self.scale = min(530.0 / span[0], 540.0 / span[1])
        self.x_offset = x_offset

    @staticmethod
    def raw(points: np.ndarray) -> np.ndarray:
        x = 0.86 * points[..., 0] - 0.50 * points[..., 2]
        y = points[..., 1] + 0.16 * points[..., 0] + 0.20 * points[..., 2]
        return np.stack((x, y), axis=-1)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        raw = self.raw(points)
        x = self.x_offset + HALF / 2 + (raw[..., 0] - self.center[0]) * self.scale
        y = 390.0 - (raw[..., 1] - self.center[1]) * self.scale
        return np.stack((x, y), axis=-1)


def decorate(frame: np.ndarray, metadata: dict, mode: str, index: int) -> None:
    cv2.rectangle(frame, (0, 0), (WIDTH, HEADER), (18, 21, 27), -1)
    cv2.putText(
        frame,
        f"{metadata['scenario_id']}  {metadata['detail_label']}  "
        f"{metadata['trial_id']}  t={index / 30.0:4.1f}s",
        (24, 31), FONT, 0.72, (245, 247, 250), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "GVHMR ground truth", (170, 61), FONT, 0.56,
        (210, 220, 232), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "CSI-only KP10 prediction", (780, 61), FONT, 0.56,
        (80, 155, 250), 1, cv2.LINE_AA,
    )
    if mode == "stickman":
        cv2.line(frame, (HALF, HEADER), (HALF, HEIGHT), (196, 202, 210), 1)


def draw_stick(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    points = np.rint(points).astype(np.int32)
    for left, right in EDGES:
        cv2.line(frame, tuple(points[left]), tuple(points[right]), color, 6, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, tuple(point), 6, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), 6, color, 2, cv2.LINE_AA)


def render_stickman(payload: dict, destination: Path, fps: float) -> None:
    target = payload["target_absolute"]
    predicted = payload["predicted_absolute"]
    left = PanelProjector(target, 0)
    right = PanelProjector(predicted, HALF)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    preview = int(payload["preview_frame"])
    for index in range(len(target)):
        frame = np.full((HEIGHT, WIDTH, 3), (242, 244, 247), np.uint8)
        for x in (80, 200, 320, 440, 560, 720, 840, 960, 1080, 1200):
            cv2.line(frame, (x, 650), (x + 55, 560), (220, 224, 230), 1)
        cv2.line(frame, (20, 650), (WIDTH - 20, 650), (185, 190, 198), 2)
        draw_stick(frame, left(target[index]), (70, 165, 105))
        draw_stick(frame, right(predicted[index]), (45, 105, 230))
        decorate(frame, payload, "stickman", index)
        writer.write(frame)
        if index == preview:
            cv2.imwrite(str(destination.with_suffix(".png")), frame)
    writer.release()


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source, axis=1, keepdims=True).clip(1e-8)
    target = target / np.linalg.norm(target, axis=1, keepdims=True).clip(1e-8)
    u, _, vh = np.linalg.svd(source.T @ target)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    return rotation.astype(np.float32)


def _shortest_arc_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the no-twist rotation that maps one direction onto another."""
    source = source / np.linalg.norm(source).clip(1e-8)
    target = target / np.linalg.norm(target).clip(1e-8)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-7:
        if cosine > 0:
            return np.eye(3, dtype=np.float32)
        axis_seed = np.eye(3, dtype=np.float32)[np.argmin(np.abs(source))]
        axis = np.cross(source, axis_seed)
        axis /= np.linalg.norm(axis).clip(1e-8)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ], dtype=np.float32)
    return (
        np.eye(3, dtype=np.float32)
        + skew
        + (skew @ skew) * ((1.0 - cosine) / (sine * sine))
    ).astype(np.float32)


class SmplPoseFitter:
    """Convert 22 joint directions to proper local SMPL rotations."""

    def __init__(self, model, matrix_to_axis_angle):
        self.model = model
        self.matrix_to_axis_angle = matrix_to_axis_angle
        with torch.no_grad():
            rest = torch.einsum(
                "jv,vc->jc", model.J_regressor, model.v_template
            )
        self.rest_joints = rest[:24].detach().cpu().numpy().astype(np.float32)
        self.parents = model.parents[:24].detach().cpu().numpy().astype(np.int64)
        self.children = [
            np.flatnonzero(self.parents == joint) for joint in range(24)
        ]
        self.source_lengths = np.linalg.norm(
            self.rest_joints[1:22]
            - self.rest_joints[self.parents[1:22]],
            axis=1,
        ).clip(1e-6)
        self.faces = np.asarray(model.faces, dtype=np.int64)

    def _local_rotations(self, target: np.ndarray) -> np.ndarray:
        frames = len(target)
        local_rotation = np.repeat(
            np.eye(3, dtype=np.float32)[None, None], frames * 24, axis=0
        ).reshape(frames, 24, 3, 3)
        global_rotation = np.repeat(
            np.eye(3, dtype=np.float32)[None, None], frames * 24, axis=0
        ).reshape(frames, 24, 3, 3)
        for frame in range(frames):
            for joint in range(24):
                parent = self.parents[joint]
                parent_rotation = (
                    np.eye(3, dtype=np.float32)
                    if parent < 0 else global_rotation[frame, parent]
                )
                children = self.children[joint]
                children = children[children < 22]
                if len(children):
                    source = self.rest_joints[children] - self.rest_joints[joint]
                    destination = target[frame, children] - target[frame, joint]
                    destination = np.einsum(
                        "ij,nj->ni", parent_rotation.T, destination
                    )
                    if len(children) == 1:
                        local_rotation[frame, joint] = _shortest_arc_rotation(
                            source[0], destination[0]
                        )
                    else:
                        local_rotation[frame, joint] = _align_vectors(
                            source, destination
                        )
                global_rotation[frame, joint] = (
                    parent_rotation @ local_rotation[frame, joint]
                )
        return local_rotation

    @torch.no_grad()
    def fit_sequence(
        self, target: np.ndarray, fixed_scale: float | None = None,
    ) -> tuple[np.ndarray, float]:
        if fixed_scale is None:
            target_lengths = np.linalg.norm(
                target[:, 1:22] - target[:, self.parents[1:22]], axis=-1
            )
            fixed_scale = float(np.median(
                target_lengths / self.source_lengths[None]
            ))
        local = torch.from_numpy(self._local_rotations(target)).cuda()
        axis_angle = self.matrix_to_axis_angle(
            local.reshape(-1, 3, 3)
        ).reshape(len(target), 24, 3)
        output = self.model(
            global_orient=axis_angle[:, 0],
            body_pose=axis_angle[:, 1:].reshape(len(target), 69),
            betas=torch.zeros(
                (len(target), 10), dtype=torch.float32, device="cuda"
            ),
        )
        root = output.joints[:, :1]
        vertices = (
            torch.from_numpy(target[:, :1]).cuda()
            + float(fixed_scale) * (output.vertices - root)
        )
        return vertices.detach().cpu().numpy().astype(np.float32), fixed_scale


def fit_surfaces(payload: dict, fitter) -> tuple[np.ndarray, np.ndarray]:
    target = payload["target_absolute"]
    predicted = payload["predicted_absolute"]
    target_vertices, body_scale = fitter.fit_sequence(target)
    predicted_vertices, _ = fitter.fit_sequence(predicted, body_scale)
    floor = float(np.percentile(target_vertices[..., 1], 0.1))
    target_vertices[..., 1] -= floor
    predicted_vertices[..., 1] -= floor
    span_x = float(
        max(np.ptp(target_vertices[..., 0]), np.ptp(predicted_vertices[..., 0]))
    )
    spacing = max(1.55, span_x + 0.75)
    target_vertices[..., 0] -= spacing / 2
    predicted_vertices[..., 0] += spacing / 2
    return target_vertices.astype(np.float32), predicted_vertices.astype(np.float32)


def render_gvhmr(payload: dict, destination: Path, fps: float, fitter, renderer_api) -> None:
    Renderer, get_global_cameras_static, get_ground_params_from_points = renderer_api
    target, predicted = fit_surfaces(payload, fitter)
    combined_np = np.concatenate((target, predicted), axis=1)
    combined = torch.from_numpy(combined_np).cuda()
    renderer = Renderer(
        WIDTH, HEIGHT, focal_length=900.0, device="cuda",
        faces=fitter.faces, bin_size=64,
    )
    renderer.renderer.rasterizer.raster_settings.max_faces_per_bin = 10000
    length, center_x, center_z = get_ground_params_from_points(
        combined[:, 0], combined
    )
    renderer.set_ground(max(float(length) * 1.35, 5.0), center_x, center_z)
    rotation, translation, lights = get_global_cameras_static(
        combined.cpu(), beta=3.20, cam_height_degree=15,
        target_center_height=0.9, vec_rot=0, device="cuda",
    )
    cameras = renderer.create_camera(rotation[0], translation[0])
    colors = torch.tensor(
        ((0.74, 0.79, 0.86), (0.12, 0.43, 0.88)),
        dtype=torch.float32, device="cuda",
    )
    target_t = torch.from_numpy(target).cuda()
    predicted_t = torch.from_numpy(predicted).cuda()
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    preview = int(payload["preview_frame"])
    for index in range(len(target)):
        vertices = torch.stack((target_t[index], predicted_t[index]))
        rgb = renderer.render_with_ground(
            vertices, colors, cameras, lights
        )
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        decorate(frame, payload, "gvhmr", index)
        writer.write(frame)
        if index == preview:
            cv2.imwrite(str(destination.with_suffix(".png")), frame)
    writer.release()


def load_payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def scalar_text(value: np.ndarray) -> str:
    return str(value.item())


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.gvhmr_root))
    from hmr4d.utils.vis.renderer import (
        Renderer, get_global_cameras_static, get_ground_params_from_points,
    )
    from pytorch3d.transforms import matrix_to_axis_angle
    from smplx.body_models import SMPL, Struct

    files = sorted(args.payload_dir.glob("*.npz"))
    if args.match:
        files = [path for path in files if args.match in path.name]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"no payloads in {args.payload_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stickman").mkdir(exist_ok=True)
    (args.out / "gvhmr").mkdir(exist_ok=True)
    body_models = args.body_models or (
        args.gvhmr_root / "inputs" / "checkpoints" / "body_models"
    )
    model_path = body_models / "smpl" / "SMPL_NEUTRAL.npz"
    with np.load(model_path, allow_pickle=True) as model_data:
        data_struct = Struct(**{
            key: model_data[key] for key in model_data.files
        })
    smpl_model = SMPL(
        str(model_path.parent),
        data_struct=data_struct,
        gender="neutral",
        batch_size=1,
        create_betas=False,
        create_global_orient=False,
        create_body_pose=False,
        create_transl=False,
    ).cuda().eval()
    fitter = SmplPoseFitter(smpl_model, matrix_to_axis_angle)
    renderer_api = (
        Renderer, get_global_cameras_static, get_ground_params_from_points,
    )
    results = []
    for count, path in enumerate(files, 1):
        payload = load_payload(path)
        metadata = {
            "scenario_id": scalar_text(payload["scenario_id"]),
            "detail_label": scalar_text(payload["detail_label"]),
            "trial_id": scalar_text(payload["trial_id"]),
        }
        payload.update(metadata)
        stem = path.stem
        print(f"[{count:02d}/{len(files)}] {stem}", flush=True)
        if args.only in ("stickman", "both"):
            output = args.out / "stickman" / f"{stem}__stickman_stage.mp4"
            render_stickman(payload, output, args.fps)
            results.append(str(output))
            print(f"  wrote {output}", flush=True)
        if args.only in ("gvhmr", "both"):
            output = args.out / "gvhmr" / f"{stem}__gvhmr_stage.mp4"
            render_gvhmr(payload, output, args.fps, fitter, renderer_api)
            results.append(str(output))
            print(f"  wrote {output}", flush=True)
    (args.out / "render_manifest.json").write_text(
        json.dumps({
            "renderer": "GVHMR hmr4d PyTorch3D Renderer",
            "layout": "GT and CSI prediction side-by-side on one 3D stage",
            "source_video_in_frame": False,
            "files": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.out), "videos": len(results)}))


if __name__ == "__main__":
    main()
