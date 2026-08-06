"""Render representative seen V13S trials as stick and SMPL-surface overlays.

The source seen GT stores GVHMR-22 joints but not the original camera or SMPL
parameters.  Therefore the original video is shown beside a metric 3D panel;
the skeleton/mesh is not projected onto video pixels.  The mesh view fits a
neutral SMPL surface to the available 22-joint trajectories for visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from ..dataio.dataset import build_datasets
from ..trainer import set_seed
from .diagnose_observability import pose_only
from .evaluate_sealed import smooth_valid
from .evaluate_v12_final import _read_locked, build_locked_model


WIDTH, HEIGHT = 1280, 720
HALF = WIDTH // 2
GT_COLOR = (70, 215, 115)
PRED_COLOR = (70, 90, 245)
GT_MESH = (230, 190, 70)
PRED_MESH = (85, 100, 245)
WHITE = (245, 245, 245)
MUTED = (175, 180, 190)
BG = (19, 21, 25)
PANEL_BG = (246, 247, 249)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-8)
    target = target / max(np.linalg.norm(target), 1e-8)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-7:
        if cosine > 0:
            return np.eye(3, dtype=np.float32)
        axis = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        if abs(source @ axis) > 0.9:
            axis = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        axis -= source * (source @ axis)
        axis /= max(np.linalg.norm(axis), 1e-8)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    axis = cross / sine
    skew = np.array(
        ((0.0, -axis[2], axis[1]),
         (axis[2], 0.0, -axis[0]),
         (-axis[1], axis[0], 0.0)),
        dtype=np.float32,
    )
    return (
        np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)
    ).astype(np.float32)


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 1:
        return _rotation_between(source[0], target[0])
    source = source / np.linalg.norm(source, axis=1, keepdims=True).clip(1e-8)
    target = target / np.linalg.norm(target, axis=1, keepdims=True).clip(1e-8)
    u, _, vh = np.linalg.svd(source.T @ target)
    row_rotation = u @ vh
    if np.linalg.det(row_rotation) < 0:
        u[:, -1] *= -1
        row_rotation = u @ vh
    return row_rotation.T.astype(np.float32)


class SmplSurfaceFitter:
    """Fast neutral-SMPL surface fit driven only by target joint positions."""

    def __init__(self, model_path: Path, face_stride: int = 8):
        with np.load(model_path, allow_pickle=False) as payload:
            self.vertices = payload["v_template"].astype(np.float32)
            self.joints = payload["J"].astype(np.float32)
            self.weights = payload["weights"].astype(np.float32)
            self.faces = payload["f"].astype(np.int32)
        self.edge_stride = max(int(face_stride), 1)
        parents = np.asarray(C.JOINT_PARENTS, dtype=np.int64)
        self.parents = np.concatenate((parents, np.array((20, 21))))
        self.children = [
            np.flatnonzero(self.parents == joint)
            for joint in range(len(self.parents))
        ]
        source_lengths = np.linalg.norm(
            self.joints[1:22] - self.joints[parents[1:]], axis=1
        )
        self.source_lengths = source_lengths.clip(1e-5)

    def fit(self, target22: np.ndarray) -> np.ndarray:
        target_lengths = np.linalg.norm(
            target22[1:] - target22[np.asarray(C.JOINT_PARENTS[1:])],
            axis=1,
        )
        scale = float(np.median(target_lengths / self.source_lengths))
        rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 24, axis=0)
        for joint in range(22):
            children = self.children[joint]
            children = children[children < 22]
            if len(children):
                source = self.joints[children] - self.joints[joint]
                target = target22[children] - target22[joint]
                rotations[joint] = _align_vectors(source, target)
            elif self.parents[joint] >= 0:
                rotations[joint] = rotations[self.parents[joint]]
        rotations[22] = rotations[20]
        rotations[23] = rotations[21]

        target = np.empty((24, 3), dtype=np.float32)
        target[:22] = target22
        for joint in (22, 23):
            parent = self.parents[joint]
            target[joint] = (
                target[parent]
                + rotations[parent] @ (
                    scale * (self.joints[joint] - self.joints[parent])
                )
            )
        translation = target - scale * np.einsum(
            "jab,jb->ja", rotations, self.joints
        )
        blended_rotation = np.einsum(
            "vj,jab->vab", self.weights, rotations, optimize=True
        )
        blended_translation = self.weights @ translation
        return (
            scale * np.einsum(
                "vab,vb->va", blended_rotation, self.vertices,
                optimize=True,
            )
            + blended_translation
        ).astype(np.float32)


class IsometricProjector:
    def __init__(self, trajectories: np.ndarray):
        projected = self._raw(trajectories.reshape(-1, 3))
        low = np.nanpercentile(projected, 0.5, axis=0)
        high = np.nanpercentile(projected, 99.5, axis=0)
        center = (low + high) / 2
        span = np.maximum(high - low, (0.8, 1.2)) * 1.22
        self.center = center
        self.scale = min(560.0 / span[0], 520.0 / span[1])

    @staticmethod
    def _raw(points: np.ndarray) -> np.ndarray:
        horizontal = 0.86 * points[..., 0] - 0.50 * points[..., 2]
        vertical = points[..., 1] + 0.16 * points[..., 0] + 0.20 * points[..., 2]
        return np.stack((horizontal, vertical), axis=-1)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        raw = self._raw(points)
        x = HALF + (raw[..., 0] - self.center[0]) * self.scale
        y = 390.0 - (raw[..., 1] - self.center[1]) * self.scale
        return np.stack((x, y), axis=-1)


def _video_path(value: str, source_video_root: Path | None = None) -> Path:
    path = Path(value)
    primary = path if path.is_absolute() else C.DATASET_ROOT / path
    if primary.exists() or source_video_root is None:
        return primary
    parts = path.parts
    if parts and parts[0].lower() == "data":
        parts = parts[1:]
    return source_video_root.joinpath(*parts)


@torch.no_grad()
def predict(model, dataset, device: str, batch_size: int):
    outputs = []
    roots = []
    classes = []
    risks = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        result = model(
            batch["csi"].to(device), batch["link_mask"].to(device)
        )
        valid = batch["valid"].bool()
        outputs.append(smooth_valid(
            result["pose_rel"].float().cpu(), valid, 5
        ))
        roots.append(smooth_valid(
            result["root"].float().cpu(), valid, 5
        ))
        classes.append(result["class_logits"].argmax(-1).cpu())
        risks.append(result["risk_logits"].argmax(-1).cpu())
    return {
        "pose": torch.cat(outputs).numpy(),
        "root": torch.cat(roots).numpy(),
        "class": torch.cat(classes).numpy(),
        "risk": torch.cat(risks).numpy(),
    }


def select_representatives(dataset, prediction, explicit: list[str] | None):
    index = dataset.index.reset_index(drop=True)
    if explicit:
        positions = []
        for trial_id in explicit:
            matched = np.flatnonzero(index.trial_id.to_numpy() == trial_id)
            if not len(matched):
                raise KeyError(f"trial is not in clean seen test: {trial_id}")
            positions.append(int(matched[0]))
        return positions

    errors = []
    for position in range(len(dataset)):
        item = dataset[position]
        valid = item["valid"].numpy().astype(bool)
        distance = np.linalg.norm(
            prediction["pose"][position] - item["pose_rel"].numpy(), axis=-1
        ).mean(-1)
        errors.append(float(distance[valid].mean()))
    errors = np.asarray(errors)
    requests = ((0, "ajh"), (10, "mhw"), (13, "lmh"))
    positions = []
    for class_id, subject in requests:
        candidates = np.flatnonzero(
            (index.class_id.to_numpy() == class_id)
            & (index.subject.to_numpy() == subject)
        )
        if not len(candidates):
            candidates = np.flatnonzero(index.class_id.to_numpy() == class_id)
        ordered = candidates[np.argsort(errors[candidates])]
        positions.append(int(ordered[len(ordered) // 2]))
    return positions


def _fit_video(frame: np.ndarray | None) -> np.ndarray:
    canvas = np.full((HEIGHT, HALF, 3), BG, np.uint8)
    if frame is None:
        cv2.putText(canvas, "Original video unavailable", (145, 350),
                    FONT, 0.72, MUTED, 2, cv2.LINE_AA)
        return canvas
    height, width = frame.shape[:2]
    scale = min(HALF / width, 540 / height)
    resized = cv2.resize(
        frame, (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    x = (HALF - resized.shape[1]) // 2
    y = 100 + (540 - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _draw_skeleton(panel: np.ndarray, points: np.ndarray,
                   color: tuple[int, int, int], thickness: int):
    points = np.rint(points).astype(np.int32)
    for parent, child in C.SKELETON_EDGES:
        cv2.line(panel, tuple(points[parent]), tuple(points[child]),
                 color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(panel, tuple(point), thickness + 1, color, -1, cv2.LINE_AA)


def _draw_mesh(panel: np.ndarray, joints_xy: np.ndarray,
               vertices_xy: np.ndarray, faces: np.ndarray,
               color: tuple[int, int, int], alpha: float):
    vertices_xy = np.rint(vertices_xy).astype(np.int32)
    face_xy = vertices_xy[faces]
    valid = (
        np.isfinite(face_xy).all((1, 2))
        & (face_xy[..., 0].min(1) >= HALF)
        & (face_xy[..., 0].max(1) < WIDTH)
        & (face_xy[..., 1].min(1) >= 70)
        & (face_xy[..., 1].max(1) < HEIGHT)
    )
    polygons = np.ascontiguousarray(face_xy[valid])
    if not len(polygons):
        return

    mask = np.zeros(panel.shape[:2], np.uint8)
    cv2.fillPoly(mask, polygons, 255, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(mask, (0, 0), 7)
    shadow_layer = np.zeros_like(panel)
    shadow_layer[:] = (32, 34, 38)
    shifted = np.zeros_like(shadow)
    shifted[5:, 5:] = shadow[:-5, :-5]
    shadow_weight = (shifted.astype(np.float32) / 255.0 * 0.16)[..., None]
    panel[:] = np.clip(
        panel.astype(np.float32) * (1.0 - shadow_weight)
        + shadow_layer.astype(np.float32) * shadow_weight,
        0, 255,
    ).astype(np.uint8)

    horizontal_light = np.linspace(
        1.08, 0.78, panel.shape[1], dtype=np.float32
    )[None, :, None]
    surface = np.clip(
        np.asarray(color, dtype=np.float32)[None, None, :] * horizontal_light,
        0, 255,
    )
    surface = np.broadcast_to(surface, panel.shape)
    surface_alpha = (
        mask.astype(np.float32) / 255.0 * min(alpha * 1.55, 0.72)
    )[..., None]
    panel[:] = np.clip(
        panel.astype(np.float32) * (1.0 - surface_alpha)
        + surface.astype(np.float32) * surface_alpha,
        0, 255,
    ).astype(np.uint8)

    highlight = cv2.erode(mask, np.ones((5, 5), np.uint8))
    highlight = cv2.GaussianBlur(highlight, (0, 0), 3)
    highlight_weight = (highlight.astype(np.float32) / 255.0 * 0.07)[..., None]
    panel[:] = np.clip(
        panel.astype(np.float32) * (1.0 - highlight_weight)
        + 255.0 * highlight_weight,
        0, 255,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    edge_color = tuple(max(channel - 85, 0) for channel in color)
    cv2.drawContours(panel, contours, -1, edge_color, 2, cv2.LINE_AA)


def _decorate(canvas: np.ndarray, row: pd.Series, frame_index: int,
              fps: float, mode: str, frame_pose_cm: float,
              trial_pose_cm: float, trial_root_cm: float | None,
              class_pred: int, risk_pred: int):
    risk_names = ("safe", "warning", "danger")
    cv2.rectangle(canvas, (0, 0), (WIDTH, 76), (8, 10, 14), -1)
    cv2.putText(
        canvas,
        f"{row.trial_id}  {row.detail_label}  t={frame_index / fps:4.1f}s",
        (20, 31), FONT, 0.68, WHITE, 2, cv2.LINE_AA,
    )
    metric_text = (
        f"frame pose {frame_pose_cm:4.1f}cm | trial pose {trial_pose_cm:4.1f}cm "
        f"| root {trial_root_cm:4.1f}cm"
        if trial_root_cm is not None else
        f"frame pose {frame_pose_cm:4.1f}cm | trial pose {trial_pose_cm:4.1f}cm "
        "| pelvis-aligned; root not evaluated"
    )
    cv2.putText(canvas, metric_text, (20, 61), FONT, 0.56, MUTED, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "Original video (visual reference)", (20, 104),
                FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    title = (
        "GVHMR-22 stick overlay" if mode == "stickman"
        else "SMPL surface fitted to GVHMR-22 joints"
    )
    cv2.putText(canvas, title, (HALF + 18, 104), FONT, 0.55,
                (38, 42, 48), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (HALF + 20, 690), FONT, 0.55,
                GT_COLOR if mode == "stickman" else GT_MESH, 2, cv2.LINE_AA)
    cv2.putText(canvas, "CSI prediction", (HALF + 70, 690), FONT, 0.55,
                PRED_COLOR if mode == "stickman" else PRED_MESH, 2, cv2.LINE_AA)
    predicted_class = int(class_pred)
    predicted_risk = int(risk_pred)
    cv2.putText(
        canvas,
        f"pred class={predicted_class}  risk={risk_names[predicted_risk]}",
        (405, 690), FONT, 0.52, WHITE, 1, cv2.LINE_AA,
    )


def render_trial(row: pd.Series, item: dict, predicted_pose: np.ndarray,
                 predicted_root: np.ndarray, class_pred: int, risk_pred: int,
                 mode: str, output: Path, fps: float,
                 fitter: SmplSurfaceFitter | None,
                 source_video_root: Path | None,
                 pelvis_align_prediction: bool = False) -> dict:
    target_pose = item["pose_rel"].numpy()
    target_root = item["root"].numpy()
    valid = item["valid"].numpy().astype(bool)
    frames = int(row.n_frames)
    target_abs = target_pose[:frames] + target_root[:frames, None]
    rendered_root = (
        target_root[:frames] if pelvis_align_prediction
        else predicted_root[:frames]
    )
    predicted_abs = predicted_pose[:frames] + rendered_root[:, None]
    projector = IsometricProjector(np.concatenate((target_abs, predicted_abs), axis=1))
    pose_error = np.linalg.norm(
        target_pose[:frames] - predicted_pose[:frames], axis=-1
    ).mean(-1)
    root_error = None if pelvis_align_prediction else np.linalg.norm(
        target_root[:frames] - predicted_root[:frames], axis=-1
    )
    selected = valid[:frames] if valid[:frames].any() else np.ones(frames, bool)
    trial_pose_cm = float(pose_error[selected].mean() * 100)
    trial_root_cm = (
        None if root_error is None
        else float(root_error[selected].mean() * 100)
    )
    motion = np.zeros(frames, dtype=np.float32)
    motion[1:] = np.linalg.norm(
        target_abs[1:] - target_abs[:-1], axis=-1
    ).mean(-1)
    preview_frame = int(np.argmax(motion))

    video = _video_path(str(row.original_video), source_video_root)
    capture = cv2.VideoCapture(str(video)) if video.exists() else None
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output}")
    preview = output.with_suffix(".png")
    target_mesh = predicted_mesh = cached_mesh_panel = None
    for frame_index in range(frames):
        frame = None
        if capture is not None:
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        canvas = _fit_video(frame)
        panel = np.full((HEIGHT, HALF, 3), PANEL_BG, np.uint8)
        canvas = np.concatenate((canvas, panel), axis=1)
        gt_xy = projector(target_abs[frame_index])
        pred_xy = projector(predicted_abs[frame_index])
        if mode == "stickman":
            _draw_skeleton(canvas, gt_xy, GT_COLOR, 5)
            _draw_skeleton(canvas, pred_xy, PRED_COLOR, 3)
        else:
            if fitter is None:
                raise RuntimeError("SMPL fitter is required for mesh rendering")
            if frame_index % 2 == 0 or target_mesh is None:
                target_mesh = fitter.fit(target_abs[frame_index])
                predicted_mesh = fitter.fit(predicted_abs[frame_index])
                _draw_mesh(
                    canvas, gt_xy, projector(target_mesh), fitter.faces,
                    GT_MESH, 0.42,
                )
                _draw_mesh(
                    canvas, pred_xy, projector(predicted_mesh), fitter.faces,
                    PRED_MESH, 0.42,
                )
                cached_mesh_panel = canvas[:, HALF:].copy()
            else:
                canvas[:, HALF:] = cached_mesh_panel
        _decorate(
            canvas, row, frame_index, fps, mode,
            float(pose_error[frame_index] * 100), trial_pose_cm,
            trial_root_cm, class_pred, risk_pred,
        )
        writer.write(canvas)
        if frame_index == preview_frame:
            cv2.imwrite(str(preview), canvas)
    writer.release()
    if capture is not None:
        capture.release()
    return {
        "trial_id": row.trial_id,
        "subject": row.subject,
        "environment": row.environment,
        "label": row.detail_label,
        "risk": row.risk,
        "mode": mode,
        "frames": frames,
        "mpjpe_cm": trial_pose_cm,
        "root_cm": trial_root_cm,
        "class_pred": class_pred,
        "risk_pred": risk_pred,
        "video": str(output),
        "preview": str(preview),
        "camera_projection": "metric isometric side panel; not video-pixel overlay",
        "pose_alignment": (
            "GT root used for both trajectories; absolute root not evaluated"
            if pelvis_align_prediction else "independent predicted and GT roots"
        ),
        "mesh_source": (
            "neutral SMPL surface fitted to GVHMR-22 joints"
            if mode == "gvhmr" else None
        ),
    }


def contact_sheet(results: list[dict], destination: Path):
    rows = []
    trial_ids = list(dict.fromkeys(item["trial_id"] for item in results))
    for trial_id in trial_ids:
        images = []
        for mode in ("stickman", "gvhmr"):
            match = next(
                item for item in results
                if item["trial_id"] == trial_id and item["mode"] == mode
            )
            image = cv2.imread(match["preview"])
            images.append(cv2.resize(image, (640, 360)))
        rows.append(np.concatenate(images, axis=1))
    cv2.imwrite(str(destination), np.concatenate(rows, axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p2-checkpoint", type=Path,
        default=Path("work_v2/runs/p2_sub_single_clean_finetune/best_model.pt"),
    )
    parser.add_argument(
        "--root-calibration", type=Path,
        default=Path("docs/results/v13s_pruned_pose_root_ensemble.json"),
    )
    parser.add_argument(
        "--classification-calibration", type=Path,
        default=Path(
            "work_v2/runs/p2_v12w_robust_classification_ensemble/validation.json"
        ),
    )
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--trial-id", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--fps", type=float, default=C.TARGET_FPS)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--smpl-model", type=Path,
        default=Path(r"C:\Users\jjeong\Desktop\NotiFi-3D\SMPLX\SMPL_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--video-root", type=Path,
        default=Path(
            r"C:\Users\jjeong\Desktop\NotiFi-3D\NotiFi-CSI-Pose-Dataset"
            r"\TRAINING_DATA"
        ),
        help="fallback root when split package omits original videos",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("work_v2/runs/v13s_seen_overlays"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_lock = _read_locked(args.root_calibration, args.exp)
    class_lock = _read_locked(args.classification_calibration, args.exp)
    model, configuration = build_locked_model(
        args, device, root_lock, class_lock
    )
    model.eval()
    dataset = pose_only(build_datasets(
        exp=args.exp, baseline="sub", seed=args.seed
    )["test"])
    prediction = predict(model, dataset, device, args.batch_size)
    positions = select_representatives(dataset, prediction, args.trial_id)
    split_index = pd.read_csv(C.SPLIT_DIR / "dev_index.csv")
    video_by_trial = dict(zip(
        split_index.trial_id.astype(str), split_index.original_video
    ))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stickman").mkdir(exist_ok=True)
    (args.out / "gvhmr").mkdir(exist_ok=True)
    fitter = SmplSurfaceFitter(args.smpl_model)
    results = []
    for count, position in enumerate(positions, 1):
        row = dataset.index.iloc[position].copy()
        row["original_video"] = video_by_trial.get(str(row.trial_id), "")
        item = dataset[position]
        print(
            f"[{count}/{len(positions)}] {row.trial_id} "
            f"{row.risk}/{row.detail_label}", flush=True
        )
        for mode in ("stickman", "gvhmr"):
            output = args.out / mode / f"{row.trial_id}_{mode}.mp4"
            results.append(render_trial(
                row, item, prediction["pose"][position],
                prediction["root"][position],
                int(prediction["class"][position]),
                int(prediction["risk"][position]),
                mode, output, args.fps, fitter, args.video_root,
            ))
            print(f"  wrote {output}", flush=True)
    contact_sheet(results, args.out / "preview_contact_sheet.png")
    report = {
        "run": "v13s_seen_representative_overlays",
        "protocol": args.exp,
        "model": "V13S validation-locked healthy-link core",
        "selection": (
            "median-MPJPE walking/ajh, stumble/mhw, fall-walking/lmh"
            if args.trial_id is None else "explicit trial ids"
        ),
        "selection_uses_test_gt": True,
        "selection_note": "visualization only; no model or metric selection",
        "configuration": configuration,
        "results": results,
        "preview_contact_sheet": str(args.out / "preview_contact_sheet.png"),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(
        args.out / "selected_trials.csv", index=False, encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.out),
        "preview": str(args.out / "preview_contact_sheet.png"),
        "trials": [dataset.index.iloc[position].trial_id for position in positions],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
