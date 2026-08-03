"""예측 시각화 — 원본 영상 + GT/예측 골격 영상 생성.

    ┌─────────────────────────────────────────────┐
    │  원본 영상 (사람이 실제로 뭘 했는가)              │
    ├──────────────────────┬──────────────────────┤
    │  FRONT  (x, 높이)     │  TOP-DOWN (x, z)     │
    │  GT 초록 / 예측 빨강    │  바닥 평면 + 이동 궤적  │
    └──────────────────────┴──────────────────────┘

**GT 를 전혀 빌리지 않는다.** 예측 골격은 모델이 낸 root(골반 절대 위치)에 모델이 낸
pose_rel 을 얹어 그린다. 즉 '방 안 어디에 서 있는가'까지 전부 예측값이다. GT root 에
예측 자세를 붙이면 root 오차(실측 42.6cm)가 통째로 숨어 성능이 실제보다 좋아 보인다.

    그리는 좌표 = pose_rel(예측) + root(예측)

**패널이 두 개인 이유.** world 는 Y-up(바닥 y=0, contract.UP_AXIS)이라 정면뷰(x,y)는
자세와 키를 보여주지만 방 안을 걸어다닌 것은 보이지 않는다. 걷기는 바닥 평면(x,z)에서
일어난다(실측: walking trial 의 root 이동이 x 1.58m 인데 y 는 0.11m 뿐). 그래서
위에서 본 뷰를 따로 두고 지나온 궤적을 꼬리로 남긴다.

전처리는 학습과 **같은 경로**를 쓴다(PoseDataset, train=False). 시각화용으로 따로
로드하면 정규화·마스크가 어긋나 "모델은 멀쩡한데 그림만 틀린" 상황이 생긴다.

실행:
    python -m notifi_pose.tools.visualize --run baseline_tcn
    python -m notifi_pose.tools.visualize --run baseline_tcn --pick worst --n 6
    python -m notifi_pose.tools.visualize --run baseline_tcn --trial-id ajh_E01_S01_t021
    python -m notifi_pose.tools.visualize --run loso_lmh --exp loso --fold test_lmh

결과는 work_v2/runs/<run>/viz/ 에 저장된다 (<trial_id>.mp4 + trial_metrics.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .. import contract as C
from .. import nets
from ..dataio import cache as cache_mod
from ..dataio import dataset as D

# ---------------------------------------------------------------- 레이아웃

VW, VH = 960, 540              # 상단 영상 (16:9)
PW, PH = 480, 470              # 하단 패널 2칸
CW, CH = VW, VH + PH

GT_COLOR = (60, 190, 60)       # BGR 초록
PR_COLOR = (60, 60, 235)       # BGR 빨강
BG = (250, 250, 250)
GRID = (215, 215, 215)
TEXT = (40, 40, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX

#: 바닥 평면 축. UP_AXIS 가 1(y)이므로 나머지 0(x), 2(z) 가 바닥이다.
FLOOR_AXES = tuple(i for i in range(3) if i != C.UP_AXIS)

#: 방 크기 (m). test 셋 실측 root 범위가 x 4.42m, z 3.72m 라 여유를 둔 값.
#: 고정해 두어야 영상마다 축척이 같아 '방 안 어디'가 비교된다.
ROOM_X = (-2.6, 2.6)
ROOM_Z = (-2.3, 2.3)

#: 정면뷰 세로 범위 (m). 바닥 0, 서 있는 사람 머리 ~1.7m.
FRONT_Y = (-0.15, 2.0)

#: 궤적 꼬리 길이 (프레임). 30fps 기준 2초.
TRAIL = 60


# ---------------------------------------------------------------- 모델 로드


#: LinkEncoder 를 nn.Sequential 에서 개별 층으로 쪼개기 전(FiLM 도입 전) 체크포인트의
#: 키 이름. 옛 실행 결과를 계속 비교할 수 있어야 하므로 이름만 갈아끼워 읽는다.
_LEGACY_KEYS = {
    "encoder.net.0": "encoder.fc1",     # Linear(228 -> 192)
    "encoder.net.1": "encoder.norm1",   # LayerNorm(192)
    "encoder.net.4": "encoder.fc2",     # Linear(192 -> 96)
}


def _remap_legacy(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        for old, new in _LEGACY_KEYS.items():
            if k.startswith(old + "."):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_model(run_dir: Path, device: str) -> tuple[nets.PoseNet, dict]:
    """best_model.pt 에서 모델을 복원한다.

    체크포인트에 cfg 가 통째로 들어있으므로 학습 때 인자를 다시 적을 필요가 없다.
    PerLinkNorm 통계도 buffer 라 state_dict 에 포함되어 함께 복원된다.
    """
    ckpt_path = run_dir / "calibrated_model.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 없음. 먼저 학습하라.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    if ckpt.get("preproc_version") != C.PREPROC_VERSION:
        raise RuntimeError(
            f"전처리 버전 불일치: 체크포인트 {ckpt.get('preproc_version')} "
            f"!= 코드 {C.PREPROC_VERSION}. 같은 규약으로 만든 모델이 아니다.")

    if cfg["arch"] in {
        "graphformer", "robust_graphformer", "impact_graphformer", "latent_flow", "v3"
    }:
        kwargs = {
            "hidden": cfg["hidden"], "n_blocks": cfg["n_blocks"],
            "dropout": cfg["dropout"], "heads": cfg.get("heads", 4),
            "graph_blocks": cfg.get("graph_blocks", 2),
        }
        if cfg["arch"] in {
            "graphformer", "robust_graphformer", "impact_graphformer", "latent_flow"
        }:
            kwargs["decoder"] = cfg.get("decoder", "tree")
            kwargs["domain_grl"] = cfg.get("domain_grl", 0.2)
            if cfg["arch"] == "impact_graphformer":
                kwargs["refiner_joint_scale"] = cfg.get("refiner_joint_scale")
            elif cfg["arch"] == "latent_flow":
                kwargs.update(
                    flow_steps=cfg.get("flow_steps", 4),
                    flow_noise=cfg.get("flow_noise", 0.25),
                )
        else:
            kwargs.update(
                frequency_tokens=cfg.get("frequency_tokens", 12),
                geometry_path=cfg.get("geometry_path"),
                domain_grl=cfg.get("domain_grl", 0.2),
            )
        model = nets.build_model(cfg["arch"], **kwargs).to(device)
        model.load_state_dict(_remap_legacy(ckpt["model"]))
        model.eval()
        return model, ckpt

    model = nets.build_model(
        cfg["arch"], hidden=cfg["hidden"], dilations=tuple(cfg["dilations"]),
        n_blocks=cfg["n_blocks"], dropout=cfg["dropout"],
        fusion=cfg.get("fusion", "gate"),              # 옛 체크포인트엔 키가 없다
        film=cfg.get("film", False)).to(device)
    model.load_state_dict(_remap_legacy(ckpt["model"]))
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------- 추론


@torch.no_grad()
def predict(model: nets.PoseNet, ds: D.PoseDataset, device: str,
            batch_size: int = 16) -> dict:
    """데이터셋 전체를 추론해 trial 단위 결과를 모은다."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pose, root, cls, risk, mpjpe, rooterr = [], [], [], [], [], []
    for b in loader:
        out = model(b["csi"].to(device), b["link_mask"].to(device))
        p, r = out["pose_rel"].cpu(), out["root"].cpu()
        m = b["valid"].to(torch.float32)                     # [B, T]
        denom = m.sum(1).clamp(min=1.0)

        d = torch.linalg.norm(p - b["pose_rel"], dim=-1)      # [B, T, J]
        mpjpe.append((d.mean(-1) * m).sum(1) / denom)
        dr = torch.linalg.norm(r - b["root"], dim=-1)         # [B, T]
        rooterr.append((dr * m).sum(1) / denom)

        pose.append(p)
        root.append(r)
        cls.append(out["class_logits"].argmax(-1).cpu())
        risk.append(out["risk_logits"].argmax(-1).cpu())

    return {
        "pose_rel": torch.cat(pose).numpy(),
        "root": torch.cat(root).numpy(),
        "class_pred": torch.cat(cls).numpy(),
        "risk_pred": torch.cat(risk).numpy(),
        "mpjpe": torch.cat(mpjpe).numpy(),
        "root_err": torch.cat(rooterr).numpy(),
    }


# ---------------------------------------------------------------- 투영


class View:
    """world 3D -> 패널 2D 투영. 축척을 고정해 종횡비를 유지한다.

    Args:
        ax_h, ax_v: 화면 가로/세로에 대응하는 world 축 인덱스
        lim_h, lim_v: 그 축의 표시 범위 (m)
        flip_v: 화면 y 를 뒤집을 것인가 (높이축은 위가 커야 하므로 True)
    """

    def __init__(self, ax_h: int, ax_v: int, lim_h, lim_v, flip_v: bool):
        self.ax_h, self.ax_v, self.flip_v = ax_h, ax_v, flip_v
        self.lim_h, self.lim_v = lim_h, lim_v
        # 가로/세로 중 더 빡빡한 쪽에 맞춘다 -> 지정 범위가 항상 화면 안에 들어온다
        self.scale = min(PW / (lim_h[1] - lim_h[0]), PH / (lim_v[1] - lim_v[0]))
        self.ch = (lim_h[0] + lim_h[1]) / 2
        self.cv = (lim_v[0] + lim_v[1]) / 2

    def __call__(self, P: np.ndarray) -> np.ndarray:
        """P [..., 3] -> [..., 2] 픽셀 좌표."""
        h = PW / 2 + (P[..., self.ax_h] - self.ch) * self.scale
        v = (P[..., self.ax_v] - self.cv) * self.scale
        v = PH / 2 - v if self.flip_v else PH / 2 + v
        return np.stack([h, v], axis=-1)


FRONT = View(ax_h=FLOOR_AXES[0], ax_v=C.UP_AXIS,
             lim_h=ROOM_X, lim_v=FRONT_Y, flip_v=True)
TOP = View(ax_h=FLOOR_AXES[0], ax_v=FLOOR_AXES[1],
           lim_h=ROOM_X, lim_v=ROOM_Z, flip_v=False)


# ---------------------------------------------------------------- 그리기


def _draw_skeleton(canvas: np.ndarray, xy: np.ndarray, color,
                   thickness: int = 2, radius: int = 3) -> None:
    h, w = canvas.shape[:2]
    ok = (np.isfinite(xy).all(axis=-1)
          & (np.abs(xy[:, 0]) < w * 4) & (np.abs(xy[:, 1]) < h * 4))
    for a, b in C.SKELETON_EDGES:
        if ok[a] and ok[b]:
            cv2.line(canvas, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)),
                     color, thickness, cv2.LINE_AA)
    for j in range(len(xy)):
        if ok[j]:
            cv2.circle(canvas, tuple(xy[j].astype(int)), radius, color, -1, cv2.LINE_AA)


def _draw_trail(canvas: np.ndarray, xy: np.ndarray, color) -> None:
    """지나온 궤적. 오래된 점일수록 옅게 그려 진행 방향이 보이게 한다."""
    for i in range(1, len(xy)):
        w = i / len(xy)
        c = tuple(int(255 - (255 - v) * (0.25 + 0.75 * w)) for v in color)
        cv2.line(canvas, tuple(xy[i - 1].astype(int)), tuple(xy[i].astype(int)),
                 c, 2, cv2.LINE_AA)


def _front_panel(gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    canvas = np.full((PH, PW, 3), BG, np.uint8)
    # 바닥선 y=0 — '누웠는가 / 떠 있는가'의 기준
    y0 = int(FRONT(np.array([[0.0, 0.0, 0.0]]))[0, 1])
    cv2.line(canvas, (0, y0), (PW, y0), (150, 150, 150), 2, cv2.LINE_AA)
    for m in np.arange(0.5, FRONT_Y[1], 0.5):               # 0.5m 눈금
        y = int(FRONT(np.array([[0.0, m, 0.0]]))[0, 1])
        cv2.line(canvas, (0, y), (PW, y), GRID, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{m:.1f}m", (4, y - 4), FONT, 0.35, (170, 170, 170), 1)

    _draw_skeleton(canvas, FRONT(gt), GT_COLOR, 3, 4)
    _draw_skeleton(canvas, FRONT(pr), PR_COLOR, 2, 3)
    _label(canvas, "FRONT  (x, height)", "GT vs predicted - both absolute")
    return canvas


def _top_panel(gt_abs: np.ndarray, pr_abs: np.ndarray,
               gt_trail: np.ndarray, pr_trail: np.ndarray) -> np.ndarray:
    canvas = np.full((PH, PW, 3), BG, np.uint8)
    for m in np.arange(-2.0, 2.5, 1.0):                     # 1m 격자
        x = int(TOP(np.array([[m, 0.0, 0.0]]))[0, 0])
        z = int(TOP(np.array([[0.0, 0.0, m]]))[0, 1])
        cv2.line(canvas, (x, 0), (x, PH), GRID, 1, cv2.LINE_AA)
        cv2.line(canvas, (0, z), (PW, z), GRID, 1, cv2.LINE_AA)

    _draw_trail(canvas, TOP(gt_trail), GT_COLOR)
    _draw_trail(canvas, TOP(pr_trail), PR_COLOR)
    # 바닥에 눕힌 골격 — 몸이 어느 쪽을 향하는지 보인다
    _draw_skeleton(canvas, TOP(gt_abs), GT_COLOR, 2, 3)
    _draw_skeleton(canvas, TOP(pr_abs), PR_COLOR, 2, 2)
    # 골반 위치를 크게 — 이게 곧 '방 안 어디'
    for P, col in ((gt_abs, GT_COLOR), (pr_abs, PR_COLOR)):
        c = TOP(P[C.ROOT_JOINT:C.ROOT_JOINT + 1])[0].astype(int)
        cv2.circle(canvas, tuple(c), 8, col, 2, cv2.LINE_AA)

    _label(canvas, "TOP-DOWN  (floor x-z)", "walking path, 1m grid")
    return canvas


def _label(canvas: np.ndarray, title: str, subtitle: str) -> None:
    cv2.putText(canvas, title, (12, 28), FONT, 0.66, TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 50), FONT, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GT", (PW - 108, 28), FONT, 0.6, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "PRED", (PW - 66, 28), FONT, 0.6, PR_COLOR, 2, cv2.LINE_AA)


def _header(canvas: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """영상 위 반투명 띠 + 텍스트 (밝은 배경에서도 읽히도록)."""
    h = 14 + 27 * len(lines)
    band = canvas[0:h, 0:VW]
    canvas[0:h, 0:VW] = cv2.addWeighted(band, 0.42, np.zeros_like(band), 0.58, 0)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (12, 31 + 27 * i), FONT, 0.62, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- trial 렌더


def render_trial(meta: pd.Series, gt_pose: np.ndarray, gt_root: np.ndarray,
                 pr_pose: np.ndarray, pr_root: np.ndarray, valid: np.ndarray,
                 cls_pred: int, risk_pred: int, class_names: dict,
                 video_path: Path | None, out_path: Path, fps: float) -> dict:
    """trial 하나를 mp4 로 렌더한다.

    Args:
        gt_pose/pr_pose: [T, 22, 3] 골반 기준 상대 좌표
        gt_root/pr_root: [T, 3] 골반 절대 위치. **예측 쪽도 모델 출력을 그대로 쓴다**
        valid: [T] GT 를 채점해도 되는 프레임
    """
    T = len(gt_pose)
    gt_abs = gt_pose + gt_root[:, None, :]
    pr_abs = pr_pose + pr_root[:, None, :]

    per_frame = np.linalg.norm(gt_pose - pr_pose, axis=-1).mean(-1)      # [T]
    per_root = np.linalg.norm(gt_root - pr_root, axis=-1)                # [T]

    cap = cv2.VideoCapture(str(video_path)) if video_path and video_path.exists() else None
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))

    risk_names = ("safe", "warning", "danger")
    label = str(meta.detail_label)
    cls_ok = int(cls_pred) == int(meta.class_id)
    risk_ok = int(risk_pred) == int(meta.risk_id)

    for t in range(T):
        canvas = np.full((CH, CW, 3), BG, np.uint8)

        # ---- 상단: 원본 영상. 캐시와 같은 30fps 격자라 프레임 번호가 곧 시각이다
        if cap is not None and t < n_video:
            ok, frame = cap.read()
            if ok:
                canvas[0:VH, 0:VW] = cv2.resize(frame, (VW, VH))
        elif cap is None:
            cv2.putText(canvas, "no video", (VW // 2 - 70, VH // 2),
                        FONT, 1.0, (175, 175, 175), 2, cv2.LINE_AA)

        v = bool(valid[t])
        _header(canvas, [
            (f"{meta.trial_id}   {label}   t={t / fps:5.2f}s", (255, 255, 255)),
            (f"MPJPE {per_frame[t] * 100:5.1f}cm    root {per_root[t] * 100:5.1f}cm"
             + ("" if v else "    [GT invalid]"),
             (255, 255, 255) if v else (155, 155, 155)),
            (f"class {label} -> {class_names.get(int(cls_pred), '?')} "
             f"[{'OK' if cls_ok else 'MISS'}]    "
             f"risk {risk_names[int(meta.risk_id)]} -> {risk_names[int(risk_pred)]} "
             f"[{'OK' if risk_ok else 'MISS'}]",
             (120, 240, 120) if (cls_ok and risk_ok) else (110, 110, 245)),
        ])

        s = max(0, t - TRAIL)
        canvas[VH:VH + PH, 0:PW] = _front_panel(gt_abs[t], pr_abs[t])
        canvas[VH:VH + PH, PW:PW * 2] = _top_panel(
            gt_abs[t], pr_abs[t], gt_root[s:t + 1], pr_root[s:t + 1])

        cv2.line(canvas, (PW, VH), (PW, CH), (200, 200, 200), 1)
        cv2.line(canvas, (0, VH), (CW, VH), (170, 170, 170), 2)
        writer.write(canvas)

    writer.release()
    if cap is not None:
        cap.release()

    sel = valid if valid.any() else np.ones(T, bool)
    return {"trial_id": meta.trial_id, "label": label,
            "mpjpe_cm": float(per_frame[sel].mean() * 100),
            "root_cm": float(per_root[sel].mean() * 100),
            "class_ok": cls_ok, "risk_ok": risk_ok,
            "video": cap is not None, "path": str(out_path)}


# ---------------------------------------------------------------- 보조


def video_paths() -> dict[str, Path]:
    """trial_id -> 원본 영상 절대경로. 경로 컬럼은 split 인덱스에만 있다."""
    out: dict[str, Path] = {}
    for name in ("dev_index.csv", "sealed_index.csv"):
        p = C.SPLIT_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "original_video" not in df.columns:
            continue
        for tid, rel in zip(df.trial_id, df.original_video):
            if isinstance(rel, str):
                out[tid] = C.DATASET_ROOT / rel
    return out


def choose(index: pd.DataFrame, mpjpe: np.ndarray, n: int, how: str) -> np.ndarray:
    """어떤 trial 을 렌더할지 고른다. 반환은 데이터셋 내 위치(0..len-1)."""
    if how == "worst":
        return np.argsort(-mpjpe)[:n]
    if how == "best":
        return np.argsort(mpjpe)[:n]
    if how == "random":
        return np.random.default_rng(0).choice(len(mpjpe), size=min(n, len(mpjpe)),
                                               replace=False)
    # spread — 클래스마다 중간 성능인 것을 하나씩. 잘 된 것만 보면 판단이 치우친다.
    # danger 부터 넣는다(배포에서 제일 중요한 쪽).
    picked: list[int] = []
    order = index.reset_index(drop=True).sort_values(["risk_id", "class_id"],
                                                     ascending=[False, True])
    for _, g in order.groupby("class_id", sort=False):
        pos = g.index.to_numpy()
        picked.append(int(pos[np.argsort(mpjpe[pos])[len(pos) // 2]]))
        if len(picked) >= n:
            break
    return np.array(picked[:n])


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="baseline_tcn", help="work_v2/runs/<run>")
    ap.add_argument("--exp", default="single_split", choices=["single_split", "loso"])
    ap.add_argument("--fold", default=None, help="exp=loso 일 때 test_lmh 등")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--trial-id", nargs="+", default=None, help="직접 지정")
    ap.add_argument("--allow-excluded", action="store_true",
                    help="role=excluded 인 trial 도 렌더 (GT 가 신뢰 불가한 데이터)")
    ap.add_argument("--pick", default="spread", choices=["spread", "worst", "best", "random"])
    ap.add_argument("-n", "--n", type=int, default=6)
    ap.add_argument("--all", action="store_true",
                    help="후보 전체를 렌더 (--pick 무시). 329 trial 기준 약 2GB")
    ap.add_argument("--prefix-error", action="store_true",
                    help="파일명 앞에 MPJPE 를 붙여 오차 순으로 정렬되게 한다. "
                         "끄면 trial_id 만 써서 다른 run 과 파일명이 맞는다")
    ap.add_argument("--fps", type=float, default=C.TARGET_FPS)
    ap.add_argument("--no-video", action="store_true", help="원본 영상 패널 생략")
    ap.add_argument("--out", default=None, help="기본: work_v2/runs/<run>/viz")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = C.WORK_ROOT / "runs" / args.run
    out_dir = Path(args.out) if args.out else run_dir / "viz"

    model, ckpt = load_model(run_dir, device)
    print(f"[viz] {args.run}  ep{ckpt['epoch']}  val_mpjpe {ckpt['val_mpjpe'] * 100:.2f}cm")
    print(f"[viz] {model.describe()}  device={device}")

    # 학습과 동일한 전처리 경로 (train=False -> link dropout 없음)
    cache = cache_mod.open_cache()
    link_ok = D._link_ok_matrix(cache)
    if args.trial_id:
        # --trial-id 는 split 필터를 건너뛴다. 제외된 trial(GT 파손 등)이 그대로
        # 들어와 "성능이 나쁘다"로 오독되기 쉬우므로 여기서 걸러낸다.
        rows, dropped = [], []
        for t in args.trial_id:
            r = cache.row_of(t)
            role = cache.index.role.iloc[r]
            if role == "excluded" and not args.allow_excluded:
                dropped.append((t, role))
            else:
                rows.append(r)
        if dropped:
            print(f"[viz] 제외 trial {len(dropped)}개 건너뜀 "
                  f"(role=excluded — GT 신뢰 불가). --allow-excluded 로 강제 가능")
            for t, role in dropped:
                print(f"        - {t}")
        if not rows:
            print("[viz] 렌더할 trial 이 없다.")
            return 1
        rows = np.array(rows)
        source = "explicit"
    else:
        experiments = D.load_experiments()
        cfg = (experiments["loso"][args.fold] if args.exp == "loso"
               else experiments["single_split"])
        rows = D.select_rows(cache.index, cfg[args.split])
        source = f"{args.exp}{'/' + args.fold if args.fold else ''} {args.split}"
    # 체크포인트가 학습에 쓴 것과 **같은** 전처리를 써야 한다. 기준선 모드를 빼먹으면
    # 모델은 '빈방 뺀 입력'을 기대하는데 원본이 들어가 결과가 통째로 무너진다
    # (실제로 겪음: p2_sub_single 의 MPJPE 가 15.8cm -> 13.8~56.7cm, 분류 전부 오답).
    baseline_mode = ckpt["cfg"].get("baseline", "none")
    ds = D.PoseDataset(rows, cache, link_ok, train=False,
                       baseline=D.SiteBaseline(baseline_mode))
    print(f"[viz] 후보 {len(ds)} trial  ({source})")

    m = predict(model, ds, device)
    index = ds.index
    class_names = dict(zip(cache.index.class_id, cache.index.detail_label))

    if args.trial_id:
        order, how = np.arange(len(ds)), "explicit"
    elif args.all:
        order, how = np.argsort(-m["mpjpe"]), "all (오차 큰 순)"
    else:
        order, how = choose(index, m["mpjpe"], args.n, args.pick), args.pick
    print(f"[viz] 렌더 {len(order)}개 ({how})")

    vids = {} if args.no_video else video_paths()
    a = cache.arrays
    for k, pos in enumerate(order, 1):
        pos = int(pos)
        row = int(ds.rows[pos])
        meta = index.iloc[pos]
        T = int(meta.n_frames)

        name = (f"{m['mpjpe'][pos]*100:05.1f}cm_{meta.trial_id}.mp4"
                if args.prefix_error else f"{meta.trial_id}.mp4")
        res = render_trial(
            meta=meta,
            gt_pose=np.asarray(a["pose_rel"][row, :T], dtype=np.float32),
            gt_root=np.asarray(a["root"][row, :T], dtype=np.float32),
            pr_pose=m["pose_rel"][pos, :T],
            pr_root=m["root"][pos, :T],
            valid=np.asarray(a["valid"][row, :T]).astype(bool),
            cls_pred=int(m["class_pred"][pos]),
            risk_pred=int(m["risk_pred"][pos]),
            class_names=class_names,
            video_path=vids.get(meta.trial_id),
            out_path=out_dir / name,
            fps=args.fps,
        )
        print(f"  [{k}/{len(order)}] {res['trial_id']:22s} {res['label']:20s} "
              f"MPJPE {res['mpjpe_cm']:5.1f}cm  root {res['root_cm']:5.1f}cm  "
              f"cls {'O' if res['class_ok'] else 'X'} "
              f"risk {'O' if res['risk_ok'] else 'X'}"
              f"{'' if res['video'] else '  (영상없음)'}")

    # 전체 trial 지표 — 다음에 뭘 더 볼지 고를 때 쓴다
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "trial_id": index.trial_id.to_numpy(),
        "subject": index.subject.to_numpy(),
        "environment": index.environment.to_numpy(),
        "label": index.detail_label.to_numpy(),
        "risk": index.risk.to_numpy(),
        "mpjpe_cm": m["mpjpe"] * 100,
        "root_cm": m["root_err"] * 100,
        "class_ok": m["class_pred"] == index.class_id.to_numpy(),
        "risk_ok": m["risk_pred"] == index.risk_id.to_numpy(),
    }).sort_values("mpjpe_cm").to_csv(out_dir / "trial_metrics.csv",
                                      index=False, encoding="utf-8")

    print(f"\n  wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
