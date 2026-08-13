"""잠긴 artifact를 yja/E02에서 단 한 번 감사 평가한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from notifi_ai_v2.link_quality import read_link_quality  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from notifi_pose.pose_simulation import retrieval_metrics  # noqa: E402
from notifi_pose.deployment import CAL44Deployment  # noqa: E402


BASIC_CLASSES = (0, 1, 2, 3, 4, 5, 7, 8)
WARNING_CLASSES = (9, 10, 11)
DANGER_CLASSES = (12, 13, 14, 15, 16)


def sha256(path: Path) -> str:
    """평가한 artifact를 재현할 수 있도록 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_support(
    table: pd.DataFrame,
    rows: np.ndarray,
    classes: tuple[int, ...],
    shots: int,
    seed: int,
) -> np.ndarray:
    """class별 고정 shot을 trial ID 정렬 뒤 seed로 선택한다."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    labels = table.class_id.to_numpy()
    trial_ids = table.trial_id.astype(str).to_numpy()
    for class_id in classes:
        candidates = rows[labels[rows] == class_id]
        candidates = candidates[np.argsort(trial_ids[candidates])]
        if len(candidates) < shots:
            raise RuntimeError(f"sealed class {class_id} lacks support")
        selected.extend(rng.permutation(candidates)[:shots].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


class CacheStore:
    """메모리맵 cache에서 필요한 행만 읽고 고장 링크를 마스킹한다."""

    def __init__(self, root: Path, table: pd.DataFrame) -> None:
        self.root = root
        self.table = table
        self.csi = np.load(root / "csi_iq.npy", mmap_mode="r")
        self.mask = np.load(root / "link_mask.npy", mmap_mode="r")
        self.pose = np.load(root / "pose_rel.npy", mmap_mode="r")
        self.valid = np.load(root / "valid.npy", mmap_mode="r")
        self.link_quality = read_link_quality(root)

    def signal(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """지정한 cache 행의 CSI와 최종 link mask를 CPU tensor로 반환한다."""
        csi = np.array(self.csi[rows], dtype=np.float32)
        mask = np.array(self.mask[rows], dtype=bool)
        for position, row in enumerate(rows):
            trial_id = str(self.table.trial_id.iloc[int(row)])
            usable = self.link_quality.get(trial_id)
            if usable is not None:
                mask[position] &= usable[None]
        csi *= mask[..., None, None].astype(np.float32)
        return torch.from_numpy(csi), torch.from_numpy(mask)


def main() -> None:
    """잠긴 설정을 바꾸지 않고 분류와 root-relative pose를 최종 채점한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirm-sealed-evaluation", action="store_true")
    options = parser.parse_args()
    if not options.confirm_sealed_evaluation:
        raise RuntimeError("sealed evaluation requires explicit confirmation")

    table = pd.read_csv(options.cache_root / "cache_index.csv")
    sealed = np.flatnonzero((
        (table.subject == "yja")
        & (table.environment == "E02")
        & table.cache_ok
    ).to_numpy())
    pose_rows = sealed[
        table.task.iloc[sealed].to_numpy() == "pose_and_action"
    ]
    absence = sealed[
        table.class_id.iloc[sealed].to_numpy() == 6
    ]
    basic = select_support(
        table, pose_rows, BASIC_CLASSES, 2, options.support_seed
    )
    warning = select_support(
        table, pose_rows, WARNING_CLASSES, 1, options.support_seed + 2000
    )
    danger = select_support(
        table, pose_rows, DANGER_CLASSES, 1, options.support_seed + 1000
    )
    support = set(np.concatenate((basic, warning, danger)).tolist())
    query = np.asarray(
        [row for row in pose_rows.tolist() if row not in support], dtype=np.int64
    )
    if len(sealed) != 275 or len(absence) != 12 or len(query) != 239:
        raise RuntimeError(
            f"unexpected sealed counts: total={len(sealed)}, "
            f"absence={len(absence)}, query={len(query)}"
        )

    store = CacheStore(options.cache_root, table)
    basic_csi, basic_mask = store.signal(basic)
    warning_csi, warning_mask = store.signal(warning)
    danger_csi, danger_mask = store.signal(danger)
    absence_csi, absence_mask = store.signal(absence)
    runtime = CAL44Deployment.load(str(options.artifact), device=options.device)
    calibration = runtime.calibrate(
        basic_csi,
        basic_mask,
        torch.tensor(table.class_id.iloc[basic].to_numpy()).long(),
        absence_csi,
        absence_mask,
        danger_csi,
        danger_mask,
        torch.tensor(table.class_id.iloc[danger].to_numpy()).long(),
        warning_csi,
        warning_mask,
        torch.tensor(table.class_id.iloc[warning].to_numpy()).long(),
    )

    action_logits, risk_logits, predicted_pose = [], [], []
    for start in range(0, len(query), options.batch_size):
        batch_rows = query[start:start + options.batch_size]
        query_csi, query_mask = store.signal(batch_rows)
        prediction = runtime.predict(
            query_csi, query_mask, calibration,
            simulate_pose=True, risk_profile="conservative",
        )
        action_logits.append(prediction["action_logits"].detach().cpu())
        risk_logits.append(prediction["risk_logits"].detach().cpu())
        predicted_pose.append(prediction["pose_rel"].detach().cpu())

    actions = torch.tensor(table.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(table.risk_id.iloc[query].to_numpy()).long()
    classification = classification_metrics(
        torch.cat(action_logits), torch.cat(risk_logits), actions, risks
    )
    classification["safe_to_danger_rate"] = (
        classification["safe_to_danger"] / max(classification["safe_total"], 1)
    )
    target_pose = torch.from_numpy(np.array(store.pose[query], dtype=np.float32))
    valid = torch.from_numpy(np.array(store.valid[query], dtype=bool))
    valid &= torch.isfinite(target_pose).all(-1).all(-1)
    pose = retrieval_metrics(
        torch.cat(predicted_pose), target_pose, valid, risks
    )
    result = {
        "run": "NOTIFI-AI-V2-SEALED-YJA-E02-FINAL",
        "protocol": (
            "locked artifact; yja/E02 final-only; support removed from query; "
            "query labels and pose used only after inference for metrics"
        ),
        "artifact": str(options.artifact.resolve()),
        "artifact_sha256": sha256(options.artifact),
        "support_seed": int(options.support_seed),
        "counts": {
            "total": len(sealed),
            "absence": len(absence),
            "basic_support": len(basic),
            "warning_support": len(warning),
            "danger_support": len(danger),
            "query": len(query),
        },
        "support_trial_ids": table.trial_id.iloc[
            np.concatenate((basic, warning, danger))
        ].astype(str).tolist(),
        "classification": classification,
        "pose": pose,
        "used_for_model_selection": False,
        "used_for_training_or_tuning": False,
        "query_labels_or_pose_gt_used_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "classification": classification,
        "pose": pose,
        "counts": result["counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
