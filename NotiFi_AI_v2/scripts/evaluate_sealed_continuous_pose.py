"""Final-only sealed evaluation of the continuous pose danger-expert artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
LEGACY = REPOSITORY / "CSI-to-Pose-v2"
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(PROJECT))

from notifi_pose.deployment import CAL44Deployment  # noqa: E402
from notifi_pose.metrics import classification_metrics  # noqa: E402
from notifi_pose.pose_simulation import retrieval_metrics  # noqa: E402
from evaluate_sealed_yja import (  # noqa: E402
    BASIC_CLASSES,
    DANGER_CLASSES,
    WARNING_CLASSES,
    CacheStore,
    select_support,
)


def sha256(path: Path) -> str:
    """Hash the locked artifact so the sealed result is reproducible."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Run one final unseen audit without feeding query labels or pose to inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirm-sealed-evaluation", action="store_true")
    options = parser.parse_args()
    if not options.confirm_sealed_evaluation:
        raise RuntimeError("sealed evaluation requires explicit confirmation")
    bundle = torch.load(options.artifact, map_location="cpu", weights_only=False)
    branch = bundle.get("continuous_pose")
    if branch is None or branch.get("sealed_target_used") is not False:
        raise RuntimeError("artifact lacks a sealed-clean continuous pose branch")
    table = pd.read_csv(options.cache_root / "cache_index.csv")
    sealed = np.flatnonzero(((table.subject == "yja")
                             & (table.environment == "E02")
                             & table.cache_ok).to_numpy())
    pose_rows = sealed[table.task.iloc[sealed].to_numpy() == "pose_and_action"]
    absence = sealed[table.class_id.iloc[sealed].to_numpy() == 6]
    basic = select_support(table, pose_rows, BASIC_CLASSES, 2, options.support_seed)
    warning = select_support(
        table, pose_rows, WARNING_CLASSES, 1, options.support_seed + 2000,
    )
    danger = select_support(
        table, pose_rows, DANGER_CLASSES, 1, options.support_seed + 1000,
    )
    support_set = set(np.concatenate((basic, warning, danger)).tolist())
    query = np.asarray(
        [row for row in pose_rows.tolist() if row not in support_set],
        dtype=np.int64,
    )
    if len(sealed) != 275 or len(absence) != 12 or len(query) != 239:
        raise RuntimeError(
            f"unexpected sealed counts: total={len(sealed)} "
            f"absence={len(absence)} query={len(query)}"
        )
    store = CacheStore(options.cache_root, table)
    basic_csi, basic_mask = store.signal(basic)
    warning_csi, warning_mask = store.signal(warning)
    danger_csi, danger_mask = store.signal(danger)
    absence_csi, absence_mask = store.signal(absence)
    runtime = CAL44Deployment.load(str(options.artifact), device=options.device)
    calibration = runtime.calibrate(
        basic_csi, basic_mask,
        torch.tensor(table.class_id.iloc[basic].to_numpy()).long(),
        absence_csi, absence_mask,
        danger_csi, danger_mask,
        torch.tensor(table.class_id.iloc[danger].to_numpy()).long(),
        warning_csi, warning_mask,
        torch.tensor(table.class_id.iloc[warning].to_numpy()).long(),
    )
    action_logits, risk_logits, predicted_pose = [], [], []
    routed = 0
    with torch.no_grad():
        for start in range(0, len(query), options.batch_size):
            rows = query[start:start + options.batch_size]
            query_csi, query_mask = store.signal(rows)
            classification = runtime.predict(
                query_csi, query_mask, calibration,
                simulate_pose=True, risk_profile="conservative",
            )
            current_action = classification["action_logits"].detach()
            current_risk = classification["risk_logits"].detach()
            gate = classification["continuous_danger_expert_used"]
            routed += int(gate.sum())
            action_logits.append(current_action.cpu())
            risk_logits.append(current_risk.cpu())
            predicted_pose.append(classification["pose_rel"].detach().cpu())
    action_logits = torch.cat(action_logits)
    risk_logits = torch.cat(risk_logits)
    predicted_pose = torch.cat(predicted_pose)
    actions = torch.tensor(table.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(table.risk_id.iloc[query].to_numpy()).long()
    classification_result = classification_metrics(
        action_logits, risk_logits, actions, risks,
    )
    classification_result["safe_to_danger_rate"] = (
        classification_result["safe_to_danger"]
        / max(classification_result["safe_total"], 1)
    )
    target = torch.from_numpy(np.array(store.pose[query], dtype=np.float32))
    valid = torch.from_numpy(np.array(store.valid[query], dtype=bool))
    valid &= torch.isfinite(target).all(-1).all(-1)
    pose_result = retrieval_metrics(predicted_pose, target, valid, risks)
    result = {
        "run": "NOTIFI-AI-V2-CONTINUOUS-DANGER-SEALED-FINAL",
        "protocol": (
            "locked full-source artifact; final-only unseen subject/environment; "
            "support excluded from 239 queries"
        ),
        "artifact": str(options.artifact.resolve()),
        "artifact_sha256": sha256(options.artifact),
        "support_seed": int(options.support_seed),
        "counts": {
            "total": len(sealed), "absence": len(absence),
            "basic_support": len(basic), "warning_support": len(warning),
            "danger_support": len(danger), "query": len(query),
            "danger_expert_routed": routed,
        },
        "classification": classification_result,
        "pose": pose_result,
        "risk_threshold": float(branch["risk_threshold"]),
        "used_for_model_selection": False,
        "used_for_training_or_tuning": False,
        "query_labels_or_pose_gt_used_at_inference": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
