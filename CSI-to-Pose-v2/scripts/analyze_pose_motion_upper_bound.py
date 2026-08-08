"""GT pose motion descriptor의 source 사람-LOSO 행동 분류 상한을 측정한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from notifi_pose.cal13 import (  # noqa: E402
    pose_motion_descriptor,
    temporal_motion_signature,
)


def _metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """action 정확도와 실제 등장 class 기준 macro-F1을 반환한다."""
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


def main() -> None:
    """GT descriptor를 추출하고 random 및 subject-LOSO probe를 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=500)
    options = parser.parse_args()
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    rows = base.select_source_rows(index)
    if "yja" in set(index.subject.iloc[rows].astype(str)):
        raise RuntimeError("sealed yja entered GT motion upper-bound probe")
    pose = np.load(WORK / "cache/pose_rel.npy", mmap_mode="r")
    valid = np.load(WORK / "cache/valid.npy", mmap_mode="r")
    parts = []
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        batch_pose = torch.from_numpy(np.asarray(pose[batch]).copy())
        batch_valid = torch.from_numpy(np.asarray(valid[batch]).copy()).bool()
        descriptor = pose_motion_descriptor(batch_pose, batch_valid)
        parts.append(
            temporal_motion_signature(descriptor, batch_valid, bins=8).numpy()
        )
    features = np.concatenate(parts).astype(np.float32)
    action = index.class_id.iloc[rows].to_numpy(dtype=np.int64)
    risk = index.risk_id.iloc[rows].to_numpy(dtype=np.int64)
    subject = index.subject.iloc[rows].astype(str).to_numpy()
    model = ExtraTreesClassifier(
        n_estimators=options.trees, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=22001,
    )
    random_prediction = cross_val_predict(
        model, features, action,
        cv=StratifiedKFold(5, shuffle=True, random_state=22001),
        n_jobs=1,
    )
    folds = {}
    loso_prediction = np.full_like(action, -1)
    risk_prediction = np.full_like(risk, -1)
    for held_out in sorted(set(subject)):
        train = subject != held_out
        test = ~train
        action_model = ExtraTreesClassifier(
            n_estimators=options.trees, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=22001,
        )
        action_model.fit(features[train], action[train])
        loso_prediction[test] = action_model.predict(features[test])
        risk_model = ExtraTreesClassifier(
            n_estimators=options.trees, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=22002,
        )
        risk_model.fit(features[train], risk[train])
        risk_prediction[test] = risk_model.predict(features[test])
        folds[held_out] = {
            "action": _metrics(action[test], loso_prediction[test]),
            "risk": _metrics(risk[test], risk_prediction[test]),
            "trials": int(test.sum()),
        }
    result = {
        "feature": "GT pose 10-channel descriptor, moments + 8 ordered bins",
        "rows": int(len(rows)),
        "dimensions": int(features.shape[1]),
        "random_5fold_action": _metrics(action, random_prediction),
        "subject_loso_action": _metrics(action, loso_prediction),
        "subject_loso_risk": _metrics(risk, risk_prediction),
        "folds": folds,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
