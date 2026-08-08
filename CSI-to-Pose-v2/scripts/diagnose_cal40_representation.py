"""CAL40 두 encoder의 행동 정보와 source 사람·site 지문을 비교한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from calibrate_cal17_style_transport import embed_site  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.model_factory import build_calibration_model  # noqa: E402
from train_cal20_source_folds import SOURCE_SITES  # noqa: E402


def classifier() -> object:
    """모든 probe에 동일한 표준화 선형 분류기를 만든다."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=22012,
        ),
    )


def random_probe(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """stratified 5-fold로 embedding에서 범주가 선형 분리되는지 측정한다."""
    predicted = cross_val_predict(
        classifier(), features, labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=22012),
        n_jobs=1,
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(
            labels, predicted, average="macro", zero_division=0,
        )),
    }


def grouped_action_probe(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
) -> dict:
    """downstream probe에서 한 사람 전체를 숨겨 행동 전달성을 측정한다."""
    prediction = np.full(len(labels), -1, dtype=np.int64)
    folds = {}
    for held_out in sorted(set(groups.tolist())):
        test = groups == held_out
        model = classifier()
        model.fit(features[~test], labels[~test])
        prediction[test] = model.predict(features[test])
        folds[str(held_out)] = {
            "accuracy": float(accuracy_score(labels[test], prediction[test])),
            "macro_f1": float(f1_score(
                labels[test], prediction[test], average="macro",
                zero_division=0,
            )),
            "trials": int(test.sum()),
        }
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(
            labels, prediction, average="macro", zero_division=0,
        )),
        "folds": folds,
    }


def load_model(path: Path, device: str):
    """봉인 target을 쓰지 않은 all-source deployment checkpoint를 복원한다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "outer_holdout_used_for_selection", "target_subject_used",
        "sealed_yja_used",
    ):
        if checkpoint.get(key) is not False:
            raise RuntimeError(f"unclean checkpoint {path}: {key}")
    model = build_calibration_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main() -> None:
    """두 checkpoint의 source-only embedding probe를 같은 support로 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-seed", type=int, default=17017)
    parser.add_argument("--absence-trials", type=int, default=12)
    options = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected = index.iloc[selected_rows]
    if "yja" in set(selected.subject.astype(str)):
        raise RuntimeError("sealed yja cannot enter representation diagnosis")
    sites = (selected.subject + "_" + selected.environment).to_numpy()
    if set(sites.tolist()) != SOURCE_SITES:
        raise RuntimeError("unexpected source site contract")
    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS)
            & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in sorted(SOURCE_SITES)
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    reports = {}
    for name, path in (
        ("cal60_grl1", options.primary),
        ("cal66_grl0", options.secondary),
    ):
        model = load_model(path, device)
        payloads = [
            embed_site(
                model, store, index, selected_rows, sites, site, device,
                options.support_seed, options.support_seed + 1,
                2, None, options.absence_trials,
            )
            for site in sorted(SOURCE_SITES)
        ]
        embedding = torch.cat([
            payload["embedding"].detach().cpu() for payload in payloads
        ]).numpy()
        action = torch.cat([
            payload["labels"].detach().cpu() for payload in payloads
        ]).numpy()
        subject = np.concatenate([
            np.full(len(payload["labels"]), site.split("_")[0], dtype=object)
            for site, payload in zip(sorted(SOURCE_SITES), payloads)
        ])
        site_label = np.concatenate([
            np.full(len(payload["labels"]), site, dtype=object)
            for site, payload in zip(sorted(SOURCE_SITES), payloads)
        ])
        reports[name] = {
            "embedding_dimensions": int(embedding.shape[1]),
            "trials": int(len(embedding)),
            "random_action_probe": random_probe(embedding, action),
            "random_subject_probe": random_probe(embedding, subject),
            "random_site_probe": random_probe(embedding, site_label),
            "posthoc_subject_holdout_action_probe_encoder_seen_all_sources": grouped_action_probe(
                embedding, action, subject,
            ),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(name, json.dumps(reports[name], ensure_ascii=False), flush=True)
    result = {
        "run": "A51-CAL40-REPRESENTATION-DIAGNOSIS",
        "protocol": (
            "all-source encoder diagnostic; encoder has seen all source subjects; "
            "posthoc probes are not unseen evaluation; no model selection"
        ),
        "models": reports,
        "support_seed": options.support_seed,
        "target_subject_used": False,
        "sealed_yja_used": False,
        "query_labels_used_for_probe_only": True,
        "probe_used_for_model_selection": False,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
