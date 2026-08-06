"""Evaluate fixed-TX dynamic-spectrum calibration before frozen KP4/KP10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import contract as C
from ..cal16_kp10 import (
    IdentitySpectrumCalibratedKP4,
    TARGET_CALIBRATION_SPLIT_SEED,
    assess_spectrum_calibration,
    fit_identity_spectrum_calibration,
    fit_site_balanced_reference,
    trial_dynamic_spectrum,
)
from ..calibration_quality import SAFE_CALIBRATION_CLASSES
from ..dataio.dataset import build_datasets
from ..quality import QualityWeightedDataset, protocol_audit_path
from ..trainer import set_seed
from .audit_kp10_paired_bootstrap import kp10_prediction
from .audit_motion_retrieval_oracle import _metric_batch
from .diagnose_observability import pose_only
from .evaluate_cal4_linkmap_kp10 import extract_features, load_coarse
from .evaluate_motion_retrieval_pose import _load_model
from .train_cal1_kp10 import (
    META_TRAIN_SITES,
    META_VALIDATION_SITES,
    add_paths,
    configure_work_root,
    prepare_custom,
    site_names,
    slice_cache,
    split_support_query,
)
from .train_cal3_kp10 import classification_with_preserved_risk
from .train_calibration_aware_v14 import subset_dataset


IDENTITY_MAPPING = tuple(range(C.N_LINKS))


def _support_spectra(dataset, positions_by_site: dict[str, np.ndarray],
                     sites: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    spectra, labels, site_ids = [], [], []
    for site_id, site in enumerate(sites):
        positions = positions_by_site[site]
        for start in range(0, len(positions), 8):
            samples = [dataset[int(position)] for position in positions[start:start + 8]]
            csi = torch.stack([sample["csi"] for sample in samples]).float()
            mask = torch.stack([sample["link_mask"] for sample in samples]).bool()
            spectra.append(trial_dynamic_spectrum(csi, mask))
            labels.append(torch.stack([sample["class_id"] for sample in samples]))
            site_ids.append(torch.full((len(samples),), site_id, dtype=torch.long))
    return torch.cat(spectra), torch.cat(labels), torch.cat(site_ids)


def _link_coverage(dataset, positions: np.ndarray) -> list[float]:
    masks = torch.stack([
        dataset[int(position)]["link_mask"] for position in positions
    ]).bool()
    active = masks.any(-1)
    denominator = active.sum().clamp_min(1)
    return [
        float((masks[..., link] & active).sum() / denominator)
        for link in range(C.N_LINKS)
    ]


def _fit_calibration(reference: dict, dataset, positions: np.ndarray,
                     max_gain: float, smoothing_width: int) -> tuple[dict, list[float]]:
    support = {"site": positions}
    spectra, labels, _ = _support_spectra(dataset, support, ("site",))
    calibration = fit_identity_spectrum_calibration(
        reference, spectra, labels,
        max_gain=max_gain, smoothing_width=smoothing_width,
    )
    return calibration, _link_coverage(dataset, positions)


def _cache_for_model(model, dataset, coarse, site: str, device: str,
                     protocol: str) -> dict:
    return extract_features(
        model, dataset, coarse, {site: IDENTITY_MAPPING}, device, protocol
    )


def _evaluate_pair(args, target, base_cache: dict, adapted_cache: dict,
                   prefix: Path, device: str) -> tuple[dict, dict]:
    base_data = prepare_custom(
        args, target, base_cache, prefix.with_name(prefix.name + "_base.pt"), device
    )
    adapted_data = prepare_custom(
        args, target, adapted_cache,
        prefix.with_name(prefix.name + "_adapted.pt"), device,
    )
    # Basic-pose support contains no danger boundary.  Raw calibration may
    # alter action/motion evidence, but cannot retune risk from safe prompts.
    adapted_data["risk_probability"] = base_data["risk_probability"]
    adapted_data["selector_risk_logits"] = base_data["selector_risk_logits"]
    base_pose = kp10_prediction(base_data, args, device).cpu()
    adapted_pose = kp10_prediction(adapted_data, args, device).cpu()
    action = base_data["target_class"]
    risk = base_data["target_risk"]
    return (
        {
            "pose": _metric_batch(
                base_pose, base_data["target_pose"], base_data["target_valid"], risk
            ),
            "classification": classification_with_preserved_risk(
                args, base_cache, base_cache, action, risk, device
            ),
        },
        {
            "pose": _metric_batch(
                adapted_pose, adapted_data["target_pose"],
                adapted_data["target_valid"], risk,
            ),
            "classification": classification_with_preserved_risk(
                args, adapted_cache, base_cache, action, risk, device
            ),
        },
    )


def _strength_score(metrics: list[dict]) -> float:
    def average(key: str) -> float:
        return float(np.mean([item["pose"][key] for item in metrics]))
    return (
        average("danger_pose_mpjpe_m")
        + 0.75 * average("danger_distal_mpjpe_m")
        + 0.25 * average("danger_endpoint_mpjpe_m")
        + 0.50 * average("mpjpe_m")
    )


def _mean_metric(metrics: list[dict], group: str, key: str) -> float:
    return float(np.mean([item[group][key] for item in metrics]))


def select_strength_on_held_sites(args, kp4, validation, support, query,
                                  reference, coarse, device: str) -> tuple[float, dict]:
    names = site_names(validation.index)
    base_by_site, calibration_by_site, target_by_site = {}, {}, {}
    audits = {}
    for site in META_VALIDATION_SITES:
        calibration, coverage = _fit_calibration(
            reference, validation, support[site], args.max_gain,
            args.smoothing_width,
        )
        audits[site] = {
            key: float(value) for key, value in calibration.items()
            if key != "gain" and key != "log_gain"
        }
        audits[site]["link_coverage"] = coverage
        calibration_by_site[site] = calibration
        local = query[names[query] == site]
        target = QualityWeightedDataset(
            subset_dataset(validation.target, local), protocol_audit_path(args.exp)
        )
        target_by_site[site] = target
        base_by_site[site] = _cache_for_model(
            kp4, target, coarse, site, device, f"CAL16 {site} base"
        )

    candidates = {}
    for strength in args.strengths:
        site_metrics = []
        for site in META_VALIDATION_SITES:
            wrapper = IdentitySpectrumCalibratedKP4(
                kp4, calibration_by_site[site]["gain"], strength,
                args.lowpass_window,
            ).to(device).eval()
            adapted = _cache_for_model(
                wrapper, target_by_site[site], coarse, site, device,
                f"CAL16 {site} strength {strength}",
            )
            _, current = _evaluate_pair(
                args, target_by_site[site], base_by_site[site], adapted,
                args.run_dir / f"source_{site}_{strength:.2f}", device,
            )
            site_metrics.append(current)
        candidates[f"strength_{strength:.2f}"] = {
            "score": _strength_score(site_metrics),
            "mean_overall_pose_m": _mean_metric(site_metrics, "pose", "mpjpe_m"),
            "mean_danger_pose_m": _mean_metric(
                site_metrics, "pose", "danger_pose_mpjpe_m"
            ),
            "mean_danger_distal_m": _mean_metric(
                site_metrics, "pose", "danger_distal_mpjpe_m"
            ),
            "mean_action_accuracy": _mean_metric(
                site_metrics, "classification", "action_accuracy"
            ),
            "sites": {
                site: metric for site, metric in zip(META_VALIDATION_SITES, site_metrics)
            },
        }
    baseline = candidates["strength_0.00"]
    eligible = [
        float(strength) for strength in args.strengths
        if (
            candidates[f"strength_{strength:.2f}"]["mean_overall_pose_m"]
            <= baseline["mean_overall_pose_m"] + 0.001
            and candidates[f"strength_{strength:.2f}"]["mean_action_accuracy"]
            >= baseline["mean_action_accuracy"] - 0.02
        )
    ]
    selected = min(
        eligible,
        key=lambda value: (
            candidates[f"strength_{value:.2f}"]["score"], -value
        ),
    ) if eligible else 0.0
    return float(selected), {
        "selected_strength": float(selected),
        "selection_uses_target_domain": False,
        "fixed_link_order": list(IDENTITY_MAPPING),
        "site_calibration": audits,
        "candidates": candidates,
    }


def evaluate_yja(args, kp4, reference, selected_strength: float,
                 coarse, device: str) -> tuple[dict, dict]:
    sealed = build_datasets(
        exp="sealed", fold="yja_E02", baseline=args.baseline, seed=args.seed
    )["test"]
    full = QualityWeightedDataset(sealed, None)
    pool, query = split_support_query(
        sealed.index, ("yja_E02",), args.target_reserve_per_class,
        TARGET_CALIBRATION_SPLIT_SEED,
    )
    support = pool["yja_E02"]
    calibration, coverage = _fit_calibration(
        reference, full, support, args.max_gain, args.smoothing_width
    )
    decision = assess_spectrum_calibration(calibration, coverage)
    target_all = QualityWeightedDataset(subset_dataset(sealed, query), None)
    base_all = _cache_for_model(
        kp4, target_all, coarse, "yja_E02", device, "CAL16 yja base"
    )
    wrapper = IdentitySpectrumCalibratedKP4(
        kp4, calibration["gain"], selected_strength, args.lowpass_window
    ).to(device).eval()
    adapted_all = _cache_for_model(
        wrapper, target_all, coarse, "yja_E02", device, "CAL16 yja calibrated"
    )
    pose_local = np.flatnonzero(
        sealed.index.iloc[query].task.to_numpy() == C.TASK_POSE
    )
    pose_target = QualityWeightedDataset(
        subset_dataset(sealed, query[pose_local]), None
    )
    local = torch.from_numpy(pose_local).long()
    base, adapted = _evaluate_pair(
        args, pose_target, slice_cache(base_all, local),
        slice_cache(adapted_all, local), args.run_dir / "yja_fixed_query", device,
    )
    # Classification includes absence trials, unlike pose metrics.
    action = torch.tensor(
        sealed.index.iloc[query].class_id.to_numpy(dtype=np.int64)
    )
    risk = torch.tensor(
        sealed.index.iloc[query].risk_id.to_numpy(dtype=np.int64)
    )
    base["classification"] = classification_with_preserved_risk(
        args, base_all, base_all, action, risk, device
    )
    adapted["classification"] = classification_with_preserved_risk(
        args, adapted_all, base_all, action, risk, device
    )
    audit = {
        key: float(value) for key, value in calibration.items()
        if key not in {"gain", "log_gain"}
    }
    audit.update({
        "link_coverage": coverage,
        "status": decision.status,
        "reason": decision.reason,
        "selected_source_strength": selected_strength,
        "accepted_for_normal_inference": (
            decision.status == "READY" and selected_strength > 0.0
        ),
    })
    result = {
        "support_trials": int(len(support)),
        "support_per_class": args.target_reserve_per_class,
        "query_trials": int(len(query)),
        "pose_query_trials": int(len(pose_local)),
        "danger_query_trials": int((risk == 2).sum()),
        "support_pose_gt_used": False,
        "query_used_for_calibration_or_selection": False,
        "calibration": audit,
        "base": base,
        "cal16": adapted,
    }
    state = {
        "run": "CAL16-IDSPEC-KP10",
        "site": "yja_E02",
        "deployable": audit["accepted_for_normal_inference"],
        "fixed_link_order": IDENTITY_MAPPING,
        "gain": calibration["gain"],
        "strength": selected_strength,
        "quality": audit,
        "support_rows": sealed.rows[support].tolist(),
        "support_class_ids": sealed.index.iloc[support].class_id.tolist(),
    }
    return result, state


def portable_reference(reference: dict) -> dict:
    return {
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in reference.items()
    }


def main() -> None:
    default_work = Path(
        r"C:\Users\jjeong\Documents\Playground"
        r"\NotiFi-CSI-to-Pose-robust\work_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=default_work)
    parser.add_argument("--exp", default="single_split_lmh_e01")
    parser.add_argument("--baseline", choices=("sub", "sub_z"), default="sub")
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--source-support-per-class", type=int, default=4)
    parser.add_argument("--target-reserve-per-class", type=int, default=4)
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=(0.0, 0.25, 0.50, 0.75, 1.0))
    parser.add_argument("--max-gain", type=float, default=1.8)
    parser.add_argument("--smoothing-width", type=int, default=9)
    parser.add_argument("--lowpass-window", type=int, default=31)
    parser.add_argument("--candidate-action-penalty", type=float, default=0.05)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--source-coarse", type=Path,
        default=default_work / "runs/kp1_v13s_coarse_single_split_lmh_e01.pt",
    )
    parser.add_argument(
        "--yja-coarse", type=Path,
        default=default_work / "runs/cal2_kp10_seed223_danger_gate"
        / "yja_e02_v13s_coarse.pt",
    )
    known, _ = parser.parse_known_args()
    add_paths(parser, known.work_root)
    args = parser.parse_args()
    configure_work_root(args.work_root)
    C.PROJECT_ROOT = args.work_root.parent
    args.run_dir = args.run_dir or args.work_root / "runs/cal16_idspec_kp10"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = build_datasets(exp=args.exp, baseline=args.baseline, seed=17)
    audit = protocol_audit_path(args.exp)
    train = QualityWeightedDataset(pose_only(datasets["train"]), audit)
    validation = QualityWeightedDataset(pose_only(datasets["val"]), audit)
    train_support, _ = split_support_query(
        train.index, META_TRAIN_SITES, args.source_support_per_class, args.seed
    )
    validation_support, validation_query = split_support_query(
        validation.index, META_VALIDATION_SITES,
        args.source_support_per_class, args.seed + 17,
    )
    source_spectra, source_labels, source_sites = _support_spectra(
        train, train_support, META_TRAIN_SITES
    )
    reference = fit_site_balanced_reference(
        source_spectra, source_labels, source_sites
    )
    kp4, _ = _load_model(
        args.work_root / "runs/kp4_dcc_staged_seed17/deployment_model.pt", device
    )
    source_coarse = load_coarse(args.source_coarse)
    selected_strength, source_selection = select_strength_on_held_sites(
        args, kp4, validation, validation_support, validation_query,
        reference, source_coarse, device,
    )

    # The target is opened only after every hyperparameter above is fixed.
    yja_coarse = load_coarse(args.yja_coarse)
    yja, deployment = evaluate_yja(
        args, kp4, reference, selected_strength, yja_coarse, device
    )
    checkpoint = {
        "run": "CAL16-IDSPEC-KP10",
        "source_reference": reference,
        "selected_strength": selected_strength,
        "max_gain": args.max_gain,
        "smoothing_width": args.smoothing_width,
        "lowpass_window": args.lowpass_window,
        "fixed_link_order": IDENTITY_MAPPING,
        "meta_train_sites": META_TRAIN_SITES,
        "meta_validation_sites": META_VALIDATION_SITES,
    }
    result = {
        "run": "CAL16-IDSPEC-KP10",
        "status": "candidate",
        "contract": {
            "fixed_physical_link_order": ["TX1_South", "TX2_West", "TX3_East"],
            "link_permutation_forbidden": True,
            "support_uses_csi_and_known_basic_action_only": True,
            "support_pose_gt_used": False,
            "target_query_used_for_training_or_selection": False,
            "strength_selected_on": list(META_VALIDATION_SITES),
            "target_opened_after_source_selection": True,
        },
        "source_meta_validation": source_selection,
        "yja_e02": yja,
    }
    torch.save(checkpoint, args.run_dir / "source_calibrator.pt")
    torch.save(deployment, args.run_dir / "calibration_candidate.pt")
    (args.run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "run": result["run"],
        "selected_strength": selected_strength,
        "calibration_quality": yja["calibration"],
        "base": yja["base"],
        "cal16": yja["cal16"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
