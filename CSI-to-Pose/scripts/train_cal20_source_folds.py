"""CAL20-RELATIVE-MOTION-DG를 yja 없이 source subject-LOSO로 학습한다."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import source_calibration_data as base  # noqa: E402
from notifi_pose import contract as C  # noqa: E402
from notifi_pose.cal12 import (  # noqa: E402
    CAL12PhysicsDG,
    cross_site_supervised_contrastive,
)
from notifi_pose.cal13 import (  # noqa: E402
    pose_motion_descriptor,
    shift_robust_motion_loss,
)
from notifi_pose.cal20 import CAL20RelativeMotionDG  # noqa: E402
from notifi_pose.meta_calibration import MOTION_PROMPT_CLASSES  # noqa: E402
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402


SOURCE_SITES = {
    "ajh_E01", "ajh_E02", "ajh_E03",
    "mhw_E01", "mhw_E02", "mhw_E03", "lmh_E01",
}


class PoseStore:
    """source GT를 학습 보조교사로만 읽고 평가·추론 입력에서는 분리한다."""

    def __init__(self, cache: Path):
        self.pose = np.load(cache / "pose_rel.npy", mmap_mode="r")
        self.valid = np.load(cache / "valid.npy", mmap_mode="r")

    def get(
        self, rows: np.ndarray, device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """선택한 source row의 자세와 유효 mask만 장치로 복사한다."""
        pose = torch.from_numpy(np.asarray(self.pose[rows]).copy()).to(device)
        valid = torch.from_numpy(np.asarray(self.valid[rows]).copy()).to(device)
        return pose, valid


def nested_site_split(
    all_sites: list[str], held_out_subject: str,
) -> tuple[list[str], list[str], list[str]]:
    """외부 평가 사람을 숨긴 채 남은 사람의 E03만 epoch 선택에 사용한다."""
    outer_test = [
        site for site in all_sites if site.startswith(f"{held_out_subject}_")
    ]
    candidates = [site for site in all_sites if site not in outer_test]
    inner_validation = [site for site in candidates if site.endswith("_E03")]
    if not inner_validation:
        # E03가 없는 구성에서도 다중 환경 사람의 마지막 site 하나만 고정한다.
        multi_site_subjects = sorted({
            site.split("_")[0] for site in candidates
            if sum(
                other.startswith(f"{site.split('_')[0]}_")
                for other in candidates
            ) > 1
        })
        if not multi_site_subjects:
            raise RuntimeError("nested validation requires a multi-site source subject")
        subject = multi_site_subjects[0]
        inner_validation = [
            sorted(
                site for site in candidates if site.startswith(f"{subject}_")
            )[-1]
        ]
    inner_train = [site for site in candidates if site not in inner_validation]
    if len(inner_train) < 2:
        raise RuntimeError(f"not enough nested training sites: {inner_train}")
    return inner_train, inner_validation, outer_test


def site_batches(
    rows: np.ndarray,
    index: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """한 site에서 action 균형 query batch를 만든다."""
    return base.balanced_batches(rows, index, batch_size, seed)


def risk_consistency_loss(
    direct_logits: torch.Tensor,
    action_risk_logits: torch.Tensor,
) -> torch.Tensor:
    """독립 위험 head와 17행동 합산 위험도가 모순되지 않게 맞춘다."""
    direct = direct_logits.log_softmax(-1)
    action = action_risk_logits.log_softmax(-1)
    middle = torch.logsumexp(
        torch.stack((direct, action), dim=0), dim=0
    ) - torch.log(torch.tensor(2.0, device=direct.device))
    return 0.5 * (
        F.kl_div(middle, direct.exp(), reduction="batchmean")
        + F.kl_div(middle, action.exp(), reduction="batchmean")
    )


@torch.no_grad()
def evaluate_site(
    model: CAL12PhysicsDG,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
    device: str,
    seed: int,
) -> dict:
    """고정 support/absence만 사용해 한 미관측 source site를 평가한다."""
    model.eval()
    rows = base.site_rows(selected_rows, sites, site)
    support = base.select_support(rows, index, seed)
    absence = base.select_absence(site, index, seed + 1)
    support_set = set(support.tolist())
    query = np.asarray([row for row in rows if row not in support_set], dtype=np.int64)
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    support_labels = torch.tensor(
        index.class_id.iloc[support].to_numpy(), device=device
    )
    action_logits, risk_logits, reliability, link_weights = [], [], [], []
    for start in range(0, len(query), 24):
        batch = query[start:start + 24]
        query_csi, query_mask = store.get(batch, device)
        output = model(
            query_csi, query_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        action_logits.append(output["action_logits"].cpu())
        risk_logits.append(output["risk_logits"].cpu())
        reliability.append(output["motion_reliability"].cpu())
        link_weights.append(output["query_link_weight"].cpu())
    labels = torch.tensor(index.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(index.risk_id.iloc[query].to_numpy()).long()
    metrics = classification_metrics(
        torch.cat(action_logits), torch.cat(risk_logits), labels, risks
    )
    metrics.update({
        "site": site,
        "support_trials": int(len(support)),
        "absence_trials": int(len(absence)),
        "motion_reliability": torch.stack(reliability).mean(0).tolist(),
        "link_weight_mean": torch.cat(link_weights).mean((0, 1)).tolist(),
    })
    return metrics


def build_episode(
    site: str,
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    device: str,
    seed: int,
    batch_size: int,
) -> dict:
    """한 epoch 동안 재사용할 site별 calibration support와 query batch를 준비한다."""
    rows = base.site_rows(selected_rows, sites, site)
    support = base.select_support(rows, index, seed)
    absence = base.select_absence(site, index, seed + 1)
    support_set = set(support.tolist())
    query = np.asarray([row for row in rows if row not in support_set], dtype=np.int64)
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    return {
        "site": site,
        "support_csi": support_csi,
        "support_mask": support_mask,
        "support_labels": torch.tensor(
            index.class_id.iloc[support].to_numpy(), device=device
        ),
        "absence_csi": absence_csi,
        "absence_mask": absence_mask,
        "batches": site_batches(query, index, batch_size, seed + 2),
    }


def train_model(
    store: base.RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    train_sites: list[str],
    validation_sites: list[str] | None,
    epochs: int,
    device: str,
    seed: int,
    batch_size: int,
    relative_support: bool,
    use_doppler: bool,
    phase_strength: float,
    pose_store: PoseStore | None = None,
    lambda_motion_grounding: float = 0.0,
    fixed_swa: bool = False,
    swa_start: int = 8,
) -> tuple[CAL12PhysicsDG, list[dict], dict | None]:
    """두 site씩 묶은 episode로 행동 불변성과 calibration을 동시에 학습한다."""
    model_kwargs = {
        "hidden": 64, "width": 192, "domains": len(train_sites),
        "dropout": 0.10, "domain_grl": 1.0,
        "relative_support": relative_support,
        "use_doppler": use_doppler,
        "phase_strength": phase_strength,
    }
    model = CAL20RelativeMotionDG(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-3)
    risk_weight = torch.tensor((0.8, 1.0, 1.25), device=device)
    risk_weight /= risk_weight.mean()
    history: list[dict] = []
    best_score = -float("inf")
    best_state = None
    stale = 0
    patience = 8
    swa_state = None
    swa_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        episodes = {
            site: build_episode(
                site, store, index, selected_rows, sites, device,
                seed * 10_000 + epoch * 101 + number * 7,
                batch_size,
            )
            for number, site in enumerate(train_sites)
        }
        schedule = []
        for site, episode in episodes.items():
            schedule.extend((site, number) for number in range(len(episode["batches"])))
        rng.shuffle(schedule)
        losses = []
        diagnostics = []
        for step in range(0, len(schedule), 2):
            chosen = schedule[step:step + 2]
            if len(chosen) == 1 or chosen[0][0] == chosen[1][0]:
                # 매 step에 서로 다른 site를 넣어 cross-site positive를 만들 수 있게 한다.
                alternatives = [site for site in train_sites if site != chosen[0][0]]
                other = str(rng.choice(alternatives))
                replacement = (
                    other, int(rng.integers(len(episodes[other]["batches"])))
                )
                if len(chosen) == 1:
                    chosen.append(replacement)
                else:
                    chosen[1] = replacement
            outputs = []
            all_labels = []
            all_risks = []
            all_domains = []
            all_rows = []
            for site, batch_number in chosen:
                episode = episodes[site]
                batch = episode["batches"][batch_number]
                query_csi, query_mask = store.get(batch, device)
                if rng.random() < 0.75:
                    (absence_csi, absence_mask), (support_csi, support_mask), (
                        query_csi, query_mask
                    ) = base.augment_site(
                        [
                            (episode["absence_csi"], episode["absence_mask"]),
                            (episode["support_csi"], episode["support_mask"]),
                            (query_csi, query_mask),
                        ],
                        seed=seed * 1_000_000 + epoch * 10_000 + step,
                    )
                else:
                    absence_csi, absence_mask = (
                        episode["absence_csi"], episode["absence_mask"]
                    )
                    support_csi, support_mask = (
                        episode["support_csi"], episode["support_mask"]
                    )
                output = model(
                    query_csi, query_mask,
                    support_csi, support_mask, episode["support_labels"],
                    absence_csi, absence_mask,
                )
                domain = train_sites.index(site)
                outputs.append(output)
                all_labels.append(torch.tensor(
                    index.class_id.iloc[batch].to_numpy(), device=device
                ))
                all_risks.append(torch.tensor(
                    index.risk_id.iloc[batch].to_numpy(), device=device
                ))
                all_domains.append(torch.full(
                    (len(batch),), domain, dtype=torch.long, device=device
                ))
                all_rows.append(batch)
                if step == 0:
                    model.canonicalizer.update_source_scale(
                        output["amp_scale"], output["phase_scale"], momentum=0.98
                    )

            action_logits = torch.cat([item["action_logits"] for item in outputs])
            risk_logits = torch.cat([item["risk_logits"] for item in outputs])
            direct_risk = torch.cat([
                item["direct_risk_logits"] for item in outputs
            ])
            action_risk = torch.cat([
                item["action_risk_logits"] for item in outputs
            ])
            embedding = torch.cat([item["embedding"] for item in outputs])
            domain_logits = torch.cat([item["domain_logits"] for item in outputs])
            base_action_logits = torch.cat([
                item["base_action_logits"] for item in outputs
            ])
            base_risk_logits = torch.cat([
                item["base_risk_logits"] for item in outputs
            ])
            action_residual = torch.cat([
                item["action_residual"] for item in outputs
            ])
            risk_residual = torch.cat([
                item["risk_residual"] for item in outputs
            ])
            adapter_gate = torch.cat([
                item["adapter_gate"] for item in outputs
            ])
            labels = torch.cat(all_labels)
            risks = torch.cat(all_risks)
            domains = torch.cat(all_domains)

            if pose_store is not None:
                pose, pose_valid = pose_store.get(
                    np.concatenate(all_rows), device
                )
                motion_target = pose_motion_descriptor(pose, pose_valid)
                predicted_motion = torch.cat([
                    item["pose_motion"] for item in outputs
                ])
                motion_grounding, selected_shift = shift_robust_motion_loss(
                    predicted_motion, motion_target, pose_valid, max_shift=6
                )
            else:
                motion_grounding = embedding.sum() * 0.0
                selected_shift = embedding.new_zeros(())

            action_loss = F.cross_entropy(action_logits, labels)
            risk_loss = F.cross_entropy(risk_logits, risks, weight=risk_weight)
            direct_risk_loss = F.cross_entropy(
                direct_risk, risks, weight=risk_weight
            )
            domain_loss = F.cross_entropy(domain_logits, domains)
            contrastive = cross_site_supervised_contrastive(
                embedding, labels, domains
            )
            consistency = risk_consistency_loss(direct_risk, action_risk)
            base_action_loss = F.cross_entropy(base_action_logits, labels)
            base_risk_loss = F.cross_entropy(
                base_risk_logits, risks, weight=risk_weight
            )
            calibrated_per_item = (
                F.cross_entropy(action_logits, labels, reduction="none")
                + 0.70 * F.cross_entropy(
                    risk_logits, risks, weight=risk_weight, reduction="none"
                )
            )
            primary_task_loss = action_loss + 0.70 * risk_loss
            base_per_item = (
                F.cross_entropy(base_action_logits, labels, reduction="none")
                + 0.70 * F.cross_entropy(
                    base_risk_logits, risks, weight=risk_weight, reduction="none"
                )
            )
            no_harm = F.relu(calibrated_per_item - base_per_item).mean()
            residual_penalty = (
                action_residual.square().mean() + risk_residual.square().mean()
            )
            link_entropy = torch.stack([
                -(item["query_link_weight"].clamp_min(1e-7).log()
                  * item["query_link_weight"]).sum(-1).mean()
                for item in outputs
            ]).mean()
            loss = (
                primary_task_loss
                + 0.20 * direct_risk_loss
                + 0.12 * domain_loss
                + 0.15 * contrastive
                + 0.08 * consistency
                + 0.30 * (base_action_loss + 0.70 * base_risk_loss)
                + 0.20 * no_harm
                + 0.005 * residual_penalty
                + 0.005 * adapter_gate.mean()
                - 0.01 * link_entropy
                + lambda_motion_grounding * motion_grounding
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            diagnostics.append({
                "action": float(action_loss.detach()),
                "risk": float(risk_loss.detach()),
                "domain": float(domain_loss.detach()),
                "contrastive": float(contrastive.detach()),
                "motion_grounding": float(motion_grounding.detach()),
                "motion_shift": float(selected_shift),
            })

        record: dict = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_components": {
                key: float(np.mean([item[key] for item in diagnostics]))
                for key in diagnostics[0]
            },
            "risk_fusion": float(torch.sigmoid(model.risk_fusion).detach()),
            "source_amp_scale": model.canonicalizer.source_amp_scale.tolist(),
            "source_phase_scale": model.canonicalizer.source_phase_scale.tolist(),
        }
        if hasattr(model, "hierarchy_logit"):
            record["hierarchy_strength"] = float(
                torch.sigmoid(model.hierarchy_logit).detach()
            )
        if fixed_swa and epoch >= swa_start:
            current_state = model.state_dict()
            if swa_state is None:
                swa_state = {
                    key: value.detach().cpu().clone()
                    for key, value in current_state.items()
                }
            else:
                for key, value in current_state.items():
                    if value.is_floating_point():
                        swa_state[key].lerp_(
                            value.detach().cpu(), 1.0 / float(swa_count + 1)
                        )
                    else:
                        swa_state[key].copy_(value.detach().cpu())
            swa_count += 1
        if validation_sites:
            validation = {
                site: evaluate_site(
                    model, store, index, selected_rows, sites,
                    site, device, seed=17017,
                )
                for site in validation_sites
            }
            site_diagnostics = {
                site: cal12_site_selection_score(metrics)
                for site, metrics in validation.items()
            }
            scores = [item["score"] for item in site_diagnostics.values()]
            mean_score = float(np.mean(scores))
            worst_score = float(np.min(scores))
            selection_score = 0.5 * mean_score + 0.5 * worst_score
            record.update({
                "validation": validation,
                "validation_diagnostics": site_diagnostics,
                "mean_site_score": mean_score,
                "worst_site_score": worst_score,
                "selection_score": selection_score,
            })
            if selection_score > best_score + 1e-5:
                best_score = selection_score
                best_state = copy.deepcopy({
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                })
                stale = 0
            else:
                stale += 1
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation_sites and not fixed_swa and stale >= patience:
            break

    if fixed_swa:
        if swa_state is None:
            raise RuntimeError("SWA did not receive any epoch")
        model.load_state_dict(swa_state)
        if validation_sites:
            validation = {
                site: evaluate_site(
                    model, store, index, selected_rows, sites,
                    site, device, seed=17017,
                )
                for site in validation_sites
            }
            diagnostics = {
                site: cal12_site_selection_score(metrics)
                for site, metrics in validation.items()
            }
            scores = [item["score"] for item in diagnostics.values()]
            best = {
                "epoch": epochs,
                "selection": "fixed_swa_no_validation_selection",
                "swa_start": swa_start,
                "swa_snapshots": swa_count,
                "validation": validation,
                "validation_diagnostics": diagnostics,
                "mean_site_score": float(np.mean(scores)),
                "worst_site_score": float(np.min(scores)),
                "selection_score": float(
                    0.5 * np.mean(scores) + 0.5 * np.min(scores)
                ),
            }
        else:
            best = None
    elif best_state is not None:
        model.load_state_dict(best_state)
        best = max(history, key=lambda item: item["selection_score"])
    else:
        best = None
    return model, history, best


def cal12_site_selection_score(metrics: dict) -> dict[str, float]:
    """행동이 무너지면 danger 점수만으로 checkpoint가 선택되지 않게 계산한다."""
    safe_specificity = 1.0 - (
        metrics["safe_to_danger"] / max(metrics["safe_total"], 1)
    )
    danger_recall = metrics["danger_recall"]
    danger_balance = (
        2.0 * danger_recall * safe_specificity
        / max(danger_recall + safe_specificity, 1e-8)
    )
    action_utility = (
        0.70 * metrics["action_macro_f1"]
        + 0.30 * metrics["action_accuracy"]
    )
    risk_utility = (
        0.50 * metrics["risk_macro_f1"] + 0.50 * danger_balance
    )
    score = (
        3.0 * action_utility
        + np.sqrt(max(action_utility * risk_utility, 0.0))
        + 0.25 * metrics["danger_action_accuracy"]
    )
    return {
        "score": float(score),
        "safe_specificity": float(safe_specificity),
        "danger_balance": float(danger_balance),
        "action_utility": float(action_utility),
        "risk_utility": float(risk_utility),
    }


def main() -> None:
    """source subject-LOSO와 전체 source deployment 학습을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--disable-relative-support", action="store_true")
    parser.add_argument("--use-doppler", action="store_true")
    parser.add_argument("--phase-strength", type=float, default=1.0)
    parser.add_argument("--motion-grounding", action="store_true")
    parser.add_argument("--lambda-motion-grounding", type=float, default=0.30)
    parser.add_argument("--fixed-swa", action="store_true")
    parser.add_argument("--swa-start", type=int, default=8)
    options = parser.parse_args()
    run = options.run_dir
    run.mkdir(parents=True, exist_ok=True)

    base.ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
    base.PROMPT_SHOTS = {class_id: 2 for class_id in MOTION_PROMPT_CLASSES}
    torch.manual_seed(12012)
    np.random.seed(12012)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    selected_rows = base.select_source_rows(index)
    selected_index = index.iloc[selected_rows]
    if "yja" in set(selected_index.subject.astype(str)):
        raise RuntimeError("sealed yja must not appear in CAL20 source rows")
    sites = (selected_index.subject + "_" + selected_index.environment).to_numpy()
    all_sites = sorted(set(sites.tolist()))
    if set(all_sites) != SOURCE_SITES:
        raise RuntimeError(f"unexpected CAL20 source sites: {all_sites}")

    absence_rows = np.concatenate([
        np.flatnonzero((
            (index.subject == site.split("_")[0])
            & (index.environment == site.split("_")[1])
            & (index.task == C.TASK_CLS)
            & (index.class_id == 6)
            & index.cache_ok
        ).to_numpy())
        for site in all_sites
    ])
    store = base.RawStore(index, np.concatenate((selected_rows, absence_rows)))
    pose_store = PoseStore(WORK / "cache") if options.motion_grounding else None

    fold_results = {}
    best_epochs = []
    for fold_number, held_out in enumerate(("ajh", "mhw", "lmh")):
        train_sites, validation_sites, outer_test_sites = nested_site_split(
            all_sites, held_out
        )
        checkpoint_path = run / f"selection_{held_out}.pt"
        if options.skip_existing and checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("protocol_version") != "nested_source_v1":
                raise RuntimeError(
                    f"legacy checkpoint cannot be resumed safely: {checkpoint_path}"
                )
            best = checkpoint["best"]
            best_epochs.append(int(best["epoch"]))
            fold_results[held_out] = {
                "train_sites": train_sites,
                "inner_validation_sites": validation_sites,
                "outer_test_sites": outer_test_sites,
                "best": best,
                "outer_test": checkpoint["outer_test"],
                "history": [],
                "resumed_from_checkpoint": True,
            }
            continue
        model, history, best = train_model(
            store, index, selected_rows, sites,
            train_sites, validation_sites,
            options.epochs, device, 12012 + fold_number * 101,
            options.batch_size,
            not options.disable_relative_support,
            options.use_doppler,
            options.phase_strength,
            pose_store,
            options.lambda_motion_grounding,
            options.fixed_swa,
            options.swa_start,
        )
        if best is None:
            raise RuntimeError(f"{held_out} fold did not produce a checkpoint")
        outer_test = {
            site: evaluate_site(
                model, store, index, selected_rows, sites,
                site, device, seed=17017,
            )
            for site in outer_test_sites
        }
        best_epochs.append(int(best["epoch"]))
        fold_results[held_out] = {
            "train_sites": train_sites,
            "inner_validation_sites": validation_sites,
            "outer_test_sites": outer_test_sites,
            "best": best,
            "outer_test": outer_test,
            "history": history,
        }
        torch.save({
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "held_out_subject": held_out,
            "best": best,
            "outer_test": outer_test,
            "protocol_version": "nested_source_v1",
            "outer_holdout_used_for_selection": False,
            "inner_validation_used_for_selection": not options.fixed_swa,
            "target_subject_used": False,
            "sealed_yja_used": False,
        }, checkpoint_path)

    locked_epochs = max(1, int(round(float(np.median(best_epochs)))))
    deployment, deployment_history, _ = train_model(
        store, index, selected_rows, sites,
        all_sites, None, locked_epochs, device, 13012, options.batch_size,
        not options.disable_relative_support,
        options.use_doppler,
        options.phase_strength,
        pose_store,
        options.lambda_motion_grounding,
        options.fixed_swa,
        options.swa_start,
    )
    source_metrics = {
        site: evaluate_site(
            deployment, store, index, selected_rows, sites,
            site, device, seed=17017,
        )
        for site in all_sites
    }
    torch.save({
        "model": deployment.state_dict(),
        "model_config": deployment.model_config(),
        "locked_epochs": locked_epochs,
        "source_sites": all_sites,
        "support_contract": {
            "absence_trials": 2,
            "prompt_classes": list(MOTION_PROMPT_CLASSES),
            "shots_per_prompt": 2,
            "target_pose_gt_used": False,
            "source_pose_gt_training_only": options.motion_grounding,
        },
        "selection_protocol": (
            "fixed_source_swa_v1" if options.fixed_swa
            else "nested_source_subject_loso_v1"
        ),
        "outer_holdout_used_for_selection": False,
        "inner_validation_used_for_selection": not options.fixed_swa,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }, run / "deployment_model.pt")
    result = {
        "run": "CAL20-RELATIVE-MOTION-DG",
        "device": device,
        "fold_results": fold_results,
        "fold_best_epochs": best_epochs,
        "locked_epochs": locked_epochs,
        "source_metrics": source_metrics,
        "deployment_history": deployment_history,
        "selection_protocol": (
            "fixed_source_swa_v1" if options.fixed_swa
            else "nested_source_subject_loso_v1"
        ),
        "outer_holdout_used_for_selection": False,
        "inner_validation_used_for_selection": not options.fixed_swa,
        "source_pose_gt_training_only": options.motion_grounding,
        "query_pose_gt_used": False,
        "target_subject_used": False,
        "sealed_yja_used": False,
    }
    (run / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "fold_best_epochs": best_epochs,
        "locked_epochs": locked_epochs,
        "fold_best": {
            subject: value["best"] for subject, value in fold_results.items()
        },
        "source_metrics": source_metrics,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
