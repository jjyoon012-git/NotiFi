"""Train raw-CSI meta calibration without exposing the target subject."""

from __future__ import annotations

import json
import math
import os
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
CODE = PROJECT
WORK = Path(os.environ.get("NOTIFI_WORK_ROOT", PROJECT / "work_v2"))
RUN = WORK / "runs/cal8_raw_source_dynamic_anchor_v2"
os.environ.setdefault("NOTIFI_WORK_ROOT", str(WORK))
sys.path.insert(0, str(CODE))

from notifi_pose import contract as C  # noqa: E402
from notifi_pose.linkqc import link_mask_per_trial  # noqa: E402
from notifi_pose.meta_calibration import (  # noqa: E402
    MOTION_PROMPT_CLASSES,
    PROMPT_CLASSES,
    RawSupportConditionedModel,
)
from notifi_pose.tools.train_dynamic_motion import classification_metrics  # noqa: E402


PROMPT_SHOTS = {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 7: 1, 8: 1}
ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES


class RawStore:
    """학습에 필요한 cache 행만 RAM에 올리고 trial 링크 품질 마스크를 적용한다."""

    def __init__(self, index: pd.DataFrame, rows: np.ndarray):
        rows = np.asarray(sorted(set(int(row) for row in rows)), dtype=np.int64)
        csi = np.load(WORK / "cache/csi_iq.npy", mmap_mode="r")
        mask = np.load(WORK / "cache/link_mask.npy", mmap_mode="r")
        self.rows = rows
        self.position = {int(row): position for position, row in enumerate(rows)}
        print(f"[cache] loading {len(rows)} raw trials into RAM", flush=True)
        self.csi = torch.from_numpy(np.array(csi[rows], dtype=np.float16))
        self.mask = torch.from_numpy(np.array(mask[rows], dtype=bool))

        trial_quality = link_mask_per_trial()
        usable = np.zeros((len(rows), C.N_LINKS), dtype=bool)
        for local, row in enumerate(rows):
            trial_id = str(index.iloc[row].trial_id)
            if trial_id in trial_quality.index:
                usable[local] = trial_quality.loc[trial_id].to_numpy(dtype=bool)
            else:
                usable[local] = True
        self.mask &= torch.from_numpy(usable)[:, None]

    def get(
        self,
        rows: torch.Tensor | np.ndarray | list[int],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """전역 cache 행 번호를 모델 입력 tensor로 변환한다."""
        local = torch.tensor([
            self.position[int(row)] for row in np.asarray(rows, dtype=np.int64)
        ]).long()
        return (
            self.csi[local].to(device, non_blocking=True).float(),
            self.mask[local].to(device, non_blocking=True),
        )


def select_support(
    rows: np.ndarray,
    index: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """기본동작 네 종류에서 두 trial씩 뽑아 8-shot support를 만든다."""
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in ACTIVE_PROMPT_CLASSES:
        candidates = rows[index.class_id.iloc[rows].to_numpy() == class_id]
        candidates = candidates[np.argsort(index.trial_id.iloc[candidates].to_numpy())]
        shots = PROMPT_SHOTS[class_id]
        if len(candidates) < shots:
            raise RuntimeError(f"class {class_id} has too few support trials")
        selected.extend(rng.permutation(candidates)[:shots].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def select_absence(
    site: str,
    index: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """현재 사이트의 absence trial 중 두 개만 빈방 기준선으로 사용한다."""
    subject, environment = site.split("_")
    keep = (
        (index.subject == subject)
        & (index.environment == environment)
        & (index.task == C.TASK_CLS)
        & (index.class_id == 6)
        & index.cache_ok
    )
    candidates = np.flatnonzero(keep.to_numpy())
    candidates = candidates[np.argsort(index.trial_id.iloc[candidates].to_numpy())]
    if len(candidates) < 2:
        raise RuntimeError(f"{site} has fewer than two absence trials")
    return np.random.default_rng(seed).permutation(candidates)[:2]


def class_weights(labels: torch.Tensor, classes: int) -> torch.Tensor:
    """희소 class가 다수 class에 묻히지 않도록 역빈도 가중치를 계산한다."""
    count = torch.bincount(labels, minlength=classes).float().clamp_min(1.0)
    weight = count.sum() / count
    return weight / weight.mean()


def balanced_batches(
    rows: np.ndarray,
    index: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """각 action class가 epoch마다 비슷한 빈도로 등장하도록 query를 재표집한다."""
    labels = torch.tensor(index.class_id.iloc[rows].to_numpy()).long()
    generator = torch.Generator().manual_seed(seed)
    frequency = torch.bincount(labels, minlength=C.N_CLASSES).float().clamp_min(1.0)
    draw = torch.multinomial(
        (1.0 / frequency)[labels], len(rows), replacement=True, generator=generator
    ).numpy()
    shuffled = rows[draw]
    return [shuffled[start:start + batch_size]
            for start in range(0, len(shuffled), batch_size)]


def augment_site(
    tensors: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """동일 현장의 absence/support/query에 같은 RF 변화를 주어 보정 가능성을 학습한다."""
    device = tensors[0][0].device
    generator = torch.Generator(device=device).manual_seed(seed)
    gain = torch.exp(0.25 * torch.randn(
        C.N_LINKS, generator=generator, device=device
    ))
    curvature = 0.30 * torch.randn(
        C.N_LINKS, generator=generator, device=device
    )
    ripple = 0.15 * torch.randn(
        C.N_LINKS, generator=generator, device=device
    )
    frequency = torch.linspace(-1.0, 1.0, C.N_LIVE_SUBCARRIERS, device=device)
    phase_shift = (
        curvature[:, None] * (frequency.square() - frequency.square().mean())[None]
        + ripple[:, None] * torch.sin(math.pi * frequency)[None]
    )
    drop = None
    if float(torch.rand((), generator=generator, device=device)) < 0.30:
        drop = int(torch.randint(
            C.N_LINKS, (1,), generator=generator, device=device
        ).item())

    augmented = []
    for csi, mask in tensors:
        values = csi.clone()
        local_mask = mask.clone()
        values[..., 0] *= gain[None, None, :, None]
        values[..., 1] += phase_shift[None, None]
        if drop is not None:
            local_mask[:, :, drop] = False
        values *= local_mask[..., None, None].to(values.dtype)
        augmented.append((values, local_mask))
    return augmented


def site_rows(selected_rows: np.ndarray, sites: np.ndarray, site: str) -> np.ndarray:
    """선택된 pose protocol에서 특정 사이트의 cache 행을 반환한다."""
    return selected_rows[sites == site]


@torch.no_grad()
def evaluate_site(
    model: RawSupportConditionedModel,
    store: RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    site: str,
    device: str,
    seed: int,
) -> dict:
    """고정 2-shot absence와 8-shot prompt만 사용해 미관측 사이트를 평가한다."""
    model.eval()
    rows = site_rows(selected_rows, sites, site)
    support = select_support(rows, index, seed)
    absence = select_absence(site, index, seed + 1)
    support_set = set(support.tolist())
    query = np.asarray([row for row in rows if row not in support_set], dtype=np.int64)
    support_csi, support_mask = store.get(support, device)
    absence_csi, absence_mask = store.get(absence, device)
    support_labels = torch.tensor(index.class_id.iloc[support].to_numpy(), device=device)
    action_logits, risk_logits, gates, link_weights = [], [], [], []
    for start in range(0, len(query), 32):
        batch = query[start:start + 32]
        query_csi, query_mask = store.get(batch, device)
        output = model(
            query_csi, query_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        action_logits.append(output["action_logits"].cpu())
        risk_logits.append(output["risk_logits"].cpu())
        gates.append(output["adapter_gate"].cpu())
        link_weights.append(output["query_link_weight"].cpu())
    labels = torch.tensor(index.class_id.iloc[query].to_numpy()).long()
    risks = torch.tensor(index.risk_id.iloc[query].to_numpy()).long()
    metrics = classification_metrics(
        torch.cat(action_logits), torch.cat(risk_logits), labels, risks
    )
    metrics.update({
        "adapter_gate_mean": float(torch.cat(gates).mean()),
        "link_weight_mean": torch.cat(link_weights).mean((0, 1)).tolist(),
        "site": site,
        "support_trials": len(support),
        "absence_trials": len(absence),
    })
    return metrics


def train_model(
    store: RawStore,
    index: pd.DataFrame,
    selected_rows: np.ndarray,
    sites: np.ndarray,
    train_sites: list[str],
    validation_site: str | None,
    epochs: int,
    device: str,
    seed: int,
) -> tuple[RawSupportConditionedModel, list[dict], dict | None]:
    """사이트별 support/query episode로 raw encoder와 보정 head를 함께 학습한다."""
    train_rows = selected_rows[np.isin(sites, train_sites)]
    train_labels = torch.tensor(index.class_id.iloc[train_rows].to_numpy()).long()
    train_risks = torch.tensor(index.risk_id.iloc[train_rows].to_numpy()).long()
    model = RawSupportConditionedModel(
        hidden=64, token_dim=96, width=192,
        domains=len(train_sites), dropout=0.10,
        prompt_classes=ACTIVE_PROMPT_CLASSES,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=2e-3
    )
    # Query 자체를 action-balanced로 뽑으므로 CE에 역빈도를 또 곱하지 않는다.
    action_weight = torch.ones(C.N_CLASSES, device=device)
    risk_weight = class_weights(train_risks, C.N_RISK).to(device)
    risk_weight /= risk_weight.mean()
    history: list[dict] = []
    best_score = -float("inf")
    best_state = None
    stale = 0
    patience = 8

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        rng = np.random.default_rng(seed + epoch)
        for site_value in rng.permutation(train_sites):
            site = str(site_value)
            domain = train_sites.index(site)
            rows = site_rows(selected_rows, sites, site)
            support = select_support(
                rows, index, seed * 1000 + epoch * 31 + domain
            )
            absence = select_absence(
                site, index, seed * 2000 + epoch * 47 + domain
            )
            support_set = set(support.tolist())
            query = np.asarray(
                [row for row in rows if row not in support_set], dtype=np.int64
            )
            support_csi, support_mask = store.get(support, device)
            absence_csi, absence_mask = store.get(absence, device)
            support_labels = torch.tensor(
                index.class_id.iloc[support].to_numpy(), device=device
            )

            for batch_number, batch in enumerate(balanced_batches(
                query, index, batch_size=24,
                seed=seed * 10_000 + epoch * 101 + domain,
            )):
                query_csi, query_mask = store.get(batch, device)
                if rng.random() < 0.80:
                    (absence_input, absence_input_mask), (
                        support_input, support_input_mask
                    ), (query_input, query_input_mask) = augment_site(
                        [(absence_csi, absence_mask),
                         (support_csi, support_mask),
                         (query_csi, query_mask)],
                        seed=seed * 1_000_000 + epoch * 10_000
                        + domain * 100 + batch_number,
                    )
                else:
                    absence_input, absence_input_mask = absence_csi, absence_mask
                    support_input, support_input_mask = support_csi, support_mask
                    query_input, query_input_mask = query_csi, query_mask

                labels = torch.tensor(
                    index.class_id.iloc[batch].to_numpy(), device=device
                )
                risks = torch.tensor(
                    index.risk_id.iloc[batch].to_numpy(), device=device
                )
                output = model(
                    query_input, query_input_mask,
                    support_input, support_input_mask, support_labels,
                    absence_input, absence_input_mask,
                )
                action_loss = F.cross_entropy(
                    output["action_logits"], labels, weight=action_weight
                )
                risk_loss = F.cross_entropy(
                    output["risk_logits"], risks, weight=risk_weight
                )
                domain_target = torch.full(
                    (len(batch),), domain, dtype=torch.long, device=device
                )
                domain_loss = F.cross_entropy(
                    output["domain_logits"], domain_target
                )
                if bool(model.calibrator.reference_initialized):
                    prototype_loss = F.smooth_l1_loss(
                        output["support_prompt_means"],
                        model.calibrator.reference_prompt.detach(),
                        beta=0.5,
                    )
                else:
                    prototype_loss = output["support_prompt_means"].sum() * 0.0
                link_entropy = -(
                    output["query_link_weight"].clamp_min(1e-7).log()
                    * output["query_link_weight"]
                ).sum(-1).mean()
                loss = (
                    action_loss + 0.65 * risk_loss
                    + 0.12 * domain_loss + 0.15 * prototype_loss
                    - 0.01 * link_entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                if batch_number == 0:
                    model.calibrator.update_reference(
                        output["support_prompt_means"], momentum=0.98
                    )
                losses.append(float(loss.detach()))

        record: dict = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
        }
        if validation_site is not None:
            validation = evaluate_site(
                model, store, index, selected_rows, sites,
                validation_site, device, seed=17017,
            )
            safe_specificity = 1.0 - (
                validation["safe_to_danger"]
                / max(validation["safe_total"], 1)
            )
            danger_recall = validation["danger_recall"]
            danger_balance = (
                2.0 * danger_recall * safe_specificity
                / max(danger_recall + safe_specificity, 1e-8)
            )
            score = (
                validation["action_macro_f1"]
                + 0.75 * validation["risk_macro_f1"]
                + danger_balance
                + 0.25 * validation["danger_action_accuracy"]
            )
            record["validation"] = validation
            record["safe_specificity"] = safe_specificity
            record["danger_balance"] = danger_balance
            record["selection_score"] = score
            if score > best_score + 1e-5:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if validation_site is not None and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best = None
    if validation_site is not None:
        best = max(history, key=lambda item: item["selection_score"])
    return model, history, best


def main() -> None:
    """source 선택·검증 뒤 epoch를 잠그고 전체 source deployment 모델을 만든다."""
    global ACTIVE_PROMPT_CLASSES, PROMPT_SHOTS, RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "dynamic"), default="dynamic")
    parser.add_argument("--run-dir", type=Path)
    options = parser.parse_args()
    if options.mode == "static":
        ACTIVE_PROMPT_CLASSES = PROMPT_CLASSES
        PROMPT_SHOTS = {0: 2, 1: 2, 2: 2, 3: 2}
        default_run = WORK / "runs/cal8_raw_source_anchor"
    else:
        ACTIVE_PROMPT_CLASSES = MOTION_PROMPT_CLASSES
        PROMPT_SHOTS = {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 7: 1, 8: 1}
        default_run = WORK / "runs/cal8_raw_source_dynamic_anchor_v2"
    RUN = options.run_dir or default_run
    torch.manual_seed(83)
    np.random.seed(83)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(WORK / "cache/cache_index.csv")
    feature_cache = torch.load(
        WORK / "runs/kp5_mpr_selector_seed17/train_features.pt",
        map_location="cpu", weights_only=False,
    )
    selected_rows = feature_cache["rows"].numpy().astype(np.int64)
    selected_index = index.iloc[selected_rows]
    sites = (
        selected_index.subject + "_" + selected_index.environment
    ).to_numpy()
    all_sites = sorted(set(sites))
    validation_site = "lmh_E01"
    selection_sites = [site for site in all_sites if site != validation_site]

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
    store = RawStore(index, np.concatenate((selected_rows, absence_rows)))
    RUN.mkdir(parents=True, exist_ok=True)

    selected_model, history, best = train_model(
        store, index, selected_rows, sites,
        selection_sites, validation_site,
        epochs=36, device=device, seed=83,
    )
    if best is None:
        raise RuntimeError("selection did not produce a checkpoint")
    torch.save({
        "model": selected_model.state_dict(),
        "model_config": selected_model.model_config(),
        "best": best,
        "protocol": "raw_ajh_mhw_train__raw_lmh_E01_validation",
        "target_subject_used": False,
    }, RUN / "selection_model.pt")

    locked_epochs = int(best["epoch"])
    deployment_model, deployment_history, _ = train_model(
        store, index, selected_rows, sites,
        all_sites, None, epochs=locked_epochs,
        device=device, seed=183,
    )
    source_metrics = {
        site: evaluate_site(
            deployment_model, store, index, selected_rows, sites,
            site, device, seed=17017,
        ) for site in all_sites
    }
    checkpoint = {
        "model": deployment_model.state_dict(),
        "model_config": deployment_model.model_config(),
        "locked_epochs": locked_epochs,
        "selection_protocol": "raw_ajh_mhw_train__raw_lmh_E01_validation",
        "source_sites": all_sites,
        "support_contract": {
            "absence_trials": 2,
            "prompt_classes": list(ACTIVE_PROMPT_CLASSES),
            "shots_by_class": PROMPT_SHOTS,
            "target_pose_gt_used": False,
        },
        "preprocessing": "absence_log_amplitude_zscore_and_wrapped_phase_residual",
        "target_subject_used": False,
    }
    torch.save(checkpoint, RUN / "deployment_model.pt")
    result = {
        "run": "CAL8-KP10-RAW-META-CALIBRATION",
        "device": device,
        "selection_best": best,
        "locked_epochs": locked_epochs,
        "source_metrics": source_metrics,
        "history": history,
        "deployment_history": deployment_history,
        "target_subject_used": False,
    }
    (RUN / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "locked_epochs": locked_epochs,
        "selection_best": best,
        "source_metrics": source_metrics,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
