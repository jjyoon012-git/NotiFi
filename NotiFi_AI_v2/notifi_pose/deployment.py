"""CAL20+CAL17+CAL23 source-only bundle의 현장 calibration과 CSI 추론."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import contract as C
from .cal13 import temporal_motion_signature
from .cal17 import (
    ANCHOR_CLASSES,
    anchor_geometry_error,
    cal17_action,
    cal17_risk,
)
from .cal27 import cal27_action
from .danger_support import (
    apply_danger_support,
    class_prototypes as danger_class_prototypes,
    support_evidence,
)
from .skeleton import sequence_bone_projection
from .model_factory import build_calibration_model
from .pose_simulation import best_motion_shift, shift_pose
from .dataio.csi_parser import load_csi_trial
from notifi_ai_v2.support_alignment import (
    action_to_risk_log_probability,
    aligned_logits,
    apply_affine_map,
    identity_ridge_map,
)


MIN_LINK_COVERAGE = 0.50
MIN_USABLE_LINKS = 2


def load_csi_csv(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """raw CSI CSV를 학습과 동일한 고정 30Hz 모델 tensor로 변환한다."""
    grid = np.arange(C.CACHE_FRAMES, dtype=np.float64) / C.TARGET_FPS
    trial = load_csi_trial(Path(path), grid_times=grid, trim_guard=True)
    return (
        torch.from_numpy(trial.iq).float(),
        torch.from_numpy(trial.link_mask).bool(),
        trial.meta,
    )


def load_csi_csv_batch(
    paths: list[str | Path],
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """여러 raw trial을 calibration 또는 query batch로 쌓는다."""
    if not paths:
        raise ValueError("at least one CSI CSV path is required")
    loaded = [load_csi_csv(path) for path in paths]
    return (
        torch.stack([item[0] for item in loaded]),
        torch.stack([item[1] for item in loaded]),
        [item[2] for item in loaded],
    )


@dataclass
class TargetCalibration:
    """새 현장에서 수집한 CSI support와 그 latent anchor를 보관한다."""

    support_csi: torch.Tensor
    support_mask: torch.Tensor
    support_labels: torch.Tensor
    absence_csi: torch.Tensor
    absence_mask: torch.Tensor
    anchors: torch.Tensor
    geometry_error: torch.Tensor
    domain_pass: torch.Tensor
    secondary_anchors: torch.Tensor | None = None
    secondary_geometry_error: torch.Tensor | None = None
    secondary_domain_pass: torch.Tensor | None = None
    danger_prototypes: torch.Tensor | None = None
    secondary_danger_prototypes: torch.Tensor | None = None
    motion_mapping: torch.Tensor | None = None
    warning_prototypes: torch.Tensor | None = None
    secondary_warning_prototypes: torch.Tensor | None = None


class CAL20Deployment:
    """봉인 target 정보 없이 export된 source bundle을 실행한다."""

    def __init__(self, bundle: dict, device: str | None = None) -> None:
        if bundle.get("sealed_yja_used") is not False:
            raise ValueError("deployment bundle must explicitly exclude sealed yja")
        if bundle.get("target_subject_used") is not False:
            raise ValueError("deployment bundle must explicitly exclude target subjects")
        if bundle.get("query_labels_or_pose_gt_used") is not False:
            raise ValueError("deployment bundle must explicitly exclude query labels and pose GT")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_calibration_model(bundle["model_config"]).to(self.device)
        self.model.load_state_dict(bundle["model"])
        self.model.eval()
        self.support_contract = dict(bundle["support_contract"])
        self.action_config = dict(bundle["action_config"])
        self.risk_config = dict(bundle["risk_config"])
        self.action_ensemble_config = list(
            bundle.get("action_ensemble_config", [])
        )
        self.risk_profiles = dict(bundle.get("risk_profiles", {
            "conservative": {"danger_bias": 0.0},
            "safety": {"danger_bias": 0.0},
        }))
        self.pose_config = dict(bundle.get("pose_config", {}))
        self.danger_support_contract = dict(
            bundle.get("danger_support_contract", {})
        )
        self.danger_support_config = dict(
            bundle.get("danger_support_config", {})
        )
        self.support_ridge_config = dict(
            bundle.get("support_ridge_config", {})
        )
        self.motion_ridge_config = dict(
            bundle.get("motion_ridge_config", {})
        )
        self.warning_support_contract = dict(
            bundle.get("warning_support_contract", {})
        )
        self.risk_from_action = bool(bundle.get("risk_from_action", False))
        self.geometry_threshold = float(
            bundle.get("calibration_geometry_threshold", float("inf"))
        )
        self.source_library = [{
            "classes": item["classes"].to(self.device).float(),
            "anchors": item["anchors"].to(self.device).float(),
        } for item in bundle["source_library"]]
        deep_action = bundle.get("deep_action")
        self.secondary_model = None
        self.secondary_source_library = []
        self.deep_action_config = []
        self.secondary_geometry_threshold = self.geometry_threshold
        if deep_action is not None:
            self.secondary_model = build_calibration_model(
                deep_action["model_config"]
            ).to(self.device)
            self.secondary_model.load_state_dict(deep_action["model"])
            self.secondary_model.eval()
            self.secondary_source_library = [{
                "classes": item["classes"].to(self.device).float(),
                "anchors": item["anchors"].to(self.device).float(),
            } for item in deep_action["source_library"]]
            self.deep_action_config = list(deep_action["configs"])
            self.secondary_geometry_threshold = float(
                deep_action.get(
                    "calibration_geometry_threshold", self.geometry_threshold
                )
            )
        pose = bundle.get("pose_library")
        self.pose_library = None if pose is None else {
            key: (
                value.float() if isinstance(value, torch.Tensor)
                and value.dtype.is_floating_point else value
            )
            for key, value in pose.items()
        }

    @classmethod
    def load(
        cls, path: str, device: str | None = None,
    ) -> "CAL20Deployment":
        """torch bundle을 CPU에서 안전하게 읽은 뒤 지정 device에 모델만 올린다."""
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        return cls(bundle, device=device)

    def _validate_support(
        self,
        support_csi: torch.Tensor,
        support_labels: torch.Tensor,
        support_mask: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_mask: torch.Tensor,
        danger_support_csi: torch.Tensor | None = None,
        danger_support_mask: torch.Tensor | None = None,
        danger_support_labels: torch.Tensor | None = None,
        warning_support_csi: torch.Tensor | None = None,
        warning_support_mask: torch.Tensor | None = None,
        warning_support_labels: torch.Tensor | None = None,
    ) -> None:
        """배포 계약의 class별 support 수와 absence 수를 강제한다."""
        self._validate_trial_batch("support", support_csi, support_mask)
        self._validate_trial_batch("absence", absence_csi, absence_mask)
        if support_labels.ndim != 1:
            raise ValueError("support labels must be a one-dimensional tensor")
        if len(support_labels) != len(support_csi):
            raise ValueError("support CSI, mask, and labels must have equal batches")
        expected = tuple(int(value) for value in self.support_contract[
            "prompt_classes"
        ])
        shots = int(self.support_contract["shots_per_prompt"])
        labels = support_labels.detach().cpu()
        for class_id in expected:
            if int((labels == class_id).sum()) != shots:
                raise ValueError(
                    f"class {class_id} requires exactly {shots} support trials"
                )
        if len(labels) != len(expected) * shots:
            raise ValueError("support contains undeclared calibration classes")
        required_absence = int(self.support_contract["absence_trials"])
        if len(absence_csi) != required_absence:
            raise ValueError(
                f"calibration requires exactly {required_absence} absence trials"
            )
        for name, mask in (
            ("support", support_mask), ("absence", absence_mask),
        ):
            coverage = mask.detach().float().mean(1)
            usable = (coverage >= MIN_LINK_COVERAGE).sum(-1)
            if bool((usable < MIN_USABLE_LINKS).any()):
                raise ValueError(
                    f"{name} trial requires at least {MIN_USABLE_LINKS} usable links"
                )
        if self.danger_support_contract:
            if any(value is None for value in (
                danger_support_csi, danger_support_mask, danger_support_labels,
            )):
                raise ValueError("deployment requires danger calibration support")
            self._validate_trial_batch(
                "danger support", danger_support_csi, danger_support_mask
            )
            if danger_support_labels.ndim != 1:
                raise ValueError("danger support labels must be one-dimensional")
            if len(danger_support_labels) != len(danger_support_csi):
                raise ValueError("danger support CSI, mask, and labels must match")
            danger_labels = danger_support_labels.detach().cpu()
            danger_classes = tuple(int(value) for value in self.danger_support_contract[
                "classes"
            ])
            danger_shots = int(self.danger_support_contract["shots_per_class"])
            for class_id in danger_classes:
                if int((danger_labels == class_id).sum()) != danger_shots:
                    raise ValueError(
                        f"danger class {class_id} requires exactly "
                        f"{danger_shots} support trials"
                    )
            if len(danger_labels) != len(danger_classes) * danger_shots:
                raise ValueError("danger support contains undeclared classes")
            danger_coverage = danger_support_mask.detach().float().mean(1)
            danger_usable = (danger_coverage >= MIN_LINK_COVERAGE).sum(-1)
            if bool((danger_usable < MIN_USABLE_LINKS).any()):
                raise ValueError(
                    "danger support trial requires at least "
                    f"{MIN_USABLE_LINKS} usable links"
                )
        elif any(value is not None for value in (
            danger_support_csi, danger_support_mask, danger_support_labels,
        )):
            raise ValueError("bundle does not declare danger calibration support")
        if self.warning_support_contract:
            if any(value is None for value in (
                warning_support_csi, warning_support_mask, warning_support_labels,
            )):
                raise ValueError("deployment requires warning calibration support")
            self._validate_trial_batch(
                "warning support", warning_support_csi, warning_support_mask
            )
            if warning_support_labels.ndim != 1:
                raise ValueError("warning support labels must be one-dimensional")
            warning_labels = warning_support_labels.detach().cpu()
            warning_classes = tuple(
                int(value) for value in self.warning_support_contract["classes"]
            )
            warning_shots = int(self.warning_support_contract["shots_per_class"])
            for class_id in warning_classes:
                if int((warning_labels == class_id).sum()) != warning_shots:
                    raise ValueError(
                        f"warning class {class_id} requires exactly "
                        f"{warning_shots} support trials"
                    )
            if len(warning_labels) != len(warning_classes) * warning_shots:
                raise ValueError("warning support contains undeclared classes")
            coverage = warning_support_mask.detach().float().mean(1)
            usable = (coverage >= MIN_LINK_COVERAGE).sum(-1)
            if bool((usable < MIN_USABLE_LINKS).any()):
                raise ValueError(
                    "warning support trial requires at least "
                    f"{MIN_USABLE_LINKS} usable links"
                )
        elif any(value is not None for value in (
            warning_support_csi, warning_support_mask, warning_support_labels,
        )):
            raise ValueError("bundle does not declare warning calibration support")

    @staticmethod
    def _validate_trial_batch(
        name: str, csi: torch.Tensor, mask: torch.Tensor,
    ) -> None:
        """CSI와 mask의 batch/time/link 축 및 유효 값 계약을 검사한다."""
        if csi.ndim != 5 or csi.shape[-1] != 2:
            raise ValueError(f"{name} CSI must have shape [B,T,L,S,2]")
        if mask.ndim != 3:
            raise ValueError(f"{name} mask must have shape [B,T,L]")
        if tuple(csi.shape[:3]) != tuple(mask.shape):
            raise ValueError(f"{name} CSI and mask axes do not match")
        if csi.shape[2] != C.N_LINKS:
            raise ValueError(f"{name} requires exactly {C.N_LINKS} CSI links")
        if len(csi) == 0:
            raise ValueError(f"{name} batch cannot be empty")
        valid = mask.to(dtype=torch.bool)[..., None, None].expand_as(csi)
        if bool(valid.any()) and not bool(torch.isfinite(csi[valid]).all()):
            raise ValueError(f"{name} contains non-finite valid CSI values")

    @staticmethod
    def signal_quality(mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """trial별 link coverage와 최소 배포 품질 통과 여부를 계산한다."""
        coverage = mask.float().mean(1)
        usable = (coverage >= MIN_LINK_COVERAGE).sum(-1)
        passed = usable >= MIN_USABLE_LINKS
        return {
            "link_coverage": coverage,
            "usable_links": usable,
            "quality_pass": passed,
            "abstain": ~passed,
        }

    @torch.inference_mode()
    def calibrate(
        self,
        support_csi: torch.Tensor,
        support_mask: torch.Tensor,
        support_labels: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_mask: torch.Tensor,
        danger_support_csi: torch.Tensor | None = None,
        danger_support_mask: torch.Tensor | None = None,
        danger_support_labels: torch.Tensor | None = None,
        warning_support_csi: torch.Tensor | None = None,
        warning_support_mask: torch.Tensor | None = None,
        warning_support_labels: torch.Tensor | None = None,
    ) -> TargetCalibration:
        """기본동작·빈 공간·통제된 낙상 CSI를 target anchor로 변환한다."""
        self._validate_support(
            support_csi, support_labels, support_mask, absence_csi, absence_mask,
            danger_support_csi, danger_support_mask, danger_support_labels,
            warning_support_csi, warning_support_mask, warning_support_labels,
        )
        support_csi = support_csi.to(self.device).float()
        support_mask = support_mask.to(self.device).bool()
        support_labels = support_labels.to(self.device).long()
        absence_csi = absence_csi.to(self.device).float()
        absence_mask = absence_mask.to(self.device).bool()
        if danger_support_csi is not None:
            danger_support_csi = danger_support_csi.to(self.device).float()
            danger_support_mask = danger_support_mask.to(self.device).bool()
            danger_support_labels = danger_support_labels.to(self.device).long()
        if warning_support_csi is not None:
            warning_support_csi = warning_support_csi.to(self.device).float()
            warning_support_mask = warning_support_mask.to(self.device).bool()
            warning_support_labels = warning_support_labels.to(self.device).long()
        support_output = self.model(
            support_csi, support_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        absence_output = self.model(
            absence_csi, absence_mask,
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask,
        )
        support_embedding = support_output["embedding"]
        absence_embedding = absence_output["embedding"]
        anchors = torch.stack([
            F.normalize(absence_embedding.mean(0), dim=0)
            if class_id == 6 else F.normalize(
                support_embedding[support_labels == class_id].mean(0), dim=0
            )
            for class_id in ANCHOR_CLASSES
        ])
        geometry_error = torch.stack([
            anchor_geometry_error(source["anchors"], anchors)
            for source in self.source_library
        ]).min()
        domain_pass = geometry_error <= self.geometry_threshold
        secondary_anchors = None
        secondary_geometry_error = None
        secondary_domain_pass = None
        danger_prototypes = None
        secondary_danger_prototypes = None
        motion_mapping = None
        warning_prototypes = None
        secondary_warning_prototypes = None
        if danger_support_csi is not None:
            danger_output = self.model(
                danger_support_csi, danger_support_mask,
                support_csi, support_mask, support_labels,
                absence_csi, absence_mask,
            )
            danger_prototypes = danger_class_prototypes(
                danger_output["embedding"], danger_support_labels,
                tuple(int(value) for value in self.danger_support_contract["classes"]),
            )
            if self.motion_ridge_config and self.pose_library is not None:
                motion_classes = tuple(
                    int(value) for value in self.motion_ridge_config["classes"]
                )
                basic_signature = temporal_motion_signature(
                    support_output["pose_motion"].detach().cpu(),
                    support_output["query_frame_mask"].detach().cpu().bool(),
                )
                danger_signature = temporal_motion_signature(
                    danger_output["pose_motion"].detach().cpu(),
                    danger_output["query_frame_mask"].detach().cpu().bool(),
                )
                target_signature = torch.cat((basic_signature, danger_signature))
                target_labels = torch.cat((
                    support_labels.detach().cpu(), danger_support_labels.detach().cpu()
                ))
                target_prototypes = torch.stack([
                    target_signature[target_labels == class_id].mean(0)
                    for class_id in motion_classes
                ])
                normalized_target = (
                    target_prototypes - self.pose_library["signature_center"]
                ) / self.pose_library["signature_scale"]
                library_labels = self.pose_library["labels"].long()
                source_prototypes = torch.stack([
                    self.pose_library["normalized_signatures"][
                        library_labels == class_id
                    ].mean(0)
                    for class_id in motion_classes
                ])
                motion_mapping = identity_ridge_map(
                    normalized_target,
                    source_prototypes,
                    float(self.motion_ridge_config["regularization"]),
                    normalize_inputs=False,
                )
        if warning_support_csi is not None:
            warning_output = self.model(
                warning_support_csi, warning_support_mask,
                support_csi, support_mask, support_labels,
                absence_csi, absence_mask,
            )
            warning_prototypes = torch.stack([
                F.normalize(
                    warning_output["embedding"][warning_support_labels == class_id].mean(0),
                    dim=0,
                )
                for class_id in self.warning_support_contract["classes"]
            ])
        if self.secondary_model is not None:
            secondary_support = self.secondary_model(
                support_csi, support_mask,
                support_csi, support_mask, support_labels,
                absence_csi, absence_mask,
            )["embedding"]
            secondary_absence = self.secondary_model(
                absence_csi, absence_mask,
                support_csi, support_mask, support_labels,
                absence_csi, absence_mask,
            )["embedding"]
            secondary_anchors = torch.stack([
                F.normalize(secondary_absence.mean(0), dim=0)
                if class_id == 6 else F.normalize(
                    secondary_support[support_labels == class_id].mean(0), dim=0
                )
                for class_id in ANCHOR_CLASSES
            ])
            secondary_geometry_error = torch.stack([
                anchor_geometry_error(source["anchors"], secondary_anchors)
                for source in self.secondary_source_library
            ]).min()
            secondary_domain_pass = (
                secondary_geometry_error <= self.secondary_geometry_threshold
            )
            if danger_support_csi is not None:
                secondary_danger_output = self.secondary_model(
                    danger_support_csi, danger_support_mask,
                    support_csi, support_mask, support_labels,
                    absence_csi, absence_mask,
                )
                secondary_danger_prototypes = danger_class_prototypes(
                    secondary_danger_output["embedding"], danger_support_labels,
                    tuple(
                        int(value)
                        for value in self.danger_support_contract["classes"]
                    ),
                )
            if warning_support_csi is not None:
                secondary_warning_output = self.secondary_model(
                    warning_support_csi, warning_support_mask,
                    support_csi, support_mask, support_labels,
                    absence_csi, absence_mask,
                )
                secondary_warning_prototypes = torch.stack([
                    F.normalize(
                        secondary_warning_output["embedding"][
                            warning_support_labels == class_id
                        ].mean(0),
                        dim=0,
                    )
                    for class_id in self.warning_support_contract["classes"]
                ])
        return TargetCalibration(
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask, anchors,
            geometry_error, domain_pass,
            secondary_anchors, secondary_geometry_error,
            secondary_domain_pass,
            danger_prototypes, secondary_danger_prototypes,
            motion_mapping,
            warning_prototypes, secondary_warning_prototypes,
        )

    def _simulate_pose(
        self,
        motion: torch.Tensor,
        frame_mask: torch.Tensor,
        action: torch.Tensor,
        risk: torch.Tensor,
        motion_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[list[str]]]:
        """CSI motion과 action으로 source 궤적 top-k를 골라 3D 동작을 합성한다."""
        if self.pose_library is None:
            return {}
        library = self.pose_library
        motion = motion.detach().cpu().float()
        frame_mask = frame_mask.detach().cpu().bool()
        probability = action.detach().cpu().softmax(-1)
        signature = temporal_motion_signature(motion, frame_mask)
        normalized = (
            signature - library["signature_center"]
        ) / library["signature_scale"]
        if motion_mapping is not None and self.motion_ridge_config:
            mapped = apply_affine_map(
                normalized, motion_mapping, normalize_output=False
            )
            danger_probability = risk.detach().cpu().softmax(-1)[:, 2]
            gate_mode = self.motion_ridge_config.get("gate", "risk_sqrt")
            if gate_mode == "risk_sqrt":
                gate = danger_probability.sqrt()
            elif gate_mode == "risk_soft":
                gate = danger_probability
            elif gate_mode == "risk_hard":
                gate = (risk.detach().cpu().argmax(-1) == 2).float()
            elif gate_mode == "all":
                gate = torch.ones_like(danger_probability)
            else:
                raise ValueError(f"unknown motion ridge gate: {gate_mode}")
            mixture = float(self.motion_ridge_config.get("mixture", 0.0))
            normalized = normalized + mixture * gate[:, None] * (
                mapped - normalized
            )
        distance = torch.cdist(
            normalized, library["normalized_signatures"]
        ).square()
        labels = library["labels"].long()
        top_actions = probability.topk(3, dim=-1).indices
        allowed = (labels[None, :, None] == top_actions[:, None, :]).any(-1)
        action_penalty = -0.25 * torch.log(
            probability[:, labels].clamp_min(1e-8)
        )
        score = (distance + action_penalty).masked_fill(~allowed, torch.inf)
        neighbors = int(self.pose_config.get("neighbors", 5))
        top_score, top_index = score.topk(neighbors, largest=False, dim=-1)
        hypotheses = []
        for query_number in range(len(top_index)):
            current = []
            for candidate_number in top_index[query_number].tolist():
                shift = best_motion_shift(
                    motion[query_number],
                    library["descriptors"][candidate_number],
                    frame_mask[query_number]
                    & library["valid"][candidate_number].bool(),
                )
                current.append(shift_pose(
                    library["pose"][candidate_number], shift
                ))
            hypotheses.append(torch.stack(current))
        hypotheses = torch.stack(hypotheses)
        temperature = float(self.pose_config.get("temperature", 0.5))
        weight = torch.softmax(
            -(top_score - top_score[:, :1]) / max(temperature, 1e-4), dim=-1
        )
        prediction = (
            hypotheses * weight[:, :, None, None, None]
        ).sum(1)
        bone_blend = float(self.pose_config.get("bone_blend", 0.5))
        if bone_blend > 0.0:
            projected = sequence_bone_projection(
                prediction, frame_mask, symmetric=True
            )
            prediction = prediction + bone_blend * (projected - prediction)
        return {
            "pose_rel": prediction,
            "pose_valid": frame_mask,
            "retrieval_scores": top_score,
            "retrieval_labels": labels[top_index],
            "retrieval_trial_ids": [
                [library["trial_ids"][int(index)] for index in row]
                for row in top_index
            ] if "trial_ids" in library else [],
        }

    @torch.inference_mode()
    def predict(
        self,
        query_csi: torch.Tensor,
        query_mask: torch.Tensor,
        calibration: TargetCalibration,
        simulate_pose: bool = True,
        risk_profile: str = "safety",
    ) -> dict[str, torch.Tensor | list[list[str]]]:
        """query CSI만으로 17-action, 3-risk와 선택적 3D pose를 반환한다."""
        self._validate_trial_batch("query", query_csi, query_mask)
        query_mask = query_mask.to(self.device).bool()
        quality = self.signal_quality(query_mask)
        quality["calibration_geometry_error"] = (
            calibration.geometry_error[None].expand(len(query_mask))
        )
        quality["calibration_domain_pass"] = (
            calibration.domain_pass[None].expand(len(query_mask))
        )
        quality["calibration_domain_warning"] = ~quality[
            "calibration_domain_pass"
        ]
        if calibration.secondary_geometry_error is not None:
            quality["secondary_calibration_geometry_error"] = (
                calibration.secondary_geometry_error[None].expand(len(query_mask))
            )
            secondary_pass = calibration.secondary_domain_pass[None].expand(
                len(query_mask)
            )
            quality["secondary_calibration_domain_pass"] = secondary_pass
            quality["calibration_domain_pass"] = (
                quality["calibration_domain_pass"] & secondary_pass
            )
            quality["calibration_domain_warning"] = ~quality[
                "calibration_domain_pass"
            ]
        quality["abstain"] = (
            quality["abstain"] | ~quality["calibration_domain_pass"]
        )
        output = self.model(
            query_csi.to(self.device).float(),
            query_mask,
            calibration.support_csi, calibration.support_mask,
            calibration.support_labels,
            calibration.absence_csi, calibration.absence_mask,
        )
        target = {
            "action": output["action_logits"],
            "risk": output["risk_logits"],
            "direct_risk": output["direct_risk_logits"],
            "embedding": output["embedding"],
            "anchors": calibration.anchors,
        }
        if self.secondary_model is not None:
            if calibration.secondary_anchors is None:
                raise RuntimeError("CAL40 requires secondary calibration anchors")
            secondary_output = self.secondary_model(
                query_csi.to(self.device).float(), query_mask,
                calibration.support_csi, calibration.support_mask,
                calibration.support_labels,
                calibration.absence_csi, calibration.absence_mask,
            )
            secondary_target = {
                "action": secondary_output["action_logits"],
                "risk": secondary_output["risk_logits"],
                "direct_risk": secondary_output["direct_risk_logits"],
                "embedding": secondary_output["embedding"],
                "anchors": calibration.secondary_anchors,
            }
            action_members = []
            for config in self.deep_action_config:
                primary_action = cal17_action(
                    target, self.source_library,
                    config["primary_linear_config"],
                )
                secondary_action = cal17_action(
                    secondary_target, self.secondary_source_library,
                    config["secondary_linear_config"],
                )
                pair = torch.logaddexp(
                    primary_action.log_softmax(-1),
                    secondary_action.log_softmax(-1),
                ) - torch.log(primary_action.new_tensor(2.0))
                action_members.append(pair)
            action = torch.logsumexp(
                torch.stack(action_members), dim=0
            ) - torch.log(action_members[0].new_tensor(len(action_members)))
        elif self.action_ensemble_config:
            action_members = []
            for config in self.action_ensemble_config:
                linear = cal17_action(
                    target, self.source_library, config["linear_config"]
                )
                kernel = cal27_action(
                    target, self.source_library, config["kernel_config"]
                )
                weight = float(config["kernel_weight"])
                action_members.append(
                    (1.0 - weight) * linear.log_softmax(-1)
                    + weight * kernel.log_softmax(-1)
                )
            action = torch.logsumexp(
                torch.stack([member.log_softmax(-1) for member in action_members]),
                dim=0,
            ) - torch.log(action_members[0].new_tensor(len(action_members)))
        else:
            action = cal17_action(
                target, self.source_library, self.action_config
            )
        if self.support_ridge_config:
            if calibration.danger_prototypes is None:
                raise RuntimeError("support ridge requires danger calibration prototypes")
            config = self.support_ridge_config
            primary_ridge = aligned_logits(
                target["embedding"],
                calibration.anchors,
                calibration.danger_prototypes,
                self.source_library,
                float(config["regularization"]),
                float(config["prototype_temperature"]),
                float(config["site_temperature"]),
                str(config.get("direction", "source_to_target")),
                target_warning=calibration.warning_prototypes,
            )
            if self.secondary_model is not None:
                secondary_ridge = aligned_logits(
                    secondary_target["embedding"],
                    calibration.secondary_anchors,
                    calibration.secondary_danger_prototypes,
                    self.secondary_source_library,
                    float(config["regularization"]),
                    float(config["prototype_temperature"]),
                    float(config["site_temperature"]),
                    str(config.get("direction", "source_to_target")),
                    target_warning=calibration.secondary_warning_prototypes,
                )
                ridge_action = torch.logaddexp(
                    primary_ridge.log_softmax(-1),
                    secondary_ridge.log_softmax(-1),
                ) - math.log(2.0)
            else:
                ridge_action = primary_ridge.log_softmax(-1)
            mixture = float(config["mixture"])
            action = torch.logaddexp(
                action.log_softmax(-1) + math.log1p(-mixture),
                ridge_action + math.log(mixture),
            )
        if risk_profile not in self.risk_profiles:
            raise ValueError(
                f"unknown risk profile {risk_profile!r}; "
                f"choose one of {sorted(self.risk_profiles)}"
            )
        if self.action_ensemble_config or self.secondary_model is not None:
            conservative_risk = target["risk"].clone()
        else:
            conservative_risk = cal17_risk(
                self.model, target, action, self.risk_config
            )
        danger_diagnostics = {}
        if self.danger_support_config:
            if calibration.danger_prototypes is None:
                raise RuntimeError("CAL44 requires danger calibration prototypes")
            temperature = float(self.danger_support_config["temperature"])
            evidence = [support_evidence(
                target["embedding"], calibration.anchors[:-1],
                calibration.danger_prototypes, temperature,
            )]
            if self.secondary_model is not None:
                if calibration.secondary_danger_prototypes is None:
                    raise RuntimeError("CAL44 requires secondary danger prototypes")
                evidence.append(support_evidence(
                    secondary_target["embedding"],
                    calibration.secondary_anchors[:-1],
                    calibration.secondary_danger_prototypes, temperature,
                ))
            action, conservative_risk, danger_diagnostics = apply_danger_support(
                action, conservative_risk, evidence, self.danger_support_config
            )
        motion_risk = conservative_risk
        if self.risk_from_action:
            conservative_risk = action_to_risk_log_probability(action)
        safety_risk = conservative_risk.clone()
        safety_risk[:, 2] += float(
            self.risk_profiles.get("safety", {}).get("danger_bias", 0.0)
        )
        risk = conservative_risk if risk_profile == "conservative" else safety_risk
        result = {
            "action_logits": action,
            "action_probability": action.softmax(-1),
            "action_id": action.argmax(-1),
            "risk_logits": risk,
            "risk_probability": risk.softmax(-1),
            "risk_id": risk.argmax(-1),
            "risk_profile": risk_profile,
            "conservative_risk_probability": conservative_risk.softmax(-1),
            "safety_risk_probability": safety_risk.softmax(-1),
            **danger_diagnostics,
            **quality,
        }
        if simulate_pose:
            result.update(self._simulate_pose(
                output["pose_motion"],
                output["query_frame_mask"],
                action,
                motion_risk,
                calibration.motion_mapping,
            ))
        return result


# 과거 bundle API와의 호환성은 유지하되 현재 공개 이름은 채택 모델에 맞춘다.
CAL44Deployment = CAL20Deployment


__all__ = (
    "CAL20Deployment", "CAL44Deployment", "TargetCalibration",
    "load_csi_csv", "load_csi_csv_batch",
)
