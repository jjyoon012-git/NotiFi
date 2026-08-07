"""CAL20+CAL17+CAL23 source-only bundle의 현장 calibration과 CSI 추론."""

from __future__ import annotations

from dataclasses import dataclass
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
from .hybrid_v10 import sequence_bone_projection
from .model_factory import build_calibration_model
from .pose_simulation import best_motion_shift, shift_pose
from .dataio.csi_parser import load_csi_trial


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
        self.pose_config = dict(bundle.get("pose_config", {}))
        self.geometry_threshold = float(
            bundle.get("calibration_geometry_threshold", float("inf"))
        )
        self.source_library = [{
            "classes": item["classes"].to(self.device).float(),
            "anchors": item["anchors"].to(self.device).float(),
        } for item in bundle["source_library"]]
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
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        return cls(bundle, device=device)

    def _validate_support(
        self,
        support_labels: torch.Tensor,
        support_mask: torch.Tensor,
        absence_csi: torch.Tensor,
        absence_mask: torch.Tensor,
    ) -> None:
        """배포 계약의 class별 support 수와 absence 수를 강제한다."""
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
    ) -> TargetCalibration:
        """새 사용자의 기본동작과 빈 공간 CSI를 target latent anchor로 변환한다."""
        self._validate_support(
            support_labels, support_mask, absence_csi, absence_mask
        )
        support_csi = support_csi.to(self.device).float()
        support_mask = support_mask.to(self.device).bool()
        support_labels = support_labels.to(self.device).long()
        absence_csi = absence_csi.to(self.device).float()
        absence_mask = absence_mask.to(self.device).bool()
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
        return TargetCalibration(
            support_csi, support_mask, support_labels,
            absence_csi, absence_mask, anchors,
            geometry_error, domain_pass,
        )

    def _simulate_pose(
        self,
        motion: torch.Tensor,
        frame_mask: torch.Tensor,
        action: torch.Tensor,
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
    ) -> dict[str, torch.Tensor | list[list[str]]]:
        """query CSI만으로 17-action, 3-risk와 선택적 3D pose를 반환한다."""
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
        output = self.model(
            query_csi.to(self.device).float(),
            query_mask,
            calibration.support_csi, calibration.support_mask,
            calibration.support_labels,
            calibration.absence_csi, calibration.absence_mask,
        )
        target = {
            "action": output["action_logits"],
            "direct_risk": output["direct_risk_logits"],
            "embedding": output["embedding"],
            "anchors": calibration.anchors,
        }
        action = cal17_action(
            target, self.source_library, self.action_config
        )
        risk = cal17_risk(self.model, target, action, self.risk_config)
        result = {
            "action_logits": action,
            "action_probability": action.softmax(-1),
            "action_id": action.argmax(-1),
            "risk_logits": risk,
            "risk_probability": risk.softmax(-1),
            "risk_id": risk.argmax(-1),
            **quality,
        }
        if simulate_pose:
            result.update(self._simulate_pose(
                output["pose_motion"], output["query_frame_mask"], action
            ))
        return result
