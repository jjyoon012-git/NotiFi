"""Unified classification, risk, retrieval-pose, and calibration runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .retrieval import (
    CandidateMotionReranker,
    MotionProfileHead,
    PartMotionProfileHead,
    ProfileCandidateRanker,
    TemporalMotionSelector,
    candidate_part_speed_profiles,
    candidate_speed_profiles,
    canonicalize,
    model_inputs,
    monotonic_energy_warp,
    part_profile_distance,
    predict_locked,
    predict_part_profile,
    predict_profile,
    predict_selector,
    profile_distance,
    render_ranked_action,
    retrieval_features,
    smooth_valid_delta,
    standardize,
)
from .calibration import CalibrationProfile
from .constants import ACTION_LABELS, MODEL_NAME, RISK_LABELS
from .preprocessing import load_csv, pad_or_trim
from .schemas import Prediction, SupportTrial


class NotiFiAIv1:
    """Frozen seen model with optional installation-specific calibration."""

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        device: str | None = None,
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.artifact_dir = Path(artifact_dir or project_root / "artifacts")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        cpu_suffix = "_cpu" if self.device == "cpu" else ""
        core_path = self.artifact_dir / f"notifi_ai_v1_core{cpu_suffix}.ts"
        feature_path = self.artifact_dir / f"notifi_ai_v1_features{cpu_suffix}.ts"
        retrieval_path = self.artifact_dir / "notifi_ai_v1_retrieval.pt"
        missing = [
            str(path)
            for path in (core_path, feature_path, retrieval_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "NotiFi_AI_v1 artifacts are missing: " + ", ".join(missing)
            )
        self.core = torch.jit.load(str(core_path), map_location=self.device).eval()
        self.feature_model = torch.jit.load(
            str(feature_path), map_location=self.device
        ).eval()
        bundle = torch.load(retrieval_path, map_location="cpu", weights_only=False)
        if bundle.get("model_name") != MODEL_NAME:
            raise RuntimeError("retrieval artifact belongs to another model")
        self.bundle_metadata = bundle["metadata"]
        self.adaptive_config = bundle["adaptive_config"]
        self.motion_strength = float(bundle["motion_strength"])
        self.pool_size = int(bundle.get("pool_size", 20))
        self.action_penalty = float(bundle.get("action_penalty", 0.05))

        self.selector_checkpoint = bundle["selector"]
        self.selector = TemporalMotionSelector(
            **self.selector_checkpoint["model_config"]
        ).to(self.device)
        self.selector.load_state_dict(self.selector_checkpoint["model"])

        reranker_checkpoint = bundle["candidate_reranker"]
        self.candidate_reranker = CandidateMotionReranker(
            **reranker_checkpoint["model_config"]
        ).to(self.device)
        self.candidate_reranker.load_state_dict(reranker_checkpoint["model"])

        scalar_checkpoint = bundle["scalar_profile"]
        self.scalar_profile = MotionProfileHead(
            **scalar_checkpoint["model_config"]
        ).to(self.device)
        self.scalar_profile.load_state_dict(scalar_checkpoint["model"])

        part_checkpoint = bundle["part_profile"]
        self.part_profile = PartMotionProfileHead(
            **part_checkpoint["model_config"]
        ).to(self.device)
        self.part_profile.load_state_dict(part_checkpoint["model"])

        classifier_checkpoint = bundle["retrieval_action_classifier"]
        self.retrieval_action_classifier = TemporalMotionSelector(
            **classifier_checkpoint["model_config"]
        ).to(self.device)
        self.retrieval_action_classifier.load_state_dict(
            classifier_checkpoint["model"]
        )

        profile_ranker_checkpoint = bundle["profile_ranker"]
        self.profile_ranker = ProfileCandidateRanker(
            **profile_ranker_checkpoint["model_config"]
        ).to(self.device)
        self.profile_ranker.load_state_dict(profile_ranker_checkpoint["model"])
        for model in (
            self.selector,
            self.candidate_reranker,
            self.scalar_profile,
            self.part_profile,
            self.retrieval_action_classifier,
            self.profile_ranker,
        ):
            model.eval()

    @torch.inference_mode()
    def _forward_components(
        self,
        csi: np.ndarray,
        link_mask: np.ndarray,
        calibration: CalibrationProfile | None,
    ) -> dict[str, torch.Tensor | dict]:
        if calibration is None:
            values, mask = pad_or_trim(csi, link_mask)
        else:
            values, mask = calibration.apply_csi(csi, link_mask)
        csi_tensor = torch.from_numpy(values)[None].to(self.device)
        mask_tensor = torch.from_numpy(mask)[None].to(self.device)
        coarse_pose, root, display_action, display_risk = self.core(
            csi_tensor, mask_tensor
        )
        (
            features,
            baseline_pose,
            base_action,
            base_risk,
            contact_logits,
            phase_logits,
            motion_activity,
        ) = self.feature_model(csi_tensor, mask_tensor, coarse_pose)
        frame_mask = mask_tensor.any(-1)
        feature_cache = {
            "features": features.detach().cpu().float(),
            "frame_mask": frame_mask.detach().cpu(),
            "baseline_pose": baseline_pose.detach().cpu().float(),
            "base_action_logits": base_action.detach().cpu().float(),
            "base_risk_logits": base_risk.detach().cpu().float(),
            "contact_logits": contact_logits.detach().cpu().float(),
            "phase_logits": phase_logits.detach().cpu().float(),
            "motion_activity": motion_activity.detach().cpu().float(),
        }
        selector_output = predict_selector(
            self.selector, feature_cache, 64, self.device
        )
        retrieval_output = self.retrieval_action_classifier(
            feature_cache["features"].to(self.device),
            feature_cache["frame_mask"].to(self.device),
        )
        retrieval_action = (
            1.50 * feature_cache["base_action_logits"]
            + 0.75 * retrieval_output["action_logits"].detach().cpu()
        )
        retrieval_risk = (
            feature_cache["base_risk_logits"]
            + selector_output["risk_logits"]
        )
        if calibration is not None:
            display_action = calibration.apply_display_action(display_action)
            display_risk = calibration.apply_display_risk(display_risk)
            retrieval_action = calibration.apply_retrieval_action(retrieval_action)
            retrieval_risk = calibration.apply_retrieval_risk(retrieval_risk)
        return {
            "csi": torch.from_numpy(values)[None],
            "link_mask": torch.from_numpy(mask)[None],
            "frame_mask": feature_cache["frame_mask"],
            "cache": feature_cache,
            "selector_output": selector_output,
            "display_action_logits": display_action.detach().cpu().float(),
            "display_risk_logits": display_risk.detach().cpu().float(),
            "retrieval_action_logits": retrieval_action.float(),
            "retrieval_risk_logits": retrieval_risk.float(),
            "root": root.detach().cpu().float(),
        }

    def _candidate_pool(
        self,
        baseline_bank: torch.Tensor,
        fused_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        train_bank = self.selector_checkpoint["train_bank"].float()
        train_class = self.selector_checkpoint["train_class"].long()
        probability = torch.softmax(fused_action, dim=-1).clamp_min(1e-6)
        all_indices, all_scores, all_action_log = [], [], []
        for item, query in enumerate(baseline_bank):
            distance = torch.linalg.vector_norm(
                train_bank - query[None], dim=-1
            ).mean((1, 2))
            scale = torch.quantile(distance, 0.50).clamp_min(1e-5)
            action_log = probability[item, train_class].log()
            score = distance / scale - self.action_penalty * action_log
            count = min(self.pool_size, len(score))
            indices = score.topk(count, largest=False).indices
            all_indices.append(indices)
            all_scores.append(score.index_select(0, indices))
            all_action_log.append(action_log.index_select(0, indices))
        return {
            "indices": torch.stack(all_indices),
            "retrieval_score": torch.stack(all_scores),
            "action_log_probability": torch.stack(all_action_log),
        }

    @torch.inference_mode()
    def _retrieve_pose(self, components: dict) -> torch.Tensor:
        cache = components["cache"]
        frame_mask = components["frame_mask"]
        baseline = cache["baseline_pose"].float()
        baseline_bank = torch.stack(
            [
                canonicalize(pose, valid, baseline.shape[1])
                for pose, valid in zip(baseline, frame_mask)
            ]
        )
        fused_action = components["retrieval_action_logits"]
        pool = self._candidate_pool(baseline_bank, fused_action)
        selector_output = components["selector_output"]
        risk_probability = torch.softmax(
            components["retrieval_risk_logits"], dim=-1
        )
        indices = torch.arange(len(baseline))
        inputs = tuple(
            value.to(self.device)
            for value in model_inputs(
                pool,
                selector_output,
                self.selector_checkpoint,
                risk_probability,
                indices,
            )
        )
        candidate_logits = self.candidate_reranker(*inputs).float().cpu()
        predicted_scalar = predict_profile(
            self.scalar_profile, cache, frame_mask, self.device
        )
        predicted_part = predict_part_profile(
            self.part_profile, cache, frame_mask, self.device
        )
        train_bank = self.selector_checkpoint["train_bank"].float()
        scalar_candidates = candidate_speed_profiles(
            train_bank, pool, frame_mask
        )
        part_candidates = candidate_part_speed_profiles(
            train_bank, pool, frame_mask
        )
        scalar_distance = standardize(
            profile_distance(predicted_scalar, scalar_candidates, frame_mask)
        )
        part_distance = part_profile_distance(
            predicted_part, part_candidates, frame_mask
        )
        data = {
            "checkpoint": self.selector_checkpoint,
            "baseline": baseline,
            "baseline_bank": baseline_bank,
            "train_bank": train_bank,
            "fused_action": fused_action,
            "risk_probability": risk_probability,
            "pool": pool,
            "logits": candidate_logits,
            "scalar_distance": scalar_distance,
            "part_distance": part_distance,
            "predicted_scalar_profile": predicted_scalar,
            "predicted_part_profile": predicted_part,
            "inference_valid": frame_mask,
            "selector_embedding": selector_output["embedding"],
        }
        current = predict_locked(data, self.adaptive_config)
        prior = render_ranked_action(
            self.profile_ranker,
            data,
            retrieval_features(data, 3, 1.0),
            self.device,
        )
        activity = 0.50 * predicted_scalar + 0.50 * predicted_part[..., 2:].mean(-1)
        prior = monotonic_energy_warp(prior, activity, frame_mask, 0.50, 0.30)
        residual = smooth_valid_delta(prior - current, frame_mask, 17)
        predicted = current + self.motion_strength * residual
        return predicted - predicted[:, :, :1]

    @torch.inference_mode()
    def predict(
        self,
        csi: np.ndarray,
        link_mask: np.ndarray,
        calibration: CalibrationProfile | None = None,
    ) -> Prediction:
        """Predict action, risk, root, and pelvis-relative SMPL-22 pose."""

        components = self._forward_components(csi, link_mask, calibration)
        pose = self._retrieve_pose(components)[0].numpy().astype(np.float32)
        action_probability = torch.softmax(
            components["display_action_logits"], dim=-1
        )[0]
        risk_probability = torch.softmax(
            components["display_risk_logits"], dim=-1
        )[0]
        action_id = int(action_probability.argmax())
        risk_id = int(risk_probability.argmax())
        mask = components["link_mask"][0].numpy().astype(bool)
        frame_valid = mask.any(-1)
        coverage = mask.mean(0)
        active_links = int((coverage >= 0.35).sum())
        quality = {
            "active_links": active_links,
            "link_coverage": coverage.astype(float).tolist(),
            "any_link_coverage": float(frame_valid.mean()),
            "action_confidence": float(action_probability.max()),
            "risk_confidence": float(risk_probability.max()),
            "calibrated": calibration is not None,
            "low_quality": bool(active_links < 2 or frame_valid.mean() < 0.60),
        }
        return Prediction(
            action_id=action_id,
            action_label=ACTION_LABELS[action_id],
            action_probability=action_probability.numpy().astype(float).tolist(),
            risk_id=risk_id,
            risk_label=RISK_LABELS[risk_id],
            risk_probability=risk_probability.numpy().astype(float).tolist(),
            pose_rel=pose,
            root=components["root"][0].numpy().astype(np.float32),
            frame_valid=frame_valid,
            quality=quality,
            metadata={
                "pose_coordinate": "pelvis_relative_meters",
                "root_coordinate": "GVHMR_camera_world_meters",
                "artifact_version": self.bundle_metadata.get("artifact_version", 1),
            },
        )

    def predict_csv(
        self,
        path: str | Path,
        calibration: CalibrationProfile,
    ) -> Prediction:
        csi, mask, meta = load_csv(path)
        prediction = self.predict(csi, mask, calibration)
        prediction.metadata["input_meta"] = meta
        return prediction

    def fit_calibration(
        self,
        device_id: str,
        absence_trials: Iterable[tuple[np.ndarray, np.ndarray]],
        support_trials: Iterable[SupportTrial] = (),
    ) -> CalibrationProfile:
        """Create a target profile without updating any model parameter."""

        profile = CalibrationProfile.fit_absence(device_id, absence_trials)
        display_action, retrieval_action, display_risk, retrieval_risk = [], [], [], []
        action_labels, risk_labels = [], []
        for trial in support_trials:
            values = self._forward_components(
                trial.csi, trial.link_mask, profile
            )
            display_action.append(values["display_action_logits"][0])
            retrieval_action.append(values["retrieval_action_logits"][0])
            display_risk.append(values["display_risk_logits"][0])
            retrieval_risk.append(values["retrieval_risk_logits"][0])
            action_labels.append(int(trial.action_id))
            risk_labels.append(int(trial.risk_id))
        if action_labels:
            profile.fit_logit_biases(
                torch.stack(display_action),
                torch.stack(retrieval_action),
                torch.stack(display_risk),
                torch.stack(retrieval_risk),
                torch.tensor(action_labels),
                torch.tensor(risk_labels),
            )
        return profile

    def describe(self) -> dict:
        return {
            "model_name": MODEL_NAME,
            "device": self.device,
            "artifact_dir": str(self.artifact_dir),
            "metadata": self.bundle_metadata,
            "actions": list(ACTION_LABELS),
            "risks": list(RISK_LABELS),
            "calibration_supported": True,
        }

    def warmup(self) -> None:
        """Run one synthetic window so later requests avoid cold CUDA setup."""

        csi = np.zeros((304, 3, 114, 2), np.float32)
        mask = np.ones((304, 3), bool)
        self.predict(csi, mask)

    def describe_json(self) -> str:
        return json.dumps(self.describe(), indent=2, ensure_ascii=False)
