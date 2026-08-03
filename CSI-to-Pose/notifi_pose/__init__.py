"""notifi_pose — 3링크 CSI → GVHMR 22관절 pose 복원 파이프라인 (v2).

legacy `scripts/` 는 MediaPipe 13관절 · 단일링크 · ambient split 기반이라 현재
데이터셋(`NotiFi_CSI_GVHMR_v2_LOSO_60_15_25`)을 읽지 못한다. 이 패키지가 그 대체다.
설계 근거는 `_local/GVHMR_MIGRATION_PLAN_2026-08-01.md` 참조.
"""

__version__ = "0.1.0"
