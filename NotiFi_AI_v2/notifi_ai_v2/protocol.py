"""Leakage-resistant source development and sealed evaluation rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .constants import EXCLUDED_SITES, SEALED_SITE, SOURCE_SITES


class ProtocolError(ValueError):
    """Raised when a trial violates the locked evaluation contract."""


class SplitRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    SOURCE_TEST = "source_test"
    SEALED_SUPPORT = "sealed_support"
    SEALED_QUERY = "sealed_query"


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    subject: str
    environment: str
    role: SplitRole

    @property
    def site(self) -> tuple[str, str]:
        return self.subject, self.environment


def validate_protocol(records: Iterable[TrialRecord]) -> list[TrialRecord]:
    """Validate sites, role boundaries, and support-query disjointness."""

    rows = list(records)
    trial_ids = [row.trial_id for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise ProtocolError("trial_id values must be unique across all roles")

    for row in rows:
        if not row.trial_id:
            raise ProtocolError("trial_id must not be empty")
        if row.site in EXCLUDED_SITES:
            raise ProtocolError(f"excluded GT site cannot be used: {row.site}")
        is_sealed_role = row.role in {
            SplitRole.SEALED_SUPPORT,
            SplitRole.SEALED_QUERY,
        }
        if row.site == SEALED_SITE and not is_sealed_role:
            raise ProtocolError("yja/E02 cannot enter source development")
        if is_sealed_role and row.site != SEALED_SITE:
            raise ProtocolError("sealed roles are reserved for yja/E02")
        if not is_sealed_role and row.site not in SOURCE_SITES:
            raise ProtocolError(f"unregistered source site: {row.site}")
    return rows


def assert_selection_is_source_only(records: Iterable[TrialRecord]) -> None:
    """Reject sealed trials from epoch, threshold, or hyperparameter selection."""

    rows = validate_protocol(records)
    sealed = [row.trial_id for row in rows if row.site == SEALED_SITE]
    if sealed:
        raise ProtocolError(
            "sealed support/query metrics cannot select a model: "
            + ", ".join(sealed[:3])
        )
