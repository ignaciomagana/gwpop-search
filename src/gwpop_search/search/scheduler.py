"""Deterministic multi-fidelity scheduling for finite model graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Iterable


class Fidelity(str, Enum):
    F0_SANITY = "F0"
    F1_SCREEN = "F1"
    F2_INFERENCE = "F2"
    F3_EVIDENCE = "F3"
    F4_PRODUCTION = "F4"

    @property
    def rank(self) -> int:
        return tuple(Fidelity).index(self)

    def next(self) -> "Fidelity | None":
        members = tuple(Fidelity)
        index = members.index(self)
        return None if index + 1 == len(members) else members[index + 1]


@dataclass(frozen=True)
class EvaluationRecord:
    model_hash: str
    fidelity: Fidelity
    diagnostics_pass: bool
    screen_value: float | None = None
    compute_cost: float = 0.0
    status: str = "complete"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fidelity", Fidelity(self.fidelity))
        if self.status not in {"complete", "failed"}:
            raise ValueError("status must be complete or failed")
        if self.screen_value is not None and not math.isfinite(self.screen_value):
            raise ValueError("screen_value must be finite when supplied")
        if not math.isfinite(self.compute_cost) or self.compute_cost < 0:
            raise ValueError("compute_cost must be finite and non-negative")


@dataclass(frozen=True)
class SchedulerConfig:
    beam_width: int = 8
    exploration_quota: int = 2
    seed: int = 20260917
    version: str = "deterministic-beam-v1"

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.exploration_quota < 0:
            raise ValueError("exploration_quota cannot be negative")


@dataclass(frozen=True)
class PromotionDecision:
    model_hash: str
    from_fidelity: Fidelity
    to_fidelity: Fidelity | None
    decision: str
    reason: str
    scheduler_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_hash": self.model_hash,
            "from_fidelity": self.from_fidelity.value,
            "to_fidelity": (
                None if self.to_fidelity is None else self.to_fidelity.value
            ),
            "decision": self.decision,
            "reason": self.reason,
            "scheduler_version": self.scheduler_version,
        }


def _exploration_key(seed: int, model_hash: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}:{model_hash}".encode("utf-8")
    ).hexdigest()


def decide_promotions(
    records: Iterable[EvaluationRecord],
    *,
    config: SchedulerConfig | None = None,
) -> tuple[PromotionDecision, ...]:
    """Choose the next-fidelity allocation from one completed fidelity cohort.

    The screen value is an allocation statistic only. This function never
    changes stored evidence or production results.
    """
    config = SchedulerConfig() if config is None else config
    records = tuple(records)
    if not records:
        return ()

    fidelities = {record.fidelity for record in records}
    if len(fidelities) != 1:
        raise ValueError("one scheduling decision must contain exactly one fidelity")
    fidelity = next(iter(fidelities))
    next_fidelity = fidelity.next()

    by_hash = {}
    for record in records:
        if record.model_hash in by_hash:
            raise ValueError(f"duplicate evaluation record for {record.model_hash}")
        by_hash[record.model_hash] = record

    decisions: dict[str, PromotionDecision] = {}
    eligible = []
    for record in records:
        if record.status != "complete":
            decisions[record.model_hash] = PromotionDecision(
                record.model_hash,
                fidelity,
                None,
                "prune",
                "evaluation_failed",
                config.version,
            )
        elif not record.diagnostics_pass:
            decisions[record.model_hash] = PromotionDecision(
                record.model_hash,
                fidelity,
                None,
                "prune",
                "diagnostic_veto",
                config.version,
            )
        elif next_fidelity is None:
            decisions[record.model_hash] = PromotionDecision(
                record.model_hash,
                fidelity,
                None,
                "complete",
                "production_complete",
                config.version,
            )
        else:
            eligible.append(record)

    if next_fidelity is not None and eligible:
        if fidelity is Fidelity.F0_SANITY:
            exploit = sorted(eligible, key=lambda item: item.model_hash)
            explore = []
        else:
            ranked = sorted(
                eligible,
                key=lambda item: (
                    -(
                        item.screen_value
                        if item.screen_value is not None
                        else -math.inf
                    ),
                    item.model_hash,
                ),
            )
            exploit = ranked[: config.beam_width]
            remaining = ranked[config.beam_width :]
            explore = sorted(
                remaining,
                key=lambda item: _exploration_key(config.seed, item.model_hash),
            )[: config.exploration_quota]

        exploit_hashes = {item.model_hash for item in exploit}
        explore_hashes = {item.model_hash for item in explore}

        for record in eligible:
            if record.model_hash in exploit_hashes:
                decisions[record.model_hash] = PromotionDecision(
                    record.model_hash,
                    fidelity,
                    next_fidelity,
                    "promote",
                    "beam",
                    config.version,
                )
            elif record.model_hash in explore_hashes:
                decisions[record.model_hash] = PromotionDecision(
                    record.model_hash,
                    fidelity,
                    next_fidelity,
                    "promote",
                    "exploration_quota",
                    config.version,
                )
            else:
                decisions[record.model_hash] = PromotionDecision(
                    record.model_hash,
                    fidelity,
                    None,
                    "prune",
                    "not_selected_at_this_fidelity",
                    config.version,
                )

    return tuple(
        decisions[model_hash]
        for model_hash in sorted(decisions)
    )
