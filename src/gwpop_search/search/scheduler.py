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
        """The next member in *enum order* (F0..F4).

        This is not the promotion order: the search follows
        ``SchedulerConfig.ladder`` (default F0 -> F3 -> F4, see
        :func:`next_in_ladder`); F1/F2 are kept only for store compatibility.
        """
        members = tuple(Fidelity)
        index = members.index(self)
        return None if index + 1 == len(members) else members[index + 1]


# Fidelity ladder v2 (dynesty): F1 (NUTS screen) and F2 (NUTS posterior) left the
# default ladder; their enum values remain so historical rows stay readable.
DEFAULT_LADDER = ("F0", "F3", "F4")
EVIDENCE_FIDELITIES = ("F3", "F4")
SCHEDULER_VERSION = "deterministic-beam-v2"


def validate_ladder(ladder) -> tuple[str, ...]:
    """Return a validated ladder of fidelity values.

    The ladder starts with F0, strictly increases in rank, and contains at
    least one evidence rung (F3 or F4).
    """
    if isinstance(ladder, str):
        raise TypeError("ladder must be a sequence of fidelity values, not a string")
    values = tuple(Fidelity(item).value for item in ladder)
    if not values:
        raise ValueError("ladder cannot be empty")
    if values[0] != Fidelity.F0_SANITY.value:
        raise ValueError("ladder must start with F0")
    ranks = [Fidelity(item).rank for item in values]
    if any(b <= a for a, b in zip(ranks[:-1], ranks[1:], strict=True)):
        raise ValueError(f"ladder fidelities must strictly increase; got {values}")
    if not set(values) & set(EVIDENCE_FIDELITIES):
        raise ValueError("ladder must contain an evidence rung (F3 or F4)")
    return values


def next_in_ladder(fidelity, ladder) -> "Fidelity | None":
    """The rung after ``fidelity`` in ``ladder`` (``None`` at the top)."""
    fidelity = Fidelity(fidelity)
    values = validate_ladder(ladder)
    if fidelity.value not in values:
        raise ValueError(f"{fidelity.value} is not a rung of the ladder {values}")
    index = values.index(fidelity.value)
    return None if index + 1 == len(values) else Fidelity(values[index + 1])


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
    """Deterministic beam scheduler over an explicit fidelity ladder.

    F0 promotes every passing model to the next rung; later rungs promote the
    top ``beam_width`` by screen value plus ``exploration_quota`` seeded
    exploration picks. ``version`` is part of the frozen campaign hash; the
    v1 (NUTS-era, enum-order) scheduler is refused.
    """

    beam_width: int = 8
    exploration_quota: int = 2
    seed: int = 20260917
    ladder: tuple[str, ...] = DEFAULT_LADDER
    version: str = SCHEDULER_VERSION

    def __post_init__(self) -> None:
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.exploration_quota < 0:
            raise ValueError("exploration_quota cannot be negative")
        if self.version != SCHEDULER_VERSION:
            raise ValueError(
                f"unsupported scheduler version {self.version!r}; the NUTS-era "
                f"{self.version!r} scheduler was replaced by {SCHEDULER_VERSION!r} "
                "(re-freeze the campaign)"
            )
        object.__setattr__(self, "ladder", validate_ladder(self.ladder))


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
    """Choose the next-rung allocation from one completed fidelity cohort.

    The next rung follows ``config.ladder``. The screen value is an
    allocation statistic only. This function never changes stored evidence or
    production results.
    """
    config = SchedulerConfig() if config is None else config
    records = tuple(records)
    if not records:
        return ()

    fidelities = {record.fidelity for record in records}
    if len(fidelities) != 1:
        raise ValueError("one scheduling decision must contain exactly one fidelity")
    fidelity = next(iter(fidelities))
    next_fidelity = next_in_ladder(fidelity, config.ladder)

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
