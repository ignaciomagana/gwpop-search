"""Deterministic end-to-end multi-fidelity search execution (ladder v2).

The executor steps through ``SearchExecutionConfig.scheduler.ladder`` (default
F0 -> F3 -> F4), evaluates each cohort with a :class:`FidelityEvaluator`,
records every evaluation durably in the SQLite state store and promotes with
the deterministic beam scheduler. Evaluation seeds are
``evaluation_seed(root_seed, model_hash, fidelity)``: identical whether a
rung is reached by the search or by evidence completion, so F3 evidence is
path independent.

Every stored evaluation row carries ``run_config = {"executor":
"deterministic-fidelity-v2", "backend": "dynesty", "fidelity": F,
"fidelity_config_sha256": H}`` (evidence completion writes
``"evidence-completion-v2"``). A state database holding rows of any other
executor (NUTS/JAXNS-era 1.x rows, whose run ids and seeds coincide with the
new ones) is refused with :class:`LegacyStateError` instead of being silently
reused, and a row produced under a different frozen numerical configuration
``H`` is refused with :class:`FidelityConfigMismatchError`: evaluation seeds
do not depend on the fidelity configuration, so re-freezing the numerics
against an existing state database would otherwise reuse stale evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Protocol

from gwpop_search.grammar import ModelGraph, ModelSpec
from gwpop_search.store import ResultStore

from .scheduler import (
    DEFAULT_LADDER,
    SCHEDULER_VERSION,
    EvaluationRecord,
    Fidelity,
    PromotionDecision,
    SchedulerConfig,
    decide_promotions,
    next_in_ladder,
)

EXECUTOR_VERSION = "deterministic-fidelity-v2"
COMPLETION_EXECUTOR_VERSION = "evidence-completion-v2"
V2_EXECUTORS = (EXECUTOR_VERSION, COMPLETION_EXECUTOR_VERSION)
SAMPLER_BACKEND = "dynesty"
EXECUTION_SUMMARY_FORMAT_VERSION = "gwpop-search-execution-summary-1.1"


class SearchBudgetExceeded(RuntimeError):
    """Raised before a deterministic search exceeds a frozen compute/model budget."""


class LegacyStateError(RuntimeError):
    """The state database holds evaluations of a retired (NUTS/JAXNS-era) executor."""


class FidelityConfigMismatchError(RuntimeError):
    """A stored evaluation was produced under a different frozen fidelity config."""


def evaluation_run_config(
    fidelity,
    *,
    executor: str = EXECUTOR_VERSION,
    fidelity_config_sha256: str | None = None,
) -> dict[str, str]:
    """The ``run_config`` stored with every v2 evaluation row.

    ``fidelity_config_sha256`` pins the frozen numerical configuration the row
    was produced under; rows written without it cannot be reused by a run that
    knows its own configuration hash.
    """
    if executor not in V2_EXECUTORS:
        raise ValueError(f"unknown executor {executor!r}")
    payload = {
        "executor": executor,
        "backend": SAMPLER_BACKEND,
        "fidelity": Fidelity(fidelity).value,
    }
    if fidelity_config_sha256 is not None:
        payload["fidelity_config_sha256"] = str(fidelity_config_sha256)
    return payload


def _row_run_config(row: Mapping[str, object]) -> dict[str, object] | None:
    try:
        payload = json.loads(str(row["run_config_json"]))
    except (KeyError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def row_executor(row: Mapping[str, object]) -> str | None:
    """The executor recorded in a state-store row (``None`` if absent)."""
    payload = _row_run_config(row)
    return None if payload is None else payload.get("executor")


def row_fidelity_config_sha256(row: Mapping[str, object]) -> str | None:
    """The frozen fidelity-config hash recorded in a row (``None`` if absent)."""
    payload = _row_run_config(row)
    if payload is None:
        return None
    value = payload.get("fidelity_config_sha256")
    return None if value is None else str(value)


def require_fidelity_config_identity(
    row: Mapping[str, object],
    expected: str | None,
) -> None:
    """Refuse reuse of a row produced under a different frozen numerical config.

    ``expected is None`` means the caller does not know its own configuration
    hash (e.g. a stub evaluator) and no check is possible.
    """
    if expected is None:
        return
    stored = row_fidelity_config_sha256(row)
    if stored == str(expected):
        return
    raise FidelityConfigMismatchError(
        f"stored evaluation {row['run_id']} was produced under fidelity config "
        f"{stored!r}, not the frozen {str(expected)!r}; evaluation seeds do not "
        "depend on the numerical configuration, so reusing it would silently mix "
        "two freezes. Re-run against a new state database, or restore the "
        "configuration the rows were produced with."
    )


def require_v2_state(store: ResultStore) -> None:
    """Refuse a state database that holds rows of a retired executor."""
    legacy = [
        (str(row["run_id"]), row_executor(row))
        for row in store.evaluations()
        if row_executor(row) not in V2_EXECUTORS
    ]
    if legacy:
        preview = ", ".join(f"{run_id} ({executor})" for run_id, executor in legacy[:4])
        raise LegacyStateError(
            f"state database {store.path} holds {len(legacy)} evaluation row(s) of a retired "
            f"executor (NUTS/JAXNS-era 1.x or unknown): {preview}. Run IDs and seeds of the "
            "dynesty ladder coincide with those rows, so they would be reused silently; "
            "use a new state database."
        )


class FidelityEvaluator(Protocol):
    """External numerical evaluator used by the deterministic search loop."""

    def evaluate(
        self,
        model: ModelSpec,
        fidelity: Fidelity,
        *,
        seed: int,
        run_dir: Path,
    ) -> EvaluationRecord: ...


@dataclass(frozen=True)
class SearchExecutionConfig:
    """Seeds, ladder window and frozen budgets of one search execution.

    ``start_fidelity`` and ``stop_fidelity`` must be rungs of
    ``scheduler.ladder``. ``max_total_compute_cost`` is in the evaluator's
    ``compute_cost`` units (wall-clock hours).
    """

    root_seed: int = 20260917
    scheduler: SchedulerConfig = SchedulerConfig()
    start_fidelity: Fidelity = Fidelity.F0_SANITY
    stop_fidelity: Fidelity = Fidelity.F4_PRODUCTION
    max_models_by_fidelity: Mapping[str, int] = field(default_factory=dict)
    max_total_compute_cost: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_fidelity", Fidelity(self.start_fidelity))
        object.__setattr__(self, "stop_fidelity", Fidelity(self.stop_fidelity))
        ladder = self.scheduler.ladder
        for name in ("start_fidelity", "stop_fidelity"):
            value = getattr(self, name).value
            if value not in ladder:
                raise ValueError(
                    f"{name} {value} is not a rung of the scheduler ladder {list(ladder)}"
                )
        if self.stop_fidelity.rank < self.start_fidelity.rank:
            raise ValueError("stop_fidelity cannot precede start_fidelity")
        limits = {str(key): int(value) for key, value in self.max_models_by_fidelity.items()}
        valid = {item.value for item in Fidelity}
        unknown = set(limits) - valid
        if unknown:
            raise ValueError(f"unknown fidelity budget key(s): {sorted(unknown)}")
        if any(value <= 0 for value in limits.values()):
            raise ValueError("model-count limits must be positive")
        object.__setattr__(self, "max_models_by_fidelity", limits)
        if self.max_total_compute_cost is not None:
            if self.max_total_compute_cost <= 0:
                raise ValueError("max_total_compute_cost must be positive")


@dataclass(frozen=True)
class SearchExecutionSummary:
    root_hash: str
    n_models_registered: int
    evaluations_by_fidelity: Mapping[str, int]
    promoted_by_fidelity: Mapping[str, int]
    pruned_by_fidelity: Mapping[str, int]
    completed_fidelity: str
    total_compute_cost: float
    ladder: tuple[str, ...] = DEFAULT_LADDER
    scheduler_version: str = SCHEDULER_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": EXECUTION_SUMMARY_FORMAT_VERSION,
            "executor": EXECUTOR_VERSION,
            "sampler_backend": SAMPLER_BACKEND,
            "ladder": list(self.ladder),
            "scheduler_version": self.scheduler_version,
            "root_hash": self.root_hash,
            "n_models_registered": self.n_models_registered,
            "evaluations_by_fidelity": dict(self.evaluations_by_fidelity),
            "promoted_by_fidelity": dict(self.promoted_by_fidelity),
            "pruned_by_fidelity": dict(self.pruned_by_fidelity),
            "completed_fidelity": self.completed_fidelity,
            "total_compute_cost": float(self.total_compute_cost),
        }


def evaluation_seed(
    root_seed: int,
    model_hash: str,
    fidelity: Fidelity,
) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{model_hash}:{fidelity.value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def evaluation_run_id(
    model_hash: str,
    fidelity: Fidelity,
    seed: int,
) -> str:
    return f"{fidelity.value}-{model_hash[:16]}-{int(seed):010d}"


def _record_from_row(row: Mapping[str, object]) -> EvaluationRecord:
    return EvaluationRecord(
        model_hash=str(row["model_hash"]),
        fidelity=Fidelity(str(row["fidelity"])),
        diagnostics_pass=bool(row["diagnostics_pass"]),
        screen_value=(
            None if row["screen_value"] is None else float(row["screen_value"])
        ),
        compute_cost=float(row["compute_cost"]),
        status=str(row["status"]),
    )


def _total_compute_cost(store: ResultStore) -> float:
    """Recompute cost from durable rows so live and resumed sums are identical."""
    return float(
        math.fsum(float(row["compute_cost"]) for row in store.evaluations())
    )


def _existing_evaluation(
    store: ResultStore,
    *,
    model_hash: str,
    fidelity: Fidelity,
    seed: int,
    fidelity_config_sha256: str | None = None,
) -> EvaluationRecord | None:
    rows = store.evaluations(model_hash=model_hash, fidelity=fidelity.value)
    matching = [row for row in rows if int(row["seed"]) == int(seed)]
    if len(matching) > 1:
        raise RuntimeError(
            f"multiple stored evaluations for {model_hash} {fidelity.value} seed={seed}"
        )
    if not matching:
        return None
    executor = row_executor(matching[0])
    if executor not in V2_EXECUTORS:
        raise LegacyStateError(
            f"stored evaluation {matching[0]['run_id']} was written by executor "
            f"{executor!r}, not by the dynesty ladder; refusing to reuse it"
        )
    require_fidelity_config_identity(matching[0], fidelity_config_sha256)
    return _record_from_row(matching[0])


def execute_search(
    graph: ModelGraph,
    evaluator: FidelityEvaluator,
    *,
    state_database: str | Path,
    artifact_root: str | Path,
    config: SearchExecutionConfig | None = None,
) -> SearchExecutionSummary:
    """Run/resume a deterministic finite search with durable evaluation state."""
    config = SearchExecutionConfig() if config is None else config
    store = ResultStore(state_database)
    require_v2_state(store)
    supported = getattr(evaluator, "supported_fidelities", None)
    if supported is not None:
        missing = [item for item in config.scheduler.ladder if item not in tuple(supported)]
        if missing:
            raise ValueError(
                f"evaluator does not support ladder rung(s) {missing}; supported: "
                f"{list(supported)}"
            )
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    # Evaluators that carry a frozen numerical configuration expose its hash;
    # every row this run writes records it, and every row it reuses must match.
    config_sha256 = getattr(evaluator, "fidelity_config_sha256", None)
    config_sha256 = None if config_sha256 is None else str(config_sha256)

    for model in graph.nodes:
        store.register_model(model)

    active_hashes = [model.model_hash for model in graph.nodes]
    evaluations_by_fidelity: dict[str, int] = {}
    promoted_by_fidelity: dict[str, int] = {}
    pruned_by_fidelity: dict[str, int] = {}

    total_compute_cost = _total_compute_cost(store)

    fidelity = config.start_fidelity
    while True:
        model_limit = config.max_models_by_fidelity.get(fidelity.value)
        if model_limit is not None and len(active_hashes) > model_limit:
            raise SearchBudgetExceeded(
                f"{fidelity.value} has {len(active_hashes)} active models, "
                f"exceeding frozen limit {model_limit}"
            )

        records: list[EvaluationRecord] = []
        for model_hash in sorted(active_hashes):
            model = graph.by_hash[model_hash]
            seed = evaluation_seed(config.root_seed, model_hash, fidelity)
            run_id = evaluation_run_id(model_hash, fidelity, seed)
            existing = _existing_evaluation(
                store,
                model_hash=model_hash,
                fidelity=fidelity,
                seed=seed,
                fidelity_config_sha256=config_sha256,
            )
            if existing is None:
                if (
                    config.max_total_compute_cost is not None
                    and total_compute_cost >= config.max_total_compute_cost
                ):
                    raise SearchBudgetExceeded(
                        "frozen compute budget exhausted before the next evaluation"
                    )
                run_dir = artifact_root / fidelity.value / model_hash
                run_dir.mkdir(parents=True, exist_ok=True)
                record = evaluator.evaluate(
                    model,
                    fidelity,
                    seed=seed,
                    run_dir=run_dir,
                )
                if record.model_hash != model_hash or record.fidelity is not fidelity:
                    raise ValueError(
                        "evaluator returned a record for the wrong model/fidelity"
                    )
                store.record_evaluation(
                    run_id,
                    record,
                    seed=seed,
                    run_config=evaluation_run_config(
                        fidelity,
                        fidelity_config_sha256=config_sha256,
                    ),
                    artifact_path=str(run_dir),
                )
                total_compute_cost = _total_compute_cost(store)
                if (
                    config.max_total_compute_cost is not None
                    and total_compute_cost > config.max_total_compute_cost
                ):
                    raise SearchBudgetExceeded(
                        "frozen compute budget was exceeded by the completed "
                        f"evaluation: {total_compute_cost:.6g} > "
                        f"{config.max_total_compute_cost:.6g}"
                    )
            else:
                record = existing
            records.append(record)

        evaluations_by_fidelity[fidelity.value] = len(records)

        if fidelity is config.stop_fidelity:
            decisions = tuple(
                PromotionDecision(
                    model_hash=record.model_hash,
                    from_fidelity=fidelity,
                    to_fidelity=None,
                    decision="complete",
                    reason="configured_stop_fidelity",
                    scheduler_version=config.scheduler.version,
                )
                for record in sorted(records, key=lambda item: item.model_hash)
            )
        else:
            decisions = decide_promotions(records, config=config.scheduler)

        store.record_promotions(decisions)
        promoted = [
            decision.model_hash
            for decision in decisions
            if decision.decision == "promote"
        ]
        pruned = [
            decision.model_hash
            for decision in decisions
            if decision.decision == "prune"
        ]
        promoted_by_fidelity[fidelity.value] = len(promoted)
        pruned_by_fidelity[fidelity.value] = len(pruned)

        if fidelity is config.stop_fidelity or not promoted:
            break

        next_fidelity = next_in_ladder(fidelity, config.scheduler.ladder)
        if next_fidelity is None:
            break
        active_hashes = promoted
        fidelity = next_fidelity

    summary = SearchExecutionSummary(
        root_hash=graph.root_hash,
        n_models_registered=len(graph.nodes),
        evaluations_by_fidelity=evaluations_by_fidelity,
        promoted_by_fidelity=promoted_by_fidelity,
        pruned_by_fidelity=pruned_by_fidelity,
        completed_fidelity=fidelity.value,
        total_compute_cost=total_compute_cost,
        ladder=tuple(config.scheduler.ladder),
        scheduler_version=config.scheduler.version,
    )
    (artifact_root / "search_execution_summary.json").write_text(
        json.dumps(summary.to_dict(), sort_keys=True, indent=2)
    )
    return summary
