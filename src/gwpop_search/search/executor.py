"""Deterministic end-to-end multi-fidelity search execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol

from gwpop_search.grammar import ModelGraph, ModelSpec
from gwpop_search.store import ResultStore

from .scheduler import (
    EvaluationRecord,
    Fidelity,
    PromotionDecision,
    SchedulerConfig,
    decide_promotions,
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
    root_seed: int = 20260917
    scheduler: SchedulerConfig = SchedulerConfig()
    start_fidelity: Fidelity = Fidelity.F0_SANITY
    stop_fidelity: Fidelity = Fidelity.F4_PRODUCTION

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_fidelity", Fidelity(self.start_fidelity))
        object.__setattr__(self, "stop_fidelity", Fidelity(self.stop_fidelity))
        if self.stop_fidelity.rank < self.start_fidelity.rank:
            raise ValueError("stop_fidelity cannot precede start_fidelity")


@dataclass(frozen=True)
class SearchExecutionSummary:
    root_hash: str
    n_models_registered: int
    evaluations_by_fidelity: Mapping[str, int]
    promoted_by_fidelity: Mapping[str, int]
    pruned_by_fidelity: Mapping[str, int]
    completed_fidelity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": "gwpop-search-execution-summary-1.0",
            "root_hash": self.root_hash,
            "n_models_registered": self.n_models_registered,
            "evaluations_by_fidelity": dict(self.evaluations_by_fidelity),
            "promoted_by_fidelity": dict(self.promoted_by_fidelity),
            "pruned_by_fidelity": dict(self.pruned_by_fidelity),
            "completed_fidelity": self.completed_fidelity,
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


def _existing_evaluation(
    store: ResultStore,
    *,
    model_hash: str,
    fidelity: Fidelity,
    seed: int,
) -> EvaluationRecord | None:
    rows = store.evaluations(model_hash=model_hash, fidelity=fidelity.value)
    matching = [row for row in rows if int(row["seed"]) == int(seed)]
    if len(matching) > 1:
        raise RuntimeError(
            f"multiple stored evaluations for {model_hash} {fidelity.value} seed={seed}"
        )
    return None if not matching else _record_from_row(matching[0])


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
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    for model in graph.nodes:
        store.register_model(model)

    active_hashes = [model.model_hash for model in graph.nodes]
    evaluations_by_fidelity: dict[str, int] = {}
    promoted_by_fidelity: dict[str, int] = {}
    pruned_by_fidelity: dict[str, int] = {}

    fidelity = config.start_fidelity
    while True:
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
            )
            if existing is None:
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
                    run_config={
                        "executor": "deterministic-fidelity-v1",
                        "fidelity": fidelity.value,
                    },
                    artifact_path=str(run_dir),
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

        next_fidelity = fidelity.next()
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
    )
    (artifact_root / "search_execution_summary.json").write_text(
        json.dumps(summary.to_dict(), sort_keys=True, indent=2)
    )
    return summary
