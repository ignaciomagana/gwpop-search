"""Deterministic event-drop stress suites for a frozen model search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

from gwpop_search.data import drop_posterior_events
from gwpop_search.grammar import ModelGraph
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.inference.numpyro import _code_identity
from gwpop_search.production import ProductionCampaignConfig
from gwpop_search.production.runner import collect_best_available_evidence
from gwpop_search.search import (
    Fidelity,
    ModelEvidence,
    SearchExecutionConfig,
    execute_search,
)


_SCENARIO_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class EventDropScenario:
    scenario_id: str
    drop_events: tuple[str, ...]
    category: str = "custom"
    note: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id or not _SCENARIO_ID.match(self.scenario_id):
            raise ValueError(
                "scenario_id must contain only letters, numbers, _, ., or -"
            )
        events = tuple(str(name) for name in self.drop_events)
        if not events:
            raise ValueError("event-drop scenario requires at least one event")
        if len(set(events)) != len(events):
            raise ValueError("drop_events must be unique")
        object.__setattr__(self, "drop_events", events)
        if self.category not in {
            "leave_one_out",
            "loud_event",
            "custom",
        }:
            raise ValueError("unsupported event-drop scenario category")


@dataclass(frozen=True)
class EventStressConfig:
    stop_fidelity: Fidelity = Fidelity.F3_EVIDENCE
    max_gpu_hours_per_scenario: float = 250.0
    max_f3_models: int = 20
    max_f4_models: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stop_fidelity",
            Fidelity(self.stop_fidelity),
        )
        if self.stop_fidelity.rank < Fidelity.F2_INFERENCE.rank:
            raise ValueError("stress suite must run through at least F2")
        if (
            not math.isfinite(self.max_gpu_hours_per_scenario)
            or self.max_gpu_hours_per_scenario <= 0.0
        ):
            raise ValueError("max_gpu_hours_per_scenario must be positive")
        if self.max_f3_models <= 0 or self.max_f4_models <= 0:
            raise ValueError("stress model limits must be positive")


def leave_one_out_scenarios(
    event_names: Iterable[str],
) -> tuple[EventDropScenario, ...]:
    names = tuple(str(name) for name in event_names)
    if len(set(names)) != len(names):
        raise ValueError("event names must be unique")
    return tuple(
        EventDropScenario(
            scenario_id=f"loo_{index:03d}_{name}",
            drop_events=(name,),
            category="leave_one_out",
            note=f"leave out {name}",
        )
        for index, name in enumerate(names)
    )


def stress_seed(
    root_seed: int,
    scenario_id: str,
) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{scenario_id}:event-stress-v1".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def stress_dataset_identity(
    base_dataset_identity: str,
    scenario: EventDropScenario,
) -> str:
    payload = {
        "base_dataset_identity": str(base_dataset_identity),
        "scenario": asdict(scenario),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def edge_log_bayes_factors(
    graph: ModelGraph,
    evidences: Mapping[str, ModelEvidence],
) -> dict[str, dict[str, object]]:
    result = {}
    for edge in graph.edges:
        if edge.parent_hash not in evidences or edge.child_hash not in evidences:
            continue
        key = (
            f"{edge.parent_hash[:12]}->{edge.child_hash[:12]}:"
            f"{edge.mutation_id}"
        )
        result[key] = {
            "parent_hash": edge.parent_hash,
            "child_hash": edge.child_hash,
            "mutation_id": edge.mutation_id,
            "log_bayes_factor": float(
                evidences[edge.child_hash].log_evidence
                - evidences[edge.parent_hash].log_evidence
            ),
        }
    return result


def compare_edge_bayes_factors(
    reference: Mapping[str, dict[str, object]],
    stressed: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    common = sorted(set(reference) & set(stressed))
    rows = []
    for key in common:
        reference_bf = float(reference[key]["log_bayes_factor"])
        stressed_bf = float(stressed[key]["log_bayes_factor"])
        rows.append(
            {
                "edge_id": key,
                "mutation_id": stressed[key]["mutation_id"],
                "reference_log_bayes_factor": reference_bf,
                "stress_log_bayes_factor": stressed_bf,
                "delta_log_bayes_factor": stressed_bf - reference_bf,
            }
        )
    return {
        "n_common_evidence_edges": len(rows),
        "max_abs_delta_log_bayes_factor": (
            None
            if not rows
            else float(
                max(abs(row["delta_log_bayes_factor"]) for row in rows)
            )
        ),
        "edges": rows,
    }


def build_event_stress_plan(
    *,
    campaign: ProductionCampaignConfig,
    base_dataset_identity: str,
    graph: ModelGraph,
    scenarios: Iterable[EventDropScenario],
    config: EventStressConfig,
) -> dict[str, object]:
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("stress suite requires at least one scenario")
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("stress scenario IDs must be unique")
    return {
        "format_version": "gwpop-search-event-stress-plan-1.0",
        "code": _code_identity(),
        "campaign_hash": campaign.campaign_hash,
        "base_dataset_identity": str(base_dataset_identity),
        "graph_root_hash": graph.root_hash,
        "config": {
            "stop_fidelity": config.stop_fidelity.value,
            "max_gpu_hours_per_scenario": config.max_gpu_hours_per_scenario,
            "max_f3_models": config.max_f3_models,
            "max_f4_models": config.max_f4_models,
        },
        "scenarios": [asdict(item) for item in scenarios],
    }


def _write_plan_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "existing event stress plan does not match requested suite"
            )
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def run_event_drop_stress_suite(
    root: str | Path,
    posterior,
    selection,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    *,
    base_dataset_identity: str,
    scenarios: Iterable[EventDropScenario],
    config: EventStressConfig | None = None,
    reference_state_database: str | Path | None = None,
) -> dict[str, object]:
    """Run/resume explicit event-drop searches under the frozen HBI/search stack."""
    config = EventStressConfig() if config is None else config
    scenarios = tuple(scenarios)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    plan = build_event_stress_plan(
        campaign=campaign,
        base_dataset_identity=base_dataset_identity,
        graph=graph,
        scenarios=scenarios,
        config=config,
    )
    _write_plan_once(root / "stress_plan.json", plan)

    reference_edges = {}
    if reference_state_database is not None:
        reference_edges = edge_log_bayes_factors(
            graph,
            collect_best_available_evidence(reference_state_database),
        )

    scenario_results = []
    for scenario in scenarios:
        subset = drop_posterior_events(
            posterior,
            scenario.drop_events,
            reason=f"{scenario.category}:{scenario.scenario_id}",
        )
        scenario_root = root / scenario.scenario_id
        artifacts = scenario_root / "artifacts"
        database = scenario_root / "state.sqlite"

        evaluator = DeterministicHBIEvaluator(
            subset,
            selection,
            config=campaign.fidelity,
            dataset_identity=stress_dataset_identity(
                base_dataset_identity,
                scenario,
            ),
        )
        execution = execute_search(
            graph,
            evaluator,
            state_database=database,
            artifact_root=artifacts,
            config=SearchExecutionConfig(
                root_seed=stress_seed(
                    campaign.seed_policy.root_seed,
                    scenario.scenario_id,
                ),
                scheduler=campaign.scheduler,
                stop_fidelity=config.stop_fidelity,
                max_models_by_fidelity={
                    "F3": config.max_f3_models,
                    "F4": config.max_f4_models,
                },
                max_total_compute_cost=config.max_gpu_hours_per_scenario,
            ),
        )

        stressed_evidence = collect_best_available_evidence(database)
        stressed_edges = edge_log_bayes_factors(
            graph,
            stressed_evidence,
        )
        comparison = (
            None
            if not reference_edges
            else compare_edge_bayes_factors(
                reference_edges,
                stressed_edges,
            )
        )
        result = {
            "scenario": asdict(scenario),
            "n_events": int(subset.n_events),
            "dataset_identity": stress_dataset_identity(
                base_dataset_identity,
                scenario,
            ),
            "execution": execution.to_dict(),
            "n_models_with_evidence": len(stressed_evidence),
            "n_evidence_edges": len(stressed_edges),
            "edge_bayes_factors": stressed_edges,
            "reference_comparison": comparison,
        }
        (scenario_root / "stress_summary.json").write_text(
            json.dumps(result, sort_keys=True, indent=2)
        )
        scenario_results.append(result)

    max_shifts = [
        item["reference_comparison"]["max_abs_delta_log_bayes_factor"]
        for item in scenario_results
        if (
            item["reference_comparison"] is not None
            and item["reference_comparison"][
                "max_abs_delta_log_bayes_factor"
            ]
            is not None
        )
    ]
    summary = {
        "format_version": "gwpop-search-event-stress-summary-1.0",
        "n_scenarios": len(scenario_results),
        "reference_evidence_available": bool(reference_edges),
        "max_abs_delta_log_bayes_factor_across_scenarios": (
            None if not max_shifts else float(max(max_shifts))
        ),
        "scenarios": scenario_results,
    }
    (root / "stress_suite_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
