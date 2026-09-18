"""Nearby-baseline reruns for deterministic search robustness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from gwpop_search.grammar import (
    ModelGraph,
    ModelSpec,
    enumerate_model_graph,
)
from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
from gwpop_search.inference.numpyro import _code_identity
from gwpop_search.production import ProductionCampaignConfig
from gwpop_search.production.runner import (
    collect_best_available_evidence,
    model_prior_from_config,
    write_scientific_scoring,
)
from gwpop_search.search import (
    Fidelity,
    ModelEvidence,
    SearchExecutionConfig,
    execute_search,
)

from .stress import edge_log_bayes_factors


_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class NearbyBaselineScenario:
    scenario_id: str
    root_spec: ModelSpec
    max_depth: int = 1
    max_models: int = 20
    note: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id or not _ID.match(self.scenario_id):
            raise ValueError(
                "scenario_id must contain only letters, numbers, _, ., or -"
            )
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if self.max_models <= 0:
            raise ValueError("max_models must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "root_spec": self.root_spec.to_dict(),
            "max_depth": int(self.max_depth),
            "max_models": int(self.max_models),
            "note": self.note,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "NearbyBaselineScenario":
        return cls(
            scenario_id=str(payload["scenario_id"]),
            root_spec=ModelSpec.from_dict(dict(payload["root_spec"])),
            max_depth=int(payload.get("max_depth", 1)),
            max_models=int(payload.get("max_models", 20)),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class NearbyBaselineConfig:
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
            raise ValueError("nearby-baseline suite must reach at least F2")
        if (
            not math.isfinite(self.max_gpu_hours_per_scenario)
            or self.max_gpu_hours_per_scenario <= 0.0
        ):
            raise ValueError("max_gpu_hours_per_scenario must be positive")
        if self.max_f3_models <= 0 or self.max_f4_models <= 0:
            raise ValueError("nearby-baseline model limits must be positive")


@dataclass(frozen=True)
class NearbyBaselineSuiteSpec:
    scenarios: tuple[NearbyBaselineScenario, ...]
    config: NearbyBaselineConfig
    format_version: str = "gwpop-search-nearby-baseline-suite-1.0"

    def __post_init__(self) -> None:
        if self.format_version != "gwpop-search-nearby-baseline-suite-1.0":
            raise ValueError("unsupported nearby-baseline suite format")
        scenarios = tuple(self.scenarios)
        if not scenarios:
            raise ValueError("nearby-baseline suite requires scenarios")
        ids = [item.scenario_id for item in scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError("nearby-baseline scenario IDs must be unique")
        object.__setattr__(self, "scenarios", scenarios)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "config": {
                "stop_fidelity": self.config.stop_fidelity.value,
                "max_gpu_hours_per_scenario": (
                    self.config.max_gpu_hours_per_scenario
                ),
                "max_f3_models": self.config.max_f3_models,
                "max_f4_models": self.config.max_f4_models,
            },
            "scenarios": [item.to_dict() for item in self.scenarios],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "NearbyBaselineSuiteSpec":
        config = dict(payload["config"])
        return cls(
            scenarios=tuple(
                NearbyBaselineScenario.from_dict(item)
                for item in payload["scenarios"]
            ),
            config=NearbyBaselineConfig(
                stop_fidelity=Fidelity(str(config["stop_fidelity"])),
                max_gpu_hours_per_scenario=float(
                    config["max_gpu_hours_per_scenario"]
                ),
                max_f3_models=int(config["max_f3_models"]),
                max_f4_models=int(config["max_f4_models"]),
            ),
            format_version=str(
                payload.get(
                    "format_version",
                    "gwpop-search-nearby-baseline-suite-1.0",
                )
            ),
        )


def save_nearby_baseline_suite_spec(
    path: str | Path,
    spec: NearbyBaselineSuiteSpec,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), sort_keys=True, indent=2))


def load_nearby_baseline_suite_spec(
    path: str | Path,
) -> NearbyBaselineSuiteSpec:
    return NearbyBaselineSuiteSpec.from_dict(json.loads(Path(path).read_text()))


def nearby_baseline_seed(root_seed: int, scenario_id: str) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{scenario_id}:nearby-baseline-v1".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def nearby_dataset_identity(
    base_dataset_identity: str,
    scenario: NearbyBaselineScenario,
) -> str:
    payload = {
        "base_dataset_identity": str(base_dataset_identity),
        "scenario_id": scenario.scenario_id,
        "root_model_hash": scenario.root_spec.model_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mutation_log_bayes_factors(
    graph: ModelGraph,
    evidences: Mapping[str, ModelEvidence],
) -> dict[str, float]:
    """Maximum evaluated edge BF for each registered mutation."""
    by_mutation: dict[str, list[float]] = {}
    for row in edge_log_bayes_factors(graph, evidences).values():
        by_mutation.setdefault(
            str(row["mutation_id"]),
            [],
        ).append(float(row["log_bayes_factor"]))
    return {
        mutation_id: float(max(values))
        for mutation_id, values in sorted(by_mutation.items())
    }


def compare_mutation_support(
    reference: Mapping[str, float],
    scenario: Mapping[str, float],
) -> dict[str, object]:
    mutations = sorted(set(reference) | set(scenario))
    rows = []
    for mutation in mutations:
        ref = reference.get(mutation)
        alt = scenario.get(mutation)
        rows.append(
            {
                "mutation_id": mutation,
                "reference_max_log_bayes_factor": ref,
                "scenario_max_log_bayes_factor": alt,
                "delta_log_bayes_factor": (
                    None
                    if ref is None or alt is None
                    else float(alt - ref)
                ),
            }
        )
    comparable = [
        row
        for row in rows
        if row["delta_log_bayes_factor"] is not None
    ]
    return {
        "n_common_mutations": len(comparable),
        "max_abs_delta_log_bayes_factor": (
            None
            if not comparable
            else float(
                max(
                    abs(row["delta_log_bayes_factor"])
                    for row in comparable
                )
            )
        ),
        "mutations": rows,
    }


def build_nearby_baseline_plan(
    *,
    campaign: ProductionCampaignConfig,
    base_dataset_identity: str,
    reference_graph: ModelGraph,
    suite: NearbyBaselineSuiteSpec,
) -> dict[str, object]:
    return {
        "format_version": "gwpop-search-nearby-baseline-plan-1.0",
        "code": _code_identity(),
        "campaign_hash": campaign.campaign_hash,
        "base_dataset_identity": str(base_dataset_identity),
        "reference_graph_root_hash": reference_graph.root_hash,
        "suite": suite.to_dict(),
    }


def _write_plan_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "existing nearby-baseline plan does not match request"
            )
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def run_nearby_baseline_suite(
    root: str | Path,
    posterior,
    selection,
    reference_graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    *,
    base_dataset_identity: str,
    suite: NearbyBaselineSuiteSpec,
    reference_state_database: str | Path | None = None,
) -> dict[str, object]:
    """Run/resume the search from explicitly declared nearby root models."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    plan = build_nearby_baseline_plan(
        campaign=campaign,
        base_dataset_identity=base_dataset_identity,
        reference_graph=reference_graph,
        suite=suite,
    )
    _write_plan_once(root / "nearby_baseline_plan.json", plan)

    reference_support = {}
    if reference_state_database is not None:
        reference_support = mutation_log_bayes_factors(
            reference_graph,
            collect_best_available_evidence(reference_state_database),
        )

    results = []
    for scenario in suite.scenarios:
        graph = enumerate_model_graph(
            scenario.root_spec,
            max_depth=scenario.max_depth,
            max_models=scenario.max_models,
        )
        scenario_root = root / scenario.scenario_id
        artifacts = scenario_root / "artifacts"
        database = scenario_root / "state.sqlite"

        evaluator = DeterministicHBIEvaluator(
            posterior,
            selection,
            config=campaign.fidelity,
            dataset_identity=nearby_dataset_identity(
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
                root_seed=nearby_baseline_seed(
                    campaign.seed_policy.root_seed,
                    scenario.scenario_id,
                ),
                scheduler=campaign.scheduler,
                stop_fidelity=suite.config.stop_fidelity,
                max_models_by_fidelity={
                    "F3": suite.config.max_f3_models,
                    "F4": suite.config.max_f4_models,
                },
                max_total_compute_cost=(
                    suite.config.max_gpu_hours_per_scenario
                ),
            ),
        )
        evidence = collect_best_available_evidence(database)
        support = mutation_log_bayes_factors(graph, evidence)
        scoring = write_scientific_scoring(
            graph,
            state_database=database,
            artifact_root=artifacts,
            model_prior=model_prior_from_config(campaign.model_prior),
        )
        comparison = (
            None
            if not reference_support
            else compare_mutation_support(reference_support, support)
        )
        row = {
            "scenario": scenario.to_dict(),
            "graph_root_hash": graph.root_hash,
            "n_graph_models": len(graph.nodes),
            "execution": execution.to_dict(),
            "n_models_with_evidence": len(evidence),
            "mutation_max_log_bayes_factors": support,
            "reference_comparison": comparison,
            "scientific_scoring": scoring,
        }
        (scenario_root / "nearby_baseline_summary.json").write_text(
            json.dumps(row, sort_keys=True, indent=2)
        )
        results.append(row)

    shifts = [
        row["reference_comparison"]["max_abs_delta_log_bayes_factor"]
        for row in results
        if (
            row["reference_comparison"] is not None
            and row["reference_comparison"][
                "max_abs_delta_log_bayes_factor"
            ]
            is not None
        )
    ]
    summary = {
        "format_version": "gwpop-search-nearby-baseline-summary-1.0",
        "n_scenarios": len(results),
        "reference_evidence_available": bool(reference_support),
        "max_abs_delta_log_bayes_factor_across_scenarios": (
            None if not shifts else float(max(shifts))
        ),
        "scenarios": results,
    }
    (root / "nearby_baseline_suite_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
