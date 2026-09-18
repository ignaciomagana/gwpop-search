"""Frozen multi-null replay campaigns using the exact deterministic search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path

from gwpop_search.grammar import ModelGraph, baseline_model_spec
from gwpop_search.inference.numpyro import _code_identity
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS
from gwpop_search.production import ProductionCampaignConfig
from gwpop_search.production.freeze import model_graph_hash
from gwpop_search.production.runner import (
    collect_best_available_evidence,
    model_prior_from_config,
)
from gwpop_search.search import Fidelity, SearchExecutionConfig

from .replay import (
    SearchReplayResult,
    calibrate_search_replays,
    null_replay_seed,
    run_null_replay_campaign,
)
from .search_replay import (
    run_baseline_null_search_replay,
    search_statistics_from_evidence,
)


@dataclass(frozen=True)
class ExactNullCampaignConfig:
    n_nulls: int = 100
    root_seed: int = 20260918
    survey: SyntheticSurveyConfig = SyntheticSurveyConfig()
    truth_hyperparameters: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_BASELINE_HYPERPARAMETERS)
    )
    format_version: str = "gwpop-search-exact-null-campaign-1.1"

    def __post_init__(self) -> None:
        if self.format_version != "gwpop-search-exact-null-campaign-1.1":
            raise ValueError("unsupported exact null campaign format")
        if self.n_nulls <= 0:
            raise ValueError("n_nulls must be positive")
        truth = {
            str(name): float(value)
            for name, value in self.truth_hyperparameters.items()
        }
        missing = set(DEFAULT_BASELINE_HYPERPARAMETERS) - set(truth)
        if missing:
            raise ValueError(
                f"null truth hyperparameters missing {sorted(missing)}"
            )
        object.__setattr__(self, "truth_hyperparameters", truth)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "n_nulls": int(self.n_nulls),
            "root_seed": int(self.root_seed),
            "survey": asdict(self.survey),
            "truth_hyperparameters": dict(self.truth_hyperparameters),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "ExactNullCampaignConfig":
        return cls(
            n_nulls=int(payload["n_nulls"]),
            root_seed=int(payload["root_seed"]),
            survey=SyntheticSurveyConfig(**dict(payload["survey"])),
            truth_hyperparameters={
                str(name): float(value)
                for name, value in dict(
                    payload["truth_hyperparameters"]
                ).items()
            },
            format_version=str(
                payload.get(
                    "format_version",
                    "gwpop-search-exact-null-campaign-1.1",
                )
            ),
        )


def save_exact_null_campaign_config(
    path: str | Path,
    config: ExactNullCampaignConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2))


def load_exact_null_campaign_config(
    path: str | Path,
) -> ExactNullCampaignConfig:
    return ExactNullCampaignConfig.from_dict(json.loads(Path(path).read_text()))


def null_search_seed(root_seed: int, null_index: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:null-search:{int(null_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_exact_null_campaign_plan(
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
) -> dict[str, object]:
    if graph.root_hash != baseline_model_spec().model_hash:
        raise ValueError(
            "exact baseline-null campaign requires the declared baseline root"
        )
    if config.n_nulls > campaign.budget.max_null_replays:
        raise ValueError(
            f"requested {config.n_nulls} nulls exceeds frozen campaign budget "
            f"{campaign.budget.max_null_replays}"
        )
    if len(graph.nodes) > campaign.budget.max_f3_models:
        raise ValueError(
            "exact null calibration requires the production campaign to permit "
            "full-graph F3 evidence completion"
        )
    return {
        "format_version": "gwpop-search-exact-null-plan-1.0",
        "code": _code_identity(),
        "production_campaign_hash": campaign.campaign_hash,
        "graph_hash": model_graph_hash(graph),
        "graph_root_hash": graph.root_hash,
        "null_config": config.to_dict(),
        "replayed_production_search": {
            "stop_fidelity": Fidelity.F4_PRODUCTION.value,
            "scheduler": asdict(campaign.scheduler),
            "max_gpu_hours": campaign.budget.max_gpu_hours,
            "max_f3_models": campaign.budget.max_f3_models,
            "max_f4_models": campaign.budget.max_f4_models,
            "evidence_completion_required": True,
        },
        "seed_policy": [
            {
                "null_index": index,
                "data_seed": null_replay_seed(config.root_seed, index),
                "search_seed": null_search_seed(config.root_seed, index),
            }
            for index in range(config.n_nulls)
        ],
    }


def _write_plan_once(path: Path, plan: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "existing exact null campaign plan does not match request"
            )
    else:
        path.write_text(json.dumps(plan, sort_keys=True, indent=2))


def _load_replay_results(
    root: Path,
    n_nulls: int,
) -> tuple[SearchReplayResult, ...]:
    results = []
    for index in range(n_nulls):
        payload = json.loads(
            (root / f"null_{index:05d}.json").read_text()
        )
        results.append(SearchReplayResult(**payload))
    return tuple(results)


def run_exact_null_campaign(
    root: str | Path,
    graph: ModelGraph,
    campaign: ProductionCampaignConfig,
    config: ExactNullCampaignConfig,
    *,
    observed_state_database: str | Path | None = None,
) -> dict[str, object]:
    """Run/resume baseline-null catalogs through the same deterministic search."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    plan = build_exact_null_campaign_plan(graph, campaign, config)
    _write_plan_once(root / "null_campaign_plan.json", plan)
    model_prior = model_prior_from_config(campaign.model_prior)

    def replay(index: int, data_seed: int) -> SearchReplayResult:
        return run_baseline_null_search_replay(
            index,
            data_seed,
            root=root,
            graph=graph,
            model_prior=model_prior,
            execution_config=SearchExecutionConfig(
                root_seed=null_search_seed(config.root_seed, index),
                scheduler=campaign.scheduler,
                stop_fidelity=Fidelity.F4_PRODUCTION,
                max_models_by_fidelity={
                    "F3": campaign.budget.max_f3_models,
                    "F4": campaign.budget.max_f4_models,
                },
                max_total_compute_cost=campaign.budget.max_gpu_hours,
            ),
            fidelity_config=campaign.fidelity,
            survey_config=config.survey,
            truth_hyperparameters=config.truth_hyperparameters,
            completion_campaign=campaign,
            completion_seed_root=null_search_seed(config.root_seed, index),
        )

    run_null_replay_campaign(
        root,
        n_nulls=config.n_nulls,
        root_seed=config.root_seed,
        replay=replay,
    )
    results = _load_replay_results(root, config.n_nulls)

    observed = None
    if observed_state_database is not None:
        evidence = collect_best_available_evidence(observed_state_database)
        if evidence:
            if len(evidence) != len(graph.nodes):
                raise ValueError(
                    "observed production state lacks complete valid evidence; "
                    "run complete-production-evidence before null calibration"
                )
            observed = search_statistics_from_evidence(
                graph,
                evidence,
                model_prior=model_prior,
            )

    calibrated = calibrate_search_replays(
        results,
        observed_max_log_bayes_factor=(
            None if observed is None else observed["max_log_bayes_factor"]
        ),
        observed_max_log_posterior_odds=(
            None
            if observed is None
            else observed["max_log_posterior_odds"]
        ),
    )
    summary = {
        "format_version": "gwpop-search-exact-null-summary-1.0",
        "production_campaign_hash": campaign.campaign_hash,
        "observed_search_statistics": observed,
        "calibration": calibrated,
    }
    (root / "exact_null_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
