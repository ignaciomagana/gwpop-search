"""Bridge the declared baseline null to the exact deterministic search pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gwpop_search.grammar import ModelGraph, baseline_model_spec
from gwpop_search.inference.fidelity import (
    DeterministicHBIEvaluator,
    FidelityRunConfig,
)
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.production.completion import complete_graph_evidence
from gwpop_search.production.runner import collect_best_available_evidence
from gwpop_search.search import (
    ModelEvidence,
    ModelPrior,
    SearchExecutionConfig,
    execute_search,
)

from .frozen_selection import generate_frozen_selection_null_dataset
from .replay import SearchReplayResult


def search_statistics_from_evidence(
    graph: ModelGraph,
    evidences: dict[str, ModelEvidence],
    *,
    model_prior: ModelPrior,
) -> dict[str, object]:
    """Compute the maximum comparison actually encountered by the search."""
    if not evidences:
        raise ValueError("search replay produced no evidence estimates")

    root = graph.by_hash[graph.root_hash]
    scores = {
        model_hash: (
            evidence.log_evidence
            + model_prior.log_prior(graph.by_hash[model_hash], root=root)
        )
        for model_hash, evidence in evidences.items()
    }
    best_model_hash = max(scores, key=scores.get)

    log_bfs = []
    log_posterior_odds = []
    for edge in graph.edges:
        if edge.parent_hash not in evidences or edge.child_hash not in evidences:
            continue
        parent = evidences[edge.parent_hash]
        child = evidences[edge.child_hash]
        log_bf = child.log_evidence - parent.log_evidence
        prior_odds = (
            model_prior.log_prior(graph.by_hash[edge.child_hash], root=root)
            - model_prior.log_prior(graph.by_hash[edge.parent_hash], root=root)
        )
        log_bfs.append(float(log_bf))
        log_posterior_odds.append(float(log_bf + prior_odds))

    # A search that reaches evidence for only the root has encountered no
    # improvement over the null. Its maximum improvement statistic is zero.
    return {
        "max_log_bayes_factor": max([0.0, *log_bfs]),
        "max_log_posterior_odds": max([0.0, *log_posterior_odds]),
        "n_models_evaluated": len(evidences),
        "best_model_hash": best_model_hash,
        "n_evidence_edges": len(log_bfs),
    }


def run_baseline_null_search_replay(
    null_index: int,
    seed: int,
    *,
    root: str | Path,
    graph: ModelGraph,
    model_prior: ModelPrior,
    execution_config: SearchExecutionConfig,
    fidelity_config: FidelityRunConfig,
    survey_config: SyntheticSurveyConfig | None = None,
    truth_hyperparameters=None,
    completion_campaign=None,
    completion_seed_root: int | None = None,
    data_mode: str = "synthetic_survey",
    observed_posterior=None,
    frozen_selection=None,
    production_dataset_identity: str | None = None,
    min_resampling_ess: float = 200.0,
) -> SearchReplayResult:
    """Generate one baseline-null catalog and execute the same F0--F4 search."""
    declared_root = baseline_model_spec()
    if graph.root_hash != declared_root.model_hash:
        raise ValueError(
            "baseline null replay currently requires the declarative baseline "
            "to be the model-graph root"
        )

    survey_config = (
        SyntheticSurveyConfig()
        if survey_config is None
        else survey_config
    )
    if data_mode == "synthetic_survey":
        dataset = generate_baseline_synthetic_dataset(
            seed=int(seed),
            config=survey_config,
            hyperparameters=truth_hyperparameters,
        )
        null_posterior = dataset.posterior
        null_selection = dataset.selection
        null_truth_hyperparameters = dict(dataset.truth_hyperparameters)
        null_data_metadata = {
            "mode": data_mode,
            "survey_config": asdict(survey_config),
        }
        dataset_identity = f"baseline-null:{int(seed)}"
    elif data_mode == "frozen_selection_resample":
        if observed_posterior is None or frozen_selection is None:
            raise ValueError(
                "frozen_selection_resample requires observed_posterior and "
                "frozen_selection"
            )
        if not production_dataset_identity:
            raise ValueError(
                "frozen_selection_resample requires production_dataset_identity"
            )
        dataset = generate_frozen_selection_null_dataset(
            seed=int(seed),
            observed_posterior=observed_posterior,
            selection=frozen_selection,
            truth_hyperparameters=truth_hyperparameters,
            survey_config=survey_config,
            min_resampling_ess=min_resampling_ess,
            model_spec=declared_root,
        )
        null_posterior = dataset.posterior
        null_selection = dataset.selection
        null_truth_hyperparameters = dict(dataset.truth_hyperparameters)
        null_data_metadata = dict(dataset.metadata)
        dataset_identity = (
            f"frozen-selection-null:{production_dataset_identity}:{int(seed)}"
        )
    else:
        raise ValueError(f"unsupported null data mode {data_mode!r}")

    root = Path(root)
    replay_root = root / "searches" / f"null_{int(null_index):05d}"
    artifacts = replay_root / "artifacts"
    database = replay_root / "state.sqlite"

    evaluator = DeterministicHBIEvaluator(
        null_posterior,
        null_selection,
        config=fidelity_config,
        dataset_identity=dataset_identity,
    )
    execution = execute_search(
        graph,
        evaluator,
        state_database=database,
        artifact_root=artifacts,
        config=execution_config,
    )
    completion = None
    if completion_campaign is not None:
        completion = complete_graph_evidence(
            graph,
            null_posterior,
            null_selection,
            completion_campaign,
            dataset_identity=dataset_identity,
            state_database=database,
            artifact_root=artifacts,
            root_seed=completion_seed_root,
        )

    evidences = collect_best_available_evidence(database)
    if completion_campaign is not None and len(evidences) != len(graph.nodes):
        raise RuntimeError(
            "exact null replay did not obtain valid evidence for every "
            "declared graph node"
        )
    stats = search_statistics_from_evidence(
        graph,
        evidences,
        model_prior=model_prior,
    )
    return SearchReplayResult(
        null_index=int(null_index),
        seed=int(seed),
        max_log_bayes_factor=float(stats["max_log_bayes_factor"]),
        max_log_posterior_odds=float(stats["max_log_posterior_odds"]),
        n_models_evaluated=int(stats["n_models_evaluated"]),
        best_model_hash=str(stats["best_model_hash"]),
        metadata={
            "null_model_hash": graph.root_hash,
            "n_evidence_edges": int(stats["n_evidence_edges"]),
            "execution": execution.to_dict(),
            "evidence_completion": completion,
            "null_data_mode": data_mode,
            "null_data_metadata": null_data_metadata,
            "survey_config": asdict(survey_config),
            "truth_hyperparameters": null_truth_hyperparameters,
        },
    )
