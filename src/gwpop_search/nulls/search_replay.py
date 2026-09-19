"""Bridge the declared baseline null to the exact deterministic search pipeline.

One replay generates a baseline-null catalog, runs the frozen search with the
production evaluator (dynesty ladder) up to the statistic's stop rung, runs
full-graph F3 evidence completion and computes the calibrated statistic from
exactly the evidence the statistic declares (``f3_completion``: F3 only).

The null is the *graph's own root*: a replay is admissible for any registered
root hyperprior profile (``phase3``, ``gwtc5-v1``, ...) whose baseline model
hash equals ``graph.root_hash``. The profile is resolved from the graph, never
assumed, so a production graph enumerated under ``gwtc5-v1`` draws its null
truths and its frozen-selection resampling weights from the same root priors
the search scores.
"""

from __future__ import annotations

from pathlib import Path

from gwpop_search.grammar import (
    HYPERPRIOR_PROFILES,
    ModelGraph,
    ModelSpec,
    baseline_hyperprior_profile,
    baseline_model_spec,
)
from gwpop_search.inference.fidelity import (
    DeterministicHBIEvaluator,
    FidelityRunConfig,
)
from gwpop_search.inference.model_spec import prior_specs_from_model_spec
from gwpop_search.inference.synthetic import (
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
    require_population_proxy_coverage,
)
from gwpop_search.production.completion import complete_graph_evidence
from gwpop_search.production.runner import collect_best_available_evidence
from gwpop_search.search import (
    ModelEvidence,
    ModelPrior,
    SearchExecutionConfig,
    execute_search,
)

from .frozen_selection import (
    generate_frozen_selection_null_dataset,
    require_frozen_selection_survey,
)
from .replay import SearchReplayResult

_STATISTIC_EVIDENCE_FIDELITIES = {"f3_completion": ("F3",)}


def declared_null_root(graph: ModelGraph) -> tuple[ModelSpec, str]:
    """Return the graph root spec and its registered hyperprior profile.

    The baseline null is defined by the declarative baseline under *some*
    registered hyperprior profile. Any profile is admissible as long as its
    baseline model hash is the graph root; this keeps the null aligned with the
    graph that is actually searched instead of pinning it to one profile.

    Raises ``ValueError`` when the graph root is not a registered baseline,
    because then the null catalogs would be drawn from a different model than
    the search's null hypothesis.
    """
    profile = baseline_hyperprior_profile(graph.by_hash[graph.root_hash])
    if profile is None:
        raise ValueError(
            "baseline null replay requires the declarative baseline to be the "
            "model-graph root under a registered hyperprior profile "
            f"(registered={list(HYPERPRIOR_PROFILES)}); graph root "
            f"{graph.root_hash} matches none of them"
        )
    return baseline_model_spec(profile), profile


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
    statistic: str = "f3_completion",
) -> SearchReplayResult:
    """Generate one baseline-null catalog and replay the frozen search procedure.

    ``execution_config`` fixes the rungs (``f3_completion``: stop at F3) and
    ``completion_campaign`` the evidence completion; the statistic is computed
    from the evidence fidelities the statistic declares.
    """
    if statistic not in _STATISTIC_EVIDENCE_FIDELITIES:
        raise ValueError(f"unsupported null statistic {statistic!r}")
    declared_root, root_profile = declared_null_root(graph)

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
            "survey_config": survey_config.to_dict(),
            "root_hyperprior_profile": root_profile,
        }
        # Every graph node's selection integral reads these injections; check
        # coverage for all of them before any compute is spent.
        for node in graph.nodes:
            require_population_proxy_coverage(
                null_selection,
                priors=prior_specs_from_model_spec(node),
                context=f"null replay model {node.model_hash}",
            )
        # Survey-v2 options change the data drawn for the same seed, so they
        # enter the identity; legacy configurations keep the original identity.
        dataset_identity = (
            f"baseline-null:{int(seed)}{survey_config.dataset_identity_suffix()}"
        )
    elif data_mode == "frozen_selection_resample":
        require_frozen_selection_survey(survey_config)
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
        null_data_metadata["root_hyperprior_profile"] = root_profile
        dataset_identity = (
            f"frozen-selection-null:{production_dataset_identity}:{int(seed)}"
            f"{survey_config.dataset_identity_suffix()}"
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

    evidences = collect_best_available_evidence(
        database, fidelities=_STATISTIC_EVIDENCE_FIDELITIES[statistic]
    )
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
            "statistic": statistic,
            "statistic_evidence_fidelities": list(_STATISTIC_EVIDENCE_FIDELITIES[statistic]),
            "n_evidence_edges": int(stats["n_evidence_edges"]),
            "execution": execution.to_dict(),
            "evidence_completion": completion,
            "null_data_mode": data_mode,
            "null_data_metadata": null_data_metadata,
            "survey_config": survey_config.to_dict(),
            "truth_hyperparameters": null_truth_hyperparameters,
        },
    )
