import math

import numpy as np
import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.search import (
    ComplexityModelPrior,
    ModelEvidence,
    UniformModelPrior,
    posterior_mass_for_structure_axis,
    score_model_graph,
)


def _evidence_for_graph(graph):
    return {
        model.model_hash: ModelEvidence(
            model_hash=model.model_hash,
            log_evidence=-0.4 * index,
            log_evidence_error=0.1 + 0.01 * index,
        )
        for index, model in enumerate(graph.nodes)
    }


def test_scored_model_probabilities_normalize_and_edge_bfs_are_local():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    evidences = _evidence_for_graph(graph)
    scored = score_model_graph(
        graph,
        evidences,
        model_prior=UniformModelPrior(),
    )

    assert np.isclose(
        sum(item.posterior_probability for item in scored.scores),
        1.0,
    )
    by_hash = scored.by_hash
    for edge in scored.edge_comparisons:
        parent = by_hash[edge.parent_hash]
        child = by_hash[edge.child_hash]
        assert np.isclose(
            edge.log_bayes_factor,
            child.log_evidence - parent.log_evidence,
        )
        assert np.isclose(
            edge.log_bayes_factor_error,
            math.hypot(
                child.log_evidence_error,
                parent.log_evidence_error,
            ),
        )


def test_complexity_prior_penalizes_structural_axes_not_sampler_settings():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    prior = ComplexityModelPrior(penalty_per_axis=math.log(3.0))
    root = graph.by_hash[graph.root_hash]

    assert prior.log_prior(root, root=root) == 0.0
    child = next(model for model in graph.nodes if model.model_hash != root.model_hash)
    assert np.isclose(
        prior.log_prior(child, root=root),
        -math.log(3.0),
    )


def test_structure_axis_posterior_mass_matches_direct_sum():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    scored = score_model_graph(graph, _evidence_for_graph(graph))
    mass = posterior_mass_for_structure_axis(
        graph,
        scored,
        "chieff.options.width_dependence",
    )

    direct = 0.0
    for item in scored.scores:
        model = graph.by_hash[item.model_hash]
        if model.chieff.options.get("width_dependence") != "constant":
            direct += item.posterior_probability
    assert np.isclose(mass, direct)


def test_scoring_rejects_unknown_evidence_model():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    with pytest.raises(ValueError, match="unknown model"):
        score_model_graph(
            graph,
            {"not-a-hash": ModelEvidence("not-a-hash", 0.0, 0.1)},
        )
