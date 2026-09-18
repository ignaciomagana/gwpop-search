import json

import numpy as np
import pytest

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference.evidence import (
    EvidenceResult,
    NestedSamplingConfig,
)
from gwpop_search.inference.evidence_campaign import (
    EvidenceCampaignConfig,
    build_evidence_campaign_manifest,
    evidence_seed,
    run_model_evidence_repeats,
)
from gwpop_search.search import ComplexityModelPrior


def _fake_evidence(seed, config, logz):
    return EvidenceResult(
        log_evidence=logz,
        log_evidence_error=0.1,
        posterior_samples={"alpha": np.asarray([3.0, 3.1])},
        diagnostics={"nested_ess": 100.0},
        seed=seed,
        backend="numpyro-jaxns",
        config=config,
    )


def test_evidence_seed_is_model_and_repeat_specific():
    h1 = "a" * 64
    h2 = "b" * 64
    seeds = {
        evidence_seed(7, h1, 0),
        evidence_seed(7, h1, 1),
        evidence_seed(7, h2, 0),
    }
    assert len(seeds) == 3
    assert evidence_seed(7, h1, 0) == evidence_seed(7, h1, 0)


def test_evidence_campaign_manifest_records_graph_data_prior_and_backend_config():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=10,
    )
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    config = EvidenceCampaignConfig(
        repeats=3,
        nested_sampling=NestedSamplingConfig(
            num_live_points=100,
            max_samples=5000,
            dlogz=0.05,
            num_posterior_samples=300,
        ),
    )
    prior = ComplexityModelPrior(penalty_per_axis=0.5)
    selected = [model.model_hash for model in graph.nodes[:3]]

    manifest = build_evidence_campaign_manifest(
        graph,
        posterior,
        selection,
        root_seed=44,
        config=config,
        model_prior=prior,
        model_hashes=selected,
        dataset_label="toy",
    )

    assert manifest["graph_root_hash"] == graph.root_hash
    assert manifest["selected_model_hashes"] == selected
    assert manifest["dataset_label"] == "toy"
    assert manifest["campaign_config"]["repeats"] == 3
    assert manifest["model_prior"]["version"] == prior.version
    assert manifest["model_prior"]["parameters"]["penalty_per_axis"] == 0.5
    assert manifest["pe_basis"] == posterior.basis.identity


def test_completed_evidence_repeats_are_reused(monkeypatch, tmp_path):
    spec = baseline_model_spec()
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    config = EvidenceCampaignConfig(
        repeats=2,
        nested_sampling=NestedSamplingConfig(
            num_live_points=50,
            max_samples=1000,
            dlogz=0.1,
            num_posterior_samples=20,
        ),
    )
    calls = []

    def fake_run(*args, seed, config, **kwargs):
        calls.append(seed)
        return _fake_evidence(seed, config, -10.0 + 0.01 * len(calls))

    monkeypatch.setattr(
        "gwpop_search.inference.evidence_campaign.run_hbi_evidence",
        fake_run,
    )
    _, summary1 = run_model_evidence_repeats(
        tmp_path,
        spec,
        posterior,
        selection,
        root_seed=91,
        config=config,
    )
    assert len(calls) == 2
    assert summary1["n_repeats"] == 2

    calls.clear()
    _, summary2 = run_model_evidence_repeats(
        tmp_path,
        spec,
        posterior,
        selection,
        root_seed=91,
        config=config,
    )
    assert calls == []
    assert summary2 == summary1
    assert json.loads((tmp_path / "model_spec.json").read_text()) == spec.to_dict()


def test_evidence_campaign_rejects_unknown_selected_model():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=10,
    )
    with pytest.raises(ValueError, match="unknown selected model"):
        build_evidence_campaign_manifest(
            graph,
            make_toy_posterior_catalog(),
            make_toy_selection_catalog(),
            root_seed=1,
            config=EvidenceCampaignConfig(),
            model_prior=ComplexityModelPrior(),
            model_hashes=["not-a-model"],
        )
