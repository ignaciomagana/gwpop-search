"""Repeated dynesty evidence per model and over finite graphs (format 2.0)."""

from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("dynesty")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from gwpop_search.data.fixtures import (  # noqa: E402
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph  # noqa: E402
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference.dynesty_backend import DynestyConfig  # noqa: E402
from gwpop_search.inference.evidence import LegacyEvidenceArtifactError  # noqa: E402
from gwpop_search.inference.evidence_campaign import (  # noqa: E402
    EVIDENCE_SUMMARY_FORMAT_VERSION,
    MODEL_EVIDENCE_FORMAT_VERSION,
    EvidenceCampaignConfig,
    build_evidence_campaign_manifest,
    build_model_evidence_manifest,
    evidence_seed,
    repeat_run_dir,
    run_model_evidence_repeats,
)
from gwpop_search.inference.label_switching import IDENTITY, parameterization_for_spec  # noqa: E402
from gwpop_search.inference.synthetic import (  # noqa: E402
    SyntheticSurveyConfig,
    generate_baseline_synthetic_dataset,
)
from gwpop_search.search import ComplexityModelPrior  # noqa: E402


def _tiny_config(**overrides):
    dynesty = DynestyConfig(
        nlive=20, bound="multi", sample="rslice", dlogz=0.5, maxiter=30, batch_size=8,
        num_posterior_samples=50,
    )
    values = dict(repeats=2, dynesty=dynesty, slices_multiplier=2)
    values.update(overrides)
    return EvidenceCampaignConfig(**values)


def test_evidence_seed_is_model_and_repeat_specific():
    h1 = "a" * 64
    h2 = "b" * 64
    seeds = {evidence_seed(7, h1, 0), evidence_seed(7, h1, 1), evidence_seed(7, h2, 0)}
    assert len(seeds) == 3
    assert evidence_seed(7, h1, 0) == evidence_seed(7, h1, 0)


def test_evidence_config_resolves_slices_per_model_and_round_trips():
    config = EvidenceCampaignConfig()
    assert config.repeats == 2
    assert config.dynesty.nlive == 1000 and config.dynesty.sample == "rslice"
    assert config.dynesty.bound == "multi" and config.dynesty.dlogz == 0.1
    assert config.dynesty_config_for(10).slices == 2 * (3 + 10)
    assert config.dynesty_config_for(13).slices == 32
    assert config.dynesty_config_for(13).nlive == 1000
    fixed = EvidenceCampaignConfig(
        dynesty=replace(config.dynesty, slices=7), slices_multiplier=None
    )
    assert fixed.dynesty_config_for(13).slices == 7
    payload = json.loads(json.dumps(config.to_dict()))
    assert EvidenceCampaignConfig.from_dict(payload) == config

    with pytest.raises(ValueError, match="not both"):
        EvidenceCampaignConfig(dynesty=replace(config.dynesty, slices=7))
    with pytest.raises(ValueError, match="slice"):
        EvidenceCampaignConfig(dynesty=DynestyConfig(sample="rwalk"))
    with pytest.raises(ValueError, match="repeats"):
        EvidenceCampaignConfig(repeats=0)
    with pytest.raises(TypeError):
        EvidenceCampaignConfig(dynesty={"nlive": 10})
    with pytest.raises(LegacyEvidenceArtifactError, match="JAXNS"):
        EvidenceCampaignConfig.from_dict(
            {"repeats": 2, "nested_sampling": {"num_live_points": 250}}
        )
    with pytest.raises(ValueError, match="unknown"):
        EvidenceCampaignConfig.from_dict({**payload, "extra": 1})


def test_evidence_campaign_manifest_records_graph_data_prior_and_backend_config():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=10)
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    config = _tiny_config(repeats=3)
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
        dataset_identity="toy-dataset-sha",
    )
    assert manifest["format_version"] == "gwpop-search-evidence-campaign-2.0"
    assert manifest["sampler_backend"]["name"] == "dynesty"
    assert manifest["graph_root_hash"] == graph.root_hash
    assert manifest["selected_model_hashes"] == selected
    assert manifest["dataset_identity"] == "toy-dataset-sha"
    assert manifest["campaign_config"] == config.to_dict()
    assert manifest["model_prior"]["parameters"]["penalty_per_axis"] == 0.5
    assert manifest["canonicalize_exchangeable_components"] is True
    with pytest.raises(ValueError, match="unknown selected model"):
        build_evidence_campaign_manifest(
            graph, posterior, selection, root_seed=1, config=config,
            model_prior=prior, model_hashes=["not-a-model"],
        )


def test_model_manifest_pins_resolved_config_seeds_priors_and_parameterization():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1)
    (mixture,) = [node for node in graph.nodes if node.chieff.family == "gaussian_mixture"]
    config = _tiny_config()
    manifest = build_model_evidence_manifest(
        mixture,
        make_toy_posterior_catalog(),
        make_toy_selection_catalog(),
        root_seed=5,
        config=config,
        hbi_config=HBIConfig(selection_chunk_size=None),
        dataset_identity="manifest-sha",
        parameterization=parameterization_for_spec(mixture),
    )
    assert manifest["format_version"] == MODEL_EVIDENCE_FORMAT_VERSION
    assert manifest["resolved_dynesty_config"]["slices"] == 2 * (3 + 13)
    assert manifest["repeat_seeds"] == [evidence_seed(5, mixture.model_hash, r) for r in (0, 1)]
    assert manifest["parameter_order"] == sorted(mixture.priors)
    assert manifest["parameterization"]["kind"] == "ordered_exchangeable_pairs"
    assert manifest["hbi_config"]["selection_chunk_size"] is None


def _tiny_survey():
    return generate_baseline_synthetic_dataset(
        seed=5,
        config=SyntheticSurveyConfig(
            n_events=6,
            posterior_samples_per_event=32,
            n_injections=3_000,
            population_batch_size=512,
            redshift_sampling_grid=1024,
        ),
    )


def test_repeats_run_resume_reuse_and_refuse_mismatches(monkeypatch, tmp_path):
    dataset = _tiny_survey()
    spec = baseline_model_spec()
    config = _tiny_config()
    with pytest.warns(UserWarning):
        results, summary = run_model_evidence_repeats(
            tmp_path,
            spec,
            dataset.posterior,
            dataset.selection,
            root_seed=91,
            config=config,
            hbi_config=HBIConfig(selection_chunk_size=None),
            dataset_identity="dataset-A",
        )
    assert [result.seed for result in results] == [
        evidence_seed(91, spec.model_hash, r) for r in range(2)
    ]
    assert all(result.config.slices == 26 for result in results)
    assert summary["format_version"] == EVIDENCE_SUMMARY_FORMAT_VERSION
    assert summary["n_repeats"] == 2
    assert summary["log_evidence_mean"] == pytest.approx(
        np.mean([result.log_evidence for result in results])
    )
    assert summary["conservative_error"] == pytest.approx(
        max(summary["repeat_std"], summary["max_reported_error"])
    )
    assert [row["termination"] for row in summary["runs"]] == ["budget", "budget"]
    for r in range(2):
        assert (repeat_run_dir(tmp_path, r) / "result.npz").exists()
    written = json.loads((tmp_path / "evidence_summary.json").read_text())
    assert written == json.loads(json.dumps(summary))
    assert json.loads((tmp_path / "model_spec.json").read_text()) == spec.to_dict()

    # Completed repeats are reused, not re-run.
    def refuse(*args, **kwargs):
        raise AssertionError("completed repeats must be reused")

    monkeypatch.setattr(
        "gwpop_search.inference.dynesty_backend.run_dynesty", refuse
    )
    again, summary_again = run_model_evidence_repeats(
        tmp_path,
        spec,
        dataset.posterior,
        dataset.selection,
        root_seed=91,
        config=config,
        hbi_config=HBIConfig(selection_chunk_size=None),
        dataset_identity="dataset-A",
    )
    assert summary_again["estimates"] == summary["estimates"]
    np.testing.assert_array_equal(again[1].samples, results[1].samples)

    for kwargs, match in (
        (dict(dataset_identity="dataset-B"), "dataset_identity"),
        (dict(root_seed=92), "root_seed"),
        (dict(config=replace(config, repeats=3)), "campaign_config"),
    ):
        call = dict(
            root_seed=91,
            config=config,
            hbi_config=HBIConfig(selection_chunk_size=None),
            dataset_identity="dataset-A",
        )
        call.update(kwargs)
        with pytest.raises(ValueError, match=match):
            run_model_evidence_repeats(
                tmp_path, spec, dataset.posterior, dataset.selection, **call
            )


def test_legacy_jaxns_evidence_directories_are_refused(tmp_path):
    spec = baseline_model_spec()
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    legacy = tmp_path / "legacy-files"
    legacy.mkdir()
    (legacy / "evidence_000.npz").write_bytes(b"old")
    with pytest.raises(LegacyEvidenceArtifactError, match="JAXNS"):
        run_model_evidence_repeats(
            legacy, spec, posterior, selection, root_seed=1, config=_tiny_config()
        )
    old_manifest = tmp_path / "legacy-manifest"
    old_manifest.mkdir()
    (old_manifest / "manifest.json").write_text(
        json.dumps({"format_version": "gwpop-search-model-evidence-1.0"})
    )
    with pytest.raises(LegacyEvidenceArtifactError, match="model-evidence-1.0"):
        run_model_evidence_repeats(
            old_manifest, spec, posterior, selection, root_seed=1, config=_tiny_config()
        )


def test_graph_campaign_scores_mean_evidence_with_conservative_errors(monkeypatch, tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=3)
    values = {node.model_hash: -10.0 - index for index, node in enumerate(graph.nodes)}
    seen = {}

    def fake_repeats(model_dir, spec, posterior, selection, **kwargs):
        seen[spec.model_hash] = kwargs["parameterization"]
        return [], {
            "log_evidence_mean": values[spec.model_hash],
            "conservative_error": 0.2,
        }

    monkeypatch.setattr(
        "gwpop_search.inference.evidence_campaign.run_model_evidence_repeats", fake_repeats
    )
    from gwpop_search.inference.evidence_campaign import run_graph_evidence_campaign

    payload = run_graph_evidence_campaign(
        tmp_path,
        graph,
        make_toy_posterior_catalog(),
        make_toy_selection_catalog(),
        root_seed=3,
        config=_tiny_config(),
        model_prior=ComplexityModelPrior(),
    )
    assert {row["model_hash"] for row in payload["scores"]} == set(values)
    assert all(row["log_evidence_error"] == 0.2 for row in payload["scores"])
    for model_hash, value in seen.items():
        family = graph.by_hash[model_hash].chieff.family
        if family == "gaussian_mixture":
            assert value.kind == "ordered_exchangeable_pairs"
        else:
            assert value is IDENTITY
    assert (tmp_path / "scored_graph.json").exists()
