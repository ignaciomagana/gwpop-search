import json

import pytest

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference.fidelity import (
    FidelityRunConfig,
    load_fidelity_run_config,
    save_fidelity_run_config,
)
from gwpop_search.production import (
    SearchBudget,
    SeedPolicy,
    build_dataset_manifest_from_canonical_files,
    build_production_campaign,
    load_production_campaign,
    model_graph_hash,
    save_production_campaign,
)
from gwpop_search.search import SchedulerConfig


def test_canonical_hdf5_pair_builds_explicit_dataset_manifest(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    pe = tmp_path / "pe.h5"
    sel = tmp_path / "selection.h5"
    posterior.to_hdf5(pe)
    selection.to_hdf5(sel)

    manifest = build_dataset_manifest_from_canonical_files(
        pe,
        sel,
        dataset_id="toy-freeze",
        event_selection={"far_per_year": 1.0},
        waveform_policy={"name": "fixture"},
        stored_pe_path="data/pe.h5",
        stored_selection_path="data/selection.h5",
        metadata={"owner": "test"},
    )

    assert manifest.coordinate_basis_identity == posterior.basis.identity
    assert manifest.event_names == posterior.event_names
    assert manifest.artifacts[0].role == "pe"
    assert manifest.artifacts[1].role == "selection"
    assert len(manifest.manifest_hash) == 64


def test_fidelity_config_file_roundtrip_is_exact(tmp_path):
    config = FidelityRunConfig(
        f0_pe_samples_per_event=17,
        f1_selected_per_campaign=1234,
    )
    path = tmp_path / "fidelity.json"
    save_fidelity_run_config(path, config)
    restored = load_fidelity_run_config(path)

    assert restored == config
    payload = json.loads(path.read_text())
    assert payload["f0_pe_samples_per_event"] == 17
    assert payload["f1_selected_per_campaign"] == 1234
    assert payload["f4_evidence"]["repeats"] == 3


def test_campaign_builder_pins_manifest_graph_and_full_fidelity(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    pe = tmp_path / "pe.h5"
    sel = tmp_path / "selection.h5"
    posterior.to_hdf5(pe)
    selection.to_hdf5(sel)
    manifest = build_dataset_manifest_from_canonical_files(
        pe,
        sel,
        dataset_id="toy-freeze",
        event_selection={"fixture": True},
        waveform_policy={"fixture": True},
    )
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=10,
    )
    fidelity = FidelityRunConfig(f0_pe_samples_per_event=19)
    campaign = build_production_campaign(
        manifest,
        graph,
        campaign_id="campaign-test",
        git_commit="a" * 40,
        model_prior={
            "version": "axis-complexity-v1",
            "penalty_per_axis": 0.7,
        },
        fidelity=fidelity,
        scheduler=SchedulerConfig(
            beam_width=4,
            exploration_quota=1,
            seed=2,
        ),
        seed_policy=SeedPolicy(root_seed=3),
        budget=SearchBudget(100.0, 10, 4, 25),
        artifact_root="runs/campaign-test",
        state_database="runs/campaign-test/state.sqlite",
    )

    assert campaign.dataset_manifest_hash == manifest.manifest_hash
    assert campaign.model_graph_hash == model_graph_hash(graph)
    assert campaign.model_graph_root_hash == graph.root_hash
    assert campaign.fidelity == fidelity
    assert campaign.format_version == "gwpop-search-production-campaign-1.1"

    path = tmp_path / "campaign.json"
    save_production_campaign(path, campaign)
    assert load_production_campaign(path) == campaign


def test_old_production_campaign_format_is_rejected(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    pe = tmp_path / "pe.h5"
    sel = tmp_path / "selection.h5"
    posterior.to_hdf5(pe)
    selection.to_hdf5(sel)
    manifest = build_dataset_manifest_from_canonical_files(
        pe,
        sel,
        dataset_id="toy-freeze",
        event_selection={"fixture": True},
        waveform_policy={"fixture": True},
    )
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    campaign = build_production_campaign(
        manifest,
        graph,
        campaign_id="campaign-test",
        git_commit="a" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=3),
        budget=SearchBudget(100.0, 1, 1, 10),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )
    payload = campaign.to_dict()
    payload["format_version"] = "gwpop-search-production-campaign-1.0"

    from gwpop_search.production import ProductionCampaignConfig

    with pytest.raises(ValueError, match="unsupported production campaign"):
        ProductionCampaignConfig.from_dict(payload)
