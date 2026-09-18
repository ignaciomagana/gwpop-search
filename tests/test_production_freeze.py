import json

import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.grammar.io import save_model_graph
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    DatasetManifest,
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    artifact_entry_from_file,
    load_dataset_manifest,
    load_production_campaign,
    model_graph_hash,
    save_dataset_manifest,
    save_production_campaign,
    validate_dataset_manifest_files,
    verify_graph_file,
)
from gwpop_search.search import ComplexityModelPrior, SchedulerConfig


def _manifest(tmp_path):
    pe = tmp_path / "pe.h5"
    selection = tmp_path / "selection.h5"
    pe.write_bytes(b"pe-data")
    selection.write_bytes(b"selection-data")

    return DatasetManifest(
        dataset_id="gwtc5-test-freeze",
        coordinate_basis_identity="gwcat_v2_chieff:test",
        event_names=("GW_A", "GW_B"),
        event_selection={"far_threshold_per_year": 1.0},
        waveform_policy={"pe_release": "test"},
        artifacts=(
            artifact_entry_from_file(pe, role="pe", stored_path="pe.h5"),
            artifact_entry_from_file(
                selection,
                role="selection",
                stored_path="selection.h5",
            ),
        ),
        metadata={"purpose": "test"},
    )


def test_dataset_manifest_roundtrip_hash_and_file_validation(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    save_dataset_manifest(path, manifest)
    restored = load_dataset_manifest(path)

    assert restored == manifest
    assert restored.manifest_hash == manifest.manifest_hash
    validation = validate_dataset_manifest_files(
        manifest,
        base_dir=tmp_path,
    )
    assert validation["valid"]
    assert all(item["valid"] for item in validation["artifacts"])

    (tmp_path / "pe.h5").write_bytes(b"tampered")
    tampered = validate_dataset_manifest_files(
        manifest,
        base_dir=tmp_path,
    )
    assert not tampered["valid"]


def test_model_graph_freeze_hash_is_canonical_and_file_inspectable(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    path = tmp_path / "graph.json"
    save_model_graph(path, graph)

    inspected = verify_graph_file(path)
    assert inspected["graph_hash"] == model_graph_hash(graph)
    assert inspected["root_hash"] == graph.root_hash
    assert inspected["n_nodes"] == len(graph.nodes)
    assert inspected["n_edges"] == len(graph.edges)


def test_production_campaign_roundtrip_freezes_search_contract(tmp_path):
    manifest = _manifest(tmp_path)
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    prior = ComplexityModelPrior(penalty_per_axis=0.5)

    config = ProductionCampaignConfig(
        campaign_id="gwtc5-bbh-search-v1",
        dataset_manifest_hash=manifest.manifest_hash,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="a" * 40,
        model_prior={
            "version": prior.version,
            "penalty_per_axis": prior.penalty_per_axis,
        },
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(
            beam_width=8,
            exploration_quota=2,
            seed=19,
        ),
        seed_policy=SeedPolicy(root_seed=123),
        budget=SearchBudget(
            max_gpu_hours=1000.0,
            max_f3_models=20,
            max_f4_models=8,
            max_null_replays=200,
        ),
        artifact_root="runs/production-v1",
        state_database="runs/production-v1/state.sqlite",
        agents_enabled=False,
    )
    path = tmp_path / "campaign.json"
    save_production_campaign(path, config)
    restored = load_production_campaign(path)

    assert restored == config
    assert restored.campaign_hash == config.campaign_hash
    assert restored.agents_enabled is False
    saved = json.loads(path.read_text())
    assert saved["budget"]["max_f4_models"] == 8
    assert saved["fidelity"]["f4_nuts"]["num_chains"] == 4
    assert saved["fidelity"]["f4_evidence"]["repeats"] == 3


def test_production_campaign_rejects_unknown_code_revision():
    with pytest.raises(ValueError, match="concrete git commit"):
        ProductionCampaignConfig(
            campaign_id="bad",
            dataset_manifest_hash="a" * 64,
            model_graph_hash="b" * 64,
            model_graph_root_hash="c" * 64,
            git_commit="unknown",
            model_prior={"version": "x"},
            fidelity=FidelityRunConfig(),
            scheduler=SchedulerConfig(),
            seed_policy=SeedPolicy(root_seed=1),
            budget=SearchBudget(10.0, 2, 1, 10),
            artifact_root="runs",
            state_database="state.sqlite",
        )
