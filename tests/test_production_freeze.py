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
    assert saved["format_version"] == "gwpop-search-production-campaign-2.0"
    assert "f4_nuts" not in saved["fidelity"]
    assert saved["fidelity"]["format_version"] == "gwpop-search-fidelity-config-2.0"
    assert saved["fidelity"]["f3_evidence"]["repeats"] == 2
    assert saved["fidelity"]["f4_evidence"]["repeats"] == 3
    assert saved["fidelity"]["f4_evidence"]["dynesty"]["sample"] == "rslice"
    assert saved["scheduler"]["ladder"] == ["F0", "F3", "F4"]
    assert saved["scheduler"]["version"] == "deterministic-beam-v2"
    assert saved["sampler_backend"] == {"name": "dynesty", "version": "3.1.0"}


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


def test_nuts_jaxns_era_campaigns_are_refused(tmp_path):
    from gwpop_search.production import LegacyCampaignError, ProductionCampaignConfig

    manifest = _manifest(tmp_path)
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    config = ProductionCampaignConfig(
        campaign_id="v2",
        dataset_manifest_hash=manifest.manifest_hash,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="a" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=1),
        budget=SearchBudget(10.0, 2, 1, 10),
        artifact_root="runs",
        state_database="state.sqlite",
    )
    payload = config.to_dict()
    for legacy in ("gwpop-search-production-campaign-1.1", None):
        old = dict(payload)
        if legacy is None:
            old.pop("format_version")
        else:
            old["format_version"] = legacy
        with pytest.raises(LegacyCampaignError, match="NUTS/JAXNS-era campaign; re-freeze"):
            ProductionCampaignConfig.from_dict(old)
    with pytest.raises(ValueError, match="supported pin"):
        ProductionCampaignConfig.from_dict(
            {**payload, "sampler_backend": {"name": "dynesty", "version": "3.0.0"}}
        )


def test_production_freeze_validation_checks_the_dynesty_pin(tmp_path, monkeypatch):
    from gwpop_search.production import (
        ProductionCampaignConfig,
        validate_production_freeze,
    )

    manifest = _manifest(tmp_path)
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    graph_path = tmp_path / "graph.json"
    save_model_graph(graph_path, graph)
    campaign = ProductionCampaignConfig(
        campaign_id="pin",
        dataset_manifest_hash=manifest.manifest_hash,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="a" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=1),
        budget=SearchBudget(10.0, 2, 1, 10),
        artifact_root="runs",
        state_database="state.sqlite",
    )
    result = validate_production_freeze(
        manifest, graph_path, campaign, data_base_dir=tmp_path, require_current_commit=False
    )
    assert result["checks"]["sampler_backend_pin_matches"]
    assert result["sampler_backend"]["installed_version"] == "3.1.0"
    assert result["valid"]

    import gwpop_search.production.validate as validate_module

    monkeypatch.setattr(
        validate_module,
        "_installed_version",
        lambda package: "3.2.0" if package == "dynesty" else "0.0",
    )
    mismatch = validate_production_freeze(
        manifest, graph_path, campaign, data_base_dir=tmp_path, require_current_commit=False
    )
    assert not mismatch["checks"]["sampler_backend_pin_matches"]
    assert not mismatch["valid"]


def test_production_freeze_names_and_requires_a_registered_root_profile(tmp_path):
    from dataclasses import replace as dataclass_replace

    from gwpop_search.grammar import PriorConfig
    from gwpop_search.production import (
        ProductionCampaignConfig,
        validate_production_freeze,
    )

    manifest = _manifest(tmp_path)
    graph = enumerate_model_graph(
        baseline_model_spec("gwtc5-v1"), max_depth=0, max_models=1
    )
    graph_path = tmp_path / "graph.json"
    save_model_graph(graph_path, graph)
    campaign = ProductionCampaignConfig(
        campaign_id="profile",
        dataset_manifest_hash=manifest.manifest_hash,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="a" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=1),
        budget=SearchBudget(10.0, 2, 1, 10),
        artifact_root="runs",
        state_database="state.sqlite",
    )
    result = validate_production_freeze(
        manifest, graph_path, campaign, data_base_dir=tmp_path, require_current_commit=False
    )
    assert result["root_hyperprior_profile"] == "gwtc5-v1"
    assert result["checks"]["model_graph_root_is_registered_baseline"]
    assert result["valid"]

    root = baseline_model_spec()
    drifted = dataclass_replace(
        root,
        priors={**root.priors, "mmax": PriorConfig("uniform", {"low": 60.0, "high": 90.0})},
    )
    drifted_graph = enumerate_model_graph(drifted, max_depth=0, max_models=1)
    drifted_path = tmp_path / "drifted.json"
    save_model_graph(drifted_path, drifted_graph)
    drifted_campaign = dataclass_replace(
        campaign,
        model_graph_hash=model_graph_hash(drifted_graph),
        model_graph_root_hash=drifted_graph.root_hash,
    )
    drifted_result = validate_production_freeze(
        manifest,
        drifted_path,
        drifted_campaign,
        data_base_dir=tmp_path,
        require_current_commit=False,
    )
    assert drifted_result["root_hyperprior_profile"] is None
    assert not drifted_result["checks"]["model_graph_root_is_registered_baseline"]
    assert not drifted_result["valid"]


def test_build_production_campaign_can_require_the_gwtc5_root_profile(tmp_path):
    from gwpop_search.production import build_production_campaign

    manifest = _manifest(tmp_path)
    kwargs = dict(
        campaign_id="require-profile",
        git_commit="a" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=1),
        budget=SearchBudget(10.0, 2, 1, 10),
        artifact_root="runs",
        state_database="state.sqlite",
    )
    gwtc5 = enumerate_model_graph(
        baseline_model_spec("gwtc5-v1"), max_depth=0, max_models=1
    )
    frozen = build_production_campaign(
        manifest, gwtc5, require_root_profile="gwtc5-v1", **kwargs
    )
    assert frozen.model_graph_root_hash == gwtc5.root_hash

    phase3 = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    with pytest.raises(ValueError, match="not the required 'gwtc5-v1'"):
        build_production_campaign(
            manifest, phase3, require_root_profile="gwtc5-v1", **kwargs
        )
