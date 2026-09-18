import json

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.grammar import (
    baseline_model_spec,
    enumerate_model_graph,
    save_model_graph,
)
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    DatasetManifest,
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    artifact_entry_from_file,
    load_frozen_dataset,
    model_graph_hash,
    run_production_search,
)
from gwpop_search.search import SearchExecutionSummary, SchedulerConfig


def _frozen_inputs(tmp_path):
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    pe_path = tmp_path / "pe.h5"
    selection_path = tmp_path / "selection.h5"
    posterior.to_hdf5(pe_path)
    selection.to_hdf5(selection_path)

    manifest = DatasetManifest(
        dataset_id="toy-frozen",
        coordinate_basis_identity=posterior.basis.identity,
        event_names=posterior.event_names,
        event_selection={"fixture": True},
        waveform_policy={"fixture": True},
        artifacts=(
            artifact_entry_from_file(
                pe_path,
                role="pe",
                stored_path="pe.h5",
            ),
            artifact_entry_from_file(
                selection_path,
                role="selection",
                stored_path="selection.h5",
            ),
        ),
    )
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    graph_path = tmp_path / "graph.json"
    save_model_graph(graph_path, graph)
    campaign = ProductionCampaignConfig(
        campaign_id="toy-production",
        dataset_manifest_hash=manifest.manifest_hash,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="a" * 40,
        model_prior={
            "version": "axis-complexity-v1",
            "penalty_per_axis": 0.5,
        },
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(
            beam_width=1,
            exploration_quota=0,
            seed=7,
        ),
        seed_policy=SeedPolicy(root_seed=9),
        budget=SearchBudget(
            max_gpu_hours=100.0,
            max_f3_models=1,
            max_f4_models=1,
            max_null_replays=10,
        ),
        artifact_root="artifacts",
        state_database="state/state.sqlite",
        agents_enabled=False,
    )
    return manifest, graph, graph_path, campaign


def test_load_frozen_dataset_reconstructs_canonical_hdf5_pair(tmp_path):
    manifest, _, _, _ = _frozen_inputs(tmp_path)
    posterior, selection = load_frozen_dataset(
        manifest,
        data_base_dir=tmp_path,
    )
    assert posterior.event_names == manifest.event_names
    assert posterior.basis.identity == manifest.coordinate_basis_identity
    assert selection.basis.identity == manifest.coordinate_basis_identity


def test_production_runner_validates_freeze_and_wires_executor(
    monkeypatch,
    tmp_path,
):
    manifest, graph, graph_path, campaign = _frozen_inputs(tmp_path)
    seen = {}

    def fake_execute(graph_arg, evaluator, *, state_database, artifact_root, config):
        seen["root"] = graph_arg.root_hash
        seen["dataset_identity"] = evaluator.dataset_identity
        seen["state_database"] = str(state_database)
        seen["artifact_root"] = str(artifact_root)
        seen["f3_limit"] = config.max_models_by_fidelity["F3"]
        seen["f4_limit"] = config.max_models_by_fidelity["F4"]
        seen["compute_limit"] = config.max_total_compute_cost
        return SearchExecutionSummary(
            root_hash=graph_arg.root_hash,
            n_models_registered=len(graph_arg.nodes),
            evaluations_by_fidelity={},
            promoted_by_fidelity={},
            pruned_by_fidelity={},
            completed_fidelity="F0",
            total_compute_cost=0.0,
        )

    monkeypatch.setattr(
        "gwpop_search.production.runner.execute_search",
        fake_execute,
    )
    result = run_production_search(
        manifest,
        graph_path,
        campaign,
        data_base_dir=tmp_path,
        work_dir=tmp_path / "work",
        require_current_commit=False,
    )

    assert seen["root"] == graph.root_hash
    assert seen["dataset_identity"] == manifest.manifest_hash
    assert seen["f3_limit"] == 1
    assert seen["f4_limit"] == 1
    assert seen["compute_limit"] == 100.0
    assert result["scientific_scoring"]["scored_graph"] is None
    assert result["scientific_scoring"]["evidence_coverage"]["complete"] is False

    summary = (
        tmp_path
        / "work"
        / "artifacts"
        / "production_run_summary.json"
    )
    assert summary.exists()
    assert json.loads(summary.read_text())["campaign_id"] == "toy-production"


def test_production_runner_refuses_agents_enabled(tmp_path):
    manifest, _, graph_path, campaign = _frozen_inputs(tmp_path)
    payload = campaign.to_dict()
    payload["agents_enabled"] = True
    enabled = ProductionCampaignConfig.from_dict(payload)

    import pytest

    with pytest.raises(ValueError, match="agents_enabled=false"):
        run_production_search(
            manifest,
            graph_path,
            enabled,
            data_base_dir=tmp_path,
            work_dir=tmp_path / "work",
            require_current_commit=False,
        )
