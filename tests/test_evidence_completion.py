import json

import pytest

from gwpop_search.grammar import (
    baseline_model_spec,
    enumerate_model_graph,
)
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
    complete_graph_evidence,
    model_graph_hash,
)
from gwpop_search.search import (
    EvaluationRecord,
    Fidelity,
    SchedulerConfig,
    SearchBudgetExceeded,
    evaluation_run_id,
    evaluation_seed,
)
from gwpop_search.store import ResultStore


def _campaign(graph, *, max_f3_models=None, max_gpu_hours=100.0):
    return ProductionCampaignConfig(
        campaign_id="completion-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash=model_graph_hash(graph),
        model_graph_root_hash=graph.root_hash,
        git_commit="b" * 40,
        model_prior={
            "version": "axis-complexity-v1",
            "penalty_per_axis": 0.5,
        },
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=17),
        budget=SearchBudget(
            max_gpu_hours=max_gpu_hours,
            max_f3_models=(
                len(graph.nodes)
                if max_f3_models is None
                else max_f3_models
            ),
            max_f4_models=max(1, min(2, len(graph.nodes))),
            max_null_replays=10,
        ),
        artifact_root="artifacts",
        state_database="state.sqlite",
    )


def _write_evaluation_artifact(
    run_dir,
    model_hash,
    fidelity,
    *,
    logz=-10.0,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    if fidelity == Fidelity.F3_EVIDENCE:
        diagnostics = {
            "evidence": {
                "log_evidence_mean": float(logz),
                "conservative_error": 0.1,
            }
        }
    else:
        diagnostics = {
            "evidence": {
                "evidence": {
                    "log_evidence_mean": float(logz),
                    "conservative_error": 0.05,
                }
            }
        }
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "model_hash": model_hash,
                "fidelity": fidelity.value,
                "diagnostics": diagnostics,
            }
        )
    )


def _record_existing(
    store,
    model,
    campaign,
    root,
    fidelity,
    *,
    diagnostics_pass,
    logz=-10.0,
    compute_cost=0.2,
):
    store.register_model(model)
    seed = evaluation_seed(
        campaign.seed_policy.root_seed,
        model.model_hash,
        fidelity,
    )
    run_dir = root / fidelity.value / model.model_hash
    _write_evaluation_artifact(
        run_dir,
        model.model_hash,
        fidelity,
        logz=logz,
    )
    store.record_evaluation(
        evaluation_run_id(model.model_hash, fidelity, seed),
        EvaluationRecord(
            model_hash=model.model_hash,
            fidelity=fidelity,
            diagnostics_pass=diagnostics_pass,
            screen_value=logz,
            compute_cost=compute_cost,
        ),
        seed=seed,
        run_config={"test": True},
        artifact_path=str(run_dir),
    )


class _FakeEvaluator:
    calls = []
    fail_hashes = set()

    def __init__(self, posterior, selection, config, dataset_identity):
        self.dataset_identity = dataset_identity

    def evaluate(self, model, fidelity, *, seed, run_dir):
        type(self).calls.append((model.model_hash, fidelity.value, seed))
        passed = model.model_hash not in type(self).fail_hashes
        _write_evaluation_artifact(
            run_dir,
            model.model_hash,
            fidelity,
            logz=-20.0 + len(type(self).calls),
        )
        return EvaluationRecord(
            model_hash=model.model_hash,
            fidelity=fidelity,
            diagnostics_pass=passed,
            screen_value=-20.0 + len(type(self).calls),
            compute_cost=0.25,
        )


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeEvaluator.calls = []
    _FakeEvaluator.fail_hashes = set()


def test_completion_reuses_valid_existing_f3_without_rerun(
    monkeypatch,
    tmp_path,
):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    campaign = _campaign(graph)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"
    store = ResultStore(database)
    _record_existing(
        store,
        graph.nodes[0],
        campaign,
        artifacts,
        Fidelity.F3_EVIDENCE,
        diagnostics_pass=True,
    )

    monkeypatch.setattr(
        "gwpop_search.production.completion.DeterministicHBIEvaluator",
        _FakeEvaluator,
    )
    summary = complete_graph_evidence(
        graph,
        object(),
        object(),
        campaign,
        dataset_identity="dataset",
        state_database=database,
        artifact_root=artifacts,
    )

    assert _FakeEvaluator.calls == []
    assert summary["n_reused"] == 1
    assert summary["n_evaluated"] == 0
    assert summary["full_model_posterior_available"]


def test_completion_blocks_frozen_invalid_f3_instead_of_retrying(
    monkeypatch,
    tmp_path,
):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    campaign = _campaign(graph)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"
    store = ResultStore(database)
    _record_existing(
        store,
        graph.nodes[0],
        campaign,
        artifacts,
        Fidelity.F3_EVIDENCE,
        diagnostics_pass=False,
    )

    monkeypatch.setattr(
        "gwpop_search.production.completion.DeterministicHBIEvaluator",
        _FakeEvaluator,
    )
    summary = complete_graph_evidence(
        graph,
        object(),
        object(),
        campaign,
        dataset_identity="dataset",
        state_database=database,
        artifact_root=artifacts,
    )

    assert _FakeEvaluator.calls == []
    assert summary["n_blocked_invalid"] == 1
    assert not summary["full_model_posterior_available"]


def test_invalid_f4_does_not_prevent_missing_f3_fallback(
    monkeypatch,
    tmp_path,
):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    campaign = _campaign(graph)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"
    store = ResultStore(database)
    _record_existing(
        store,
        graph.nodes[0],
        campaign,
        artifacts,
        Fidelity.F4_PRODUCTION,
        diagnostics_pass=False,
    )

    monkeypatch.setattr(
        "gwpop_search.production.completion.DeterministicHBIEvaluator",
        _FakeEvaluator,
    )
    summary = complete_graph_evidence(
        graph,
        object(),
        object(),
        campaign,
        dataset_identity="dataset",
        state_database=database,
        artifact_root=artifacts,
    )

    assert len(_FakeEvaluator.calls) == 1
    assert _FakeEvaluator.calls[0][1] == "F3"
    assert summary["n_valid_evidence_final"] == 1
    assert summary["full_model_posterior_available"]


def test_completion_runs_only_missing_nodes_and_enables_full_scoring(
    monkeypatch,
    tmp_path,
):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=3,
    )
    campaign = _campaign(graph)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"
    store = ResultStore(database)
    _record_existing(
        store,
        graph.nodes[0],
        campaign,
        artifacts,
        Fidelity.F3_EVIDENCE,
        diagnostics_pass=True,
        logz=-10.0,
    )

    monkeypatch.setattr(
        "gwpop_search.production.completion.DeterministicHBIEvaluator",
        _FakeEvaluator,
    )
    summary = complete_graph_evidence(
        graph,
        object(),
        object(),
        campaign,
        dataset_identity="dataset",
        state_database=database,
        artifact_root=artifacts,
    )

    assert len(_FakeEvaluator.calls) == len(graph.nodes) - 1
    assert summary["n_valid_evidence_final"] == len(graph.nodes)
    assert summary["full_model_posterior_available"]
    assert summary["scientific_scoring"]["scored_graph"] is not None


def test_fresh_failed_f3_remains_outside_scientific_evidence(
    monkeypatch,
    tmp_path,
):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    campaign = _campaign(graph)
    _FakeEvaluator.fail_hashes = {graph.root_hash}

    monkeypatch.setattr(
        "gwpop_search.production.completion.DeterministicHBIEvaluator",
        _FakeEvaluator,
    )
    summary = complete_graph_evidence(
        graph,
        object(),
        object(),
        campaign,
        dataset_identity="dataset",
        state_database=tmp_path / "state.sqlite",
        artifact_root=tmp_path / "artifacts",
    )

    assert summary["n_evaluated"] == 1
    assert summary["n_blocked_invalid"] == 1
    assert summary["n_valid_evidence_final"] == 0
    assert not summary["full_model_posterior_available"]


def test_completion_cannot_bypass_frozen_f3_model_budget(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=3,
    )
    campaign = _campaign(graph, max_f3_models=1)

    with pytest.raises(SearchBudgetExceeded, match="declared graph has"):
        complete_graph_evidence(
            graph,
            object(),
            object(),
            campaign,
            dataset_identity="dataset",
            state_database=tmp_path / "state.sqlite",
            artifact_root=tmp_path / "artifacts",
        )
