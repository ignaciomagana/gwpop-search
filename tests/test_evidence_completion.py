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
from gwpop_search.production import collect_best_available_evidence
from gwpop_search.search import (
    EvaluationRecord,
    Fidelity,
    LegacyStateError,
    SchedulerConfig,
    SearchBudgetExceeded,
    evaluation_run_config,
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
    passed=True,
    failure=None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "passed": bool(passed),
        "checks": [],
        "failure": failure,
        "evidence": {
            "log_evidence_mean": float(logz),
            "conservative_error": 0.1 if fidelity == Fidelity.F3_EVIDENCE else 0.05,
        },
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "format_version": "gwpop-search-fidelity-evaluation-2.0",
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
        passed=diagnostics_pass,
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
        run_config=evaluation_run_config(fidelity),
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
            passed=passed,
            failure=None if passed else {"type": "no_finite_support", "message": "test"},
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
    assert summary["blocked_invalid"][0]["failure"]["type"] == "no_finite_support"
    assert summary["n_valid_evidence_final"] == 0
    assert not summary["full_model_posterior_available"]
    assert summary["format_version"] == "gwpop-search-evidence-completion-1.1"
    rows = ResultStore(tmp_path / "state.sqlite").evaluations()
    assert [json.loads(row["run_config_json"])["executor"] for row in rows] == [
        "evidence-completion-v2"
    ]


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


def test_collection_prefers_f4_filters_f3_only_and_refuses_legacy_rows(tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=2)
    campaign = _campaign(graph)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"
    store = ResultStore(database)
    first, second = graph.nodes
    _record_existing(store, first, campaign, artifacts, Fidelity.F3_EVIDENCE,
                     diagnostics_pass=True, logz=-10.0)
    _record_existing(store, first, campaign, artifacts, Fidelity.F4_PRODUCTION,
                     diagnostics_pass=True, logz=-9.5)
    _record_existing(store, second, campaign, artifacts, Fidelity.F3_EVIDENCE,
                     diagnostics_pass=True, logz=-11.0)
    _record_existing(store, second, campaign, artifacts, Fidelity.F4_PRODUCTION,
                     diagnostics_pass=False, logz=-1.0)

    best = collect_best_available_evidence(database)
    assert best[first.model_hash].log_evidence == -9.5  # passed F4 preferred
    assert best[second.model_hash].log_evidence == -11.0  # failed F4 falls back to F3
    f3_only = collect_best_available_evidence(database, fidelities=("F3",))
    assert f3_only[first.model_hash].log_evidence == -10.0
    with pytest.raises(ValueError, match="subset"):
        collect_best_available_evidence(database, fidelities=("F2",))

    # A NUTS/JAXNS-era evaluation artifact behind a valid row is refused.
    legacy_dir = artifacts / "F3" / first.model_hash
    payload = json.loads((legacy_dir / "evaluation.json").read_text())
    payload["format_version"] = "gwpop-search-fidelity-evaluation-1.0"
    (legacy_dir / "evaluation.json").write_text(json.dumps(payload))
    with pytest.raises(LegacyStateError, match="evaluation-1.0"):
        collect_best_available_evidence(database, fidelities=("F3",))

    # So is a state database with rows of a retired executor.
    old_db = tmp_path / "old.sqlite"
    old_store = ResultStore(old_db)
    old_store.register_model(first)
    old_store.record_evaluation(
        "F3-old",
        EvaluationRecord(first.model_hash, Fidelity.F3_EVIDENCE, True, -10.0, 0.1),
        seed=1,
        run_config={"executor": "evidence-completion-v1", "fidelity": "F3"},
        artifact_path=str(tmp_path),
    )
    with pytest.raises(LegacyStateError, match="retired executor"):
        collect_best_available_evidence(old_db)
    with pytest.raises(LegacyStateError, match="retired executor"):
        complete_graph_evidence(
            graph, object(), object(), campaign, dataset_identity="dataset",
            state_database=old_db, artifact_root=tmp_path / "x",
        )
