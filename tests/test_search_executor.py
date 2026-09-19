import json

import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.search.executor import (
    LegacyStateError,
    SearchBudgetExceeded,
    SearchExecutionConfig,
    evaluation_run_id,
    evaluation_seed,
    execute_search,
)
from gwpop_search.search.scheduler import (
    EvaluationRecord,
    Fidelity,
    SchedulerConfig,
)
from gwpop_search.store import ResultStore


class StubEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, model, fidelity, *, seed, run_dir):
        self.calls.append((model.model_hash, fidelity.value, seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "done.txt").write_text("done")
        score = float(int(model.model_hash[:8], 16) % 1000)
        return EvaluationRecord(
            model_hash=model.model_hash,
            fidelity=fidelity,
            diagnostics_pass=True,
            screen_value=score,
            compute_cost=0.01,
        )


def test_evaluation_seed_is_deterministic_and_fidelity_specific():
    model_hash = baseline_model_spec().model_hash
    a = evaluation_seed(1, model_hash, Fidelity.F3_EVIDENCE)
    b = evaluation_seed(1, model_hash, Fidelity.F3_EVIDENCE)
    c = evaluation_seed(1, model_hash, Fidelity.F4_PRODUCTION)
    assert a == b
    assert a != c


def test_search_executor_resumes_without_duplicate_evaluations_or_promotions(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=6,
    )
    config = SearchExecutionConfig(
        root_seed=123,
        scheduler=SchedulerConfig(
            beam_width=2,
            exploration_quota=0,
            seed=456,
        ),
        start_fidelity=Fidelity.F0_SANITY,
        stop_fidelity=Fidelity.F4_PRODUCTION,
    )
    evaluator = StubEvaluator()
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"

    first = execute_search(
        graph,
        evaluator,
        state_database=database,
        artifact_root=artifacts,
        config=config,
    )
    first_call_count = len(evaluator.calls)
    store = ResultStore(database)
    first_evaluations = store.evaluations()
    first_history = store.promotion_history()

    assert first.completed_fidelity == "F4"
    assert first_call_count == len(first_evaluations)
    # The ladder skips F1/F2: F0 -> F3 (every pass) -> F4 (beam).
    assert [call[1] for call in evaluator.calls].count("F1") == 0
    assert [call[1] for call in evaluator.calls].count("F2") == 0
    assert first.evaluations_by_fidelity == {
        "F0": len(graph.nodes),
        "F3": len(graph.nodes),
        "F4": 2,
    }
    assert first.promoted_by_fidelity["F0"] == len(graph.nodes)
    assert first.promoted_by_fidelity["F3"] == 2
    payload = first.to_dict()
    assert payload["format_version"] == "gwpop-search-execution-summary-1.1"
    assert payload["ladder"] == ["F0", "F3", "F4"]
    assert payload["sampler_backend"] == "dynesty"
    rows = ResultStore(database).evaluations()
    assert {json.loads(row["run_config_json"])["executor"] for row in rows} == {
        "deterministic-fidelity-v2"
    }
    assert {json.loads(row["run_config_json"])["backend"] for row in rows} == {"dynesty"}

    evaluator.calls.clear()
    second = execute_search(
        graph,
        evaluator,
        state_database=database,
        artifact_root=artifacts,
        config=config,
    )
    second_evaluations = store.evaluations()
    second_history = store.promotion_history()

    assert evaluator.calls == []
    assert second.to_dict() == first.to_dict()
    assert second_evaluations == first_evaluations
    assert second_history == first_history


def test_promotion_replay_conflict_is_rejected(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    store = ResultStore(tmp_path / "state.sqlite")
    model = graph.nodes[0]
    store.register_model(model)

    from gwpop_search.search.scheduler import PromotionDecision

    first = PromotionDecision(
        model_hash=model.model_hash,
        from_fidelity=Fidelity.F3_EVIDENCE,
        to_fidelity=Fidelity.F4_PRODUCTION,
        decision="promote",
        reason="beam",
        scheduler_version="v2",
    )
    store.record_promotions([first])
    store.record_promotions([first])

    conflict = PromotionDecision(
        model_hash=model.model_hash,
        from_fidelity=Fidelity.F3_EVIDENCE,
        to_fidelity=None,
        decision="prune",
        reason="changed",
        scheduler_version="v2",
    )

    with pytest.raises(ValueError, match="conflicts"):
        store.record_promotions([conflict])



def test_search_executor_enforces_fidelity_model_budget(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=4,
    )
    evaluator = StubEvaluator()
    config = SearchExecutionConfig(
        scheduler=SchedulerConfig(
            beam_width=4,
            exploration_quota=0,
        ),
        stop_fidelity=Fidelity.F4_PRODUCTION,
        max_models_by_fidelity={"F3": 2},
    )

    with pytest.raises(SearchBudgetExceeded, match="F3 has"):
        execute_search(
            graph,
            evaluator,
            state_database=tmp_path / "state.sqlite",
            artifact_root=tmp_path / "artifacts",
            config=config,
        )

    # F0 finished durably; the over-budget F3 cohort never launched.
    store = ResultStore(tmp_path / "state.sqlite")
    assert len(store.evaluations(fidelity="F0")) == len(graph.nodes)
    assert store.evaluations(fidelity="F3") == []


def test_search_executor_enforces_durable_compute_budget(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )
    evaluator = StubEvaluator()
    config = SearchExecutionConfig(
        stop_fidelity=Fidelity.F0_SANITY,
        max_total_compute_cost=0.005,
    )

    with pytest.raises(SearchBudgetExceeded, match="exceeded"):
        execute_search(
            graph,
            evaluator,
            state_database=tmp_path / "state.sqlite",
            artifact_root=tmp_path / "artifacts",
            config=config,
        )

    # The completed over-budget evaluation is retained rather than lost.
    store = ResultStore(tmp_path / "state.sqlite")
    rows = store.evaluations()
    assert len(rows) == 1
    assert rows[0]["compute_cost"] == 0.01


def test_execution_config_requires_ladder_rungs():
    with pytest.raises(ValueError, match="not a rung"):
        SearchExecutionConfig(stop_fidelity=Fidelity.F2_INFERENCE)
    with pytest.raises(ValueError, match="not a rung"):
        SearchExecutionConfig(start_fidelity=Fidelity.F1_SCREEN)
    short = SearchExecutionConfig(
        scheduler=SchedulerConfig(ladder=("F0", "F3")),
        stop_fidelity=Fidelity.F3_EVIDENCE,
    )
    assert short.stop_fidelity is Fidelity.F3_EVIDENCE
    with pytest.raises(ValueError, match="not a rung"):
        SearchExecutionConfig(scheduler=SchedulerConfig(ladder=("F0", "F3")))


def test_executor_refuses_nuts_era_state_rows(tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    model = graph.nodes[0]
    store = ResultStore(tmp_path / "state.sqlite")
    store.register_model(model)
    seed = evaluation_seed(20260917, model.model_hash, Fidelity.F0_SANITY)
    # A 1.x row has the same run id and seed as the v2 F0 evaluation would.
    store.record_evaluation(
        evaluation_run_id(model.model_hash, Fidelity.F0_SANITY, seed),
        EvaluationRecord(model.model_hash, Fidelity.F0_SANITY, True, 0.0, 0.01),
        seed=seed,
        run_config={"executor": "deterministic-fidelity-v1", "fidelity": "F0"},
        artifact_path=str(tmp_path / "old"),
    )
    evaluator = StubEvaluator()
    with pytest.raises(LegacyStateError, match="retired executor"):
        execute_search(
            graph,
            evaluator,
            state_database=tmp_path / "state.sqlite",
            artifact_root=tmp_path / "artifacts",
            config=SearchExecutionConfig(stop_fidelity=Fidelity.F0_SANITY),
        )
    assert evaluator.calls == []


def test_executor_checks_the_ladder_against_the_evaluator(tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)

    class F0Only(StubEvaluator):
        supported_fidelities = ("F0",)

    with pytest.raises(ValueError, match="does not support ladder rung"):
        execute_search(
            graph,
            F0Only(),
            state_database=tmp_path / "state.sqlite",
            artifact_root=tmp_path / "artifacts",
        )


class _ConfiguredEvaluator(StubEvaluator):
    """Evaluator that carries the hash of a frozen numerical configuration."""

    def __init__(self, fidelity_config_sha256):
        super().__init__()
        self.fidelity_config_sha256 = fidelity_config_sha256


def _single_rung_config():
    return SearchExecutionConfig(
        root_seed=11,
        scheduler=SchedulerConfig(ladder=("F0", "F3"), beam_width=1, exploration_quota=0),
        start_fidelity=Fidelity.F0_SANITY,
        stop_fidelity=Fidelity.F0_SANITY,
    )


def test_stored_rows_record_the_frozen_fidelity_config_hash(tmp_path):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    evaluator = _ConfiguredEvaluator("a" * 64)
    database = tmp_path / "state.sqlite"

    execute_search(
        graph,
        evaluator,
        state_database=database,
        artifact_root=tmp_path / "artifacts",
        config=_single_rung_config(),
    )
    rows = ResultStore(database).evaluations()
    assert len(rows) == 1
    assert json.loads(rows[0]["run_config_json"])["fidelity_config_sha256"] == "a" * 64


def test_re_freezing_the_numerics_refuses_to_reuse_stale_evaluations(tmp_path):
    """Seeds do not depend on the fidelity config, so reuse must check it.

    Re-freezing the numerics against the same state database used to be
    silently accepted: the stored evaluation of the previous configuration was
    reused and the new configuration never ran.
    """
    from gwpop_search.search import FidelityConfigMismatchError

    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"

    first = _ConfiguredEvaluator("a" * 64)
    execute_search(
        graph,
        first,
        state_database=database,
        artifact_root=artifacts,
        config=_single_rung_config(),
    )
    assert len(first.calls) == 1

    reused = _ConfiguredEvaluator("a" * 64)
    execute_search(
        graph,
        reused,
        state_database=database,
        artifact_root=artifacts,
        config=_single_rung_config(),
    )
    assert reused.calls == []

    refrozen = _ConfiguredEvaluator("c" * 64)
    with pytest.raises(FidelityConfigMismatchError, match="fidelity config"):
        execute_search(
            graph,
            refrozen,
            state_database=database,
            artifact_root=artifacts,
            config=_single_rung_config(),
        )
    assert refrozen.calls == []


def test_rows_without_a_config_hash_are_not_reused_by_a_configured_run(tmp_path):
    from gwpop_search.search import FidelityConfigMismatchError

    graph = enumerate_model_graph(baseline_model_spec(), max_depth=0, max_models=1)
    database = tmp_path / "state.sqlite"
    artifacts = tmp_path / "artifacts"

    # A stub evaluator declares no configuration, so its rows carry none.
    execute_search(
        graph,
        StubEvaluator(),
        state_database=database,
        artifact_root=artifacts,
        config=_single_rung_config(),
    )
    with pytest.raises(FidelityConfigMismatchError):
        execute_search(
            graph,
            _ConfiguredEvaluator("a" * 64),
            state_database=database,
            artifact_root=artifacts,
            config=_single_rung_config(),
        )
