from pathlib import Path

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.search.executor import (
    SearchBudgetExceeded,
    SearchExecutionConfig,
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
    a = evaluation_seed(1, model_hash, Fidelity.F1_SCREEN)
    b = evaluation_seed(1, model_hash, Fidelity.F1_SCREEN)
    c = evaluation_seed(1, model_hash, Fidelity.F2_INFERENCE)
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
        stop_fidelity=Fidelity.F2_INFERENCE,
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

    assert first.completed_fidelity == "F2"
    assert first_call_count == len(first_evaluations)
    assert first.promoted_by_fidelity["F0"] == len(graph.nodes)
    assert first.promoted_by_fidelity["F1"] == 2

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

    original = EvaluationRecord(
        model_hash=model.model_hash,
        fidelity=Fidelity.F1_SCREEN,
        diagnostics_pass=True,
        screen_value=1.0,
    )
    from gwpop_search.search.scheduler import PromotionDecision

    first = PromotionDecision(
        model_hash=model.model_hash,
        from_fidelity=Fidelity.F1_SCREEN,
        to_fidelity=Fidelity.F2_INFERENCE,
        decision="promote",
        reason="beam",
        scheduler_version="v1",
    )
    store.record_promotions([first])
    store.record_promotions([first])

    conflict = PromotionDecision(
        model_hash=model.model_hash,
        from_fidelity=Fidelity.F1_SCREEN,
        to_fidelity=None,
        decision="prune",
        reason="changed",
        scheduler_version="v1",
    )
    import pytest

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
        stop_fidelity=Fidelity.F2_INFERENCE,
        max_models_by_fidelity={"F1": 2},
    )

    import pytest

    with pytest.raises(SearchBudgetExceeded, match="F1 has"):
        execute_search(
            graph,
            evaluator,
            state_database=tmp_path / "state.sqlite",
            artifact_root=tmp_path / "artifacts",
            config=config,
        )

    # F0 finished durably; the over-budget F1 cohort never launched.
    store = ResultStore(tmp_path / "state.sqlite")
    assert len(store.evaluations(fidelity="F0")) == len(graph.nodes)
    assert store.evaluations(fidelity="F1") == []


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

    import pytest

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
