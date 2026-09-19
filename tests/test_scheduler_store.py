import sqlite3

import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.search import (
    DEFAULT_LADDER,
    EvaluationRecord,
    Fidelity,
    SchedulerConfig,
    decide_promotions,
    next_in_ladder,
    validate_ladder,
)
from gwpop_search.store import ResultStore


def _records(n=8, *, fidelity=Fidelity.F3_EVIDENCE):
    return [
        EvaluationRecord(
            model_hash=f"{index:064x}",
            fidelity=fidelity,
            diagnostics_pass=True,
            screen_value=float(n - index),
            compute_cost=0.1 * index,
        )
        for index in range(n)
    ]


def test_scheduler_is_deterministic_under_input_reordering():
    config = SchedulerConfig(beam_width=3, exploration_quota=2, seed=11)
    records = _records(8)

    a = decide_promotions(records, config=config)
    b = decide_promotions(reversed(records), config=config)

    assert [item.to_dict() for item in a] == [item.to_dict() for item in b]
    promoted = [item for item in a if item.decision == "promote"]
    assert len(promoted) == 5
    assert sum(item.reason == "beam" for item in promoted) == 3
    assert sum(item.reason == "exploration_quota" for item in promoted) == 2


def test_diagnostics_veto_even_high_screen_value():
    records = _records(4)
    bad = EvaluationRecord(
        model_hash=records[0].model_hash,
        fidelity=Fidelity.F3_EVIDENCE,
        diagnostics_pass=False,
        screen_value=1e6,
    )
    records[0] = bad

    decisions = decide_promotions(
        records,
        config=SchedulerConfig(beam_width=2, exploration_quota=0),
    )
    by_hash = {item.model_hash: item for item in decisions}
    assert by_hash[bad.model_hash].decision == "prune"
    assert by_hash[bad.model_hash].reason == "diagnostic_veto"


def test_f0_advances_all_valid_models_to_f3_and_f4_is_terminal():
    f0 = _records(5, fidelity=Fidelity.F0_SANITY)
    decisions = decide_promotions(f0)
    assert all(item.decision == "promote" for item in decisions)
    assert all(item.to_fidelity is Fidelity.F3_EVIDENCE for item in decisions)
    assert all(item.scheduler_version == "deterministic-beam-v2" for item in decisions)

    f3 = _records(12, fidelity=Fidelity.F3_EVIDENCE)
    beam = decide_promotions(f3, config=SchedulerConfig(beam_width=3, exploration_quota=1))
    promoted = [item for item in beam if item.decision == "promote"]
    assert len(promoted) == 4
    assert all(item.to_fidelity is Fidelity.F4_PRODUCTION for item in promoted)

    f4 = _records(3, fidelity=Fidelity.F4_PRODUCTION)
    terminal = decide_promotions(f4)
    assert all(item.decision == "complete" for item in terminal)
    assert all(item.to_fidelity is None for item in terminal)


def test_scheduler_rejects_mixed_fidelity_cohort():
    records = [
        EvaluationRecord("a" * 64, Fidelity.F3_EVIDENCE, True, 1.0),
        EvaluationRecord("b" * 64, Fidelity.F4_PRODUCTION, True, 2.0),
    ]
    with pytest.raises(ValueError, match="one fidelity"):
        decide_promotions(records)


def test_ladder_v2_skips_the_nuts_era_rungs():
    assert DEFAULT_LADDER == ("F0", "F3", "F4")
    config = SchedulerConfig()
    assert config.ladder == DEFAULT_LADDER
    assert config.version == "deterministic-beam-v2"
    assert next_in_ladder(Fidelity.F0_SANITY, config.ladder) is Fidelity.F3_EVIDENCE
    assert next_in_ladder(Fidelity.F3_EVIDENCE, config.ladder) is Fidelity.F4_PRODUCTION
    assert next_in_ladder(Fidelity.F4_PRODUCTION, config.ladder) is None
    # F1/F2 are never scheduled by the v2 ladder; the enum order still lists them.
    assert Fidelity.F0_SANITY.next() is Fidelity.F1_SCREEN
    with pytest.raises(ValueError, match="not a rung"):
        decide_promotions([EvaluationRecord("a" * 64, Fidelity.F1_SCREEN, True, 1.0)])
    # A shorter ladder (e.g. the F3-completion null replays stop at F3).
    short = SchedulerConfig(ladder=("F0", "F3"))
    terminal = decide_promotions(_records(3, fidelity=Fidelity.F3_EVIDENCE), config=short)
    assert all(item.decision == "complete" for item in terminal)
    # JSON round trip through a list keeps the tuple.
    assert SchedulerConfig(ladder=["F0", "F3", "F4"]) == config


@pytest.mark.parametrize(
    "ladder, match",
    [
        ((), "empty"),
        (("F3", "F4"), "start with F0"),
        (("F0", "F4", "F3"), "strictly increase"),
        (("F0", "F1"), "evidence rung"),
        (("F0", "F9"), "F9"),
    ],
)
def test_invalid_ladders_are_rejected(ladder, match):
    with pytest.raises(ValueError, match=match):
        validate_ladder(ladder)


def test_nuts_era_scheduler_version_is_refused():
    with pytest.raises(ValueError, match="re-freeze"):
        SchedulerConfig(version="deterministic-beam-v1")


def test_sqlite_store_is_append_only_and_promotion_history_is_queryable(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=5,
    )
    store = ResultStore(tmp_path / "state.sqlite")
    for model in graph.nodes:
        store.register_model(model)

    root = graph.nodes[0]
    record = EvaluationRecord(
        model_hash=root.model_hash,
        fidelity=Fidelity.F3_EVIDENCE,
        diagnostics_pass=True,
        screen_value=2.0,
        compute_cost=0.25,
    )
    store.record_evaluation(
        "run-001",
        record,
        seed=7,
        run_config={"samples": 100},
        artifact_path="runs/run-001",
    )

    rows = store.evaluations(model_hash=root.model_hash)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-001"
    assert rows[0]["fidelity"] == "F3"
    assert rows[0]["screen_value"] == 2.0

    with pytest.raises(sqlite3.IntegrityError):
        store.record_evaluation(
            "run-001",
            record,
            seed=7,
            run_config={"samples": 100},
        )

    decisions = decide_promotions(
        [record],
        config=SchedulerConfig(beam_width=1, exploration_quota=0),
    )
    store.record_promotions(decisions)
    history = store.promotion_history()
    assert len(history) == 1
    assert history[0]["model_hash"] == root.model_hash
    assert history[0]["decision"] == "promote"
    assert history[0]["to_fidelity"] == "F4"


def test_store_model_registration_is_idempotent(tmp_path):
    model = baseline_model_spec()
    store = ResultStore(tmp_path / "state.sqlite")
    store.register_model(model)
    store.register_model(model)
