import json

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.search import (
    ModelEvidence,
    SchedulerConfig,
)
from gwpop_search.validation import (
    EventDropScenario,
    EventStressConfig,
    EventStressSuiteSpec,
    build_event_stress_plan,
    compare_edge_bayes_factors,
    edge_log_bayes_factors,
    leave_one_out_scenarios,
    load_event_stress_suite_spec,
    save_event_stress_suite_spec,
    stress_dataset_identity,
    stress_seed,
)


def _campaign():
    return ProductionCampaignConfig(
        campaign_id="stress-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash="b" * 64,
        model_graph_root_hash="c" * 64,
        git_commit="d" * 40,
        model_prior={
            "version": "axis-complexity-v1",
            "penalty_per_axis": 0.5,
        },
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(
            beam_width=4,
            exploration_quota=1,
            seed=7,
        ),
        seed_policy=SeedPolicy(root_seed=8),
        budget=SearchBudget(100.0, 10, 4, 20),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )


def test_leave_one_out_scenarios_are_explicit_and_deterministic():
    scenarios = leave_one_out_scenarios(("GW_A", "GW_B"))
    assert [item.scenario_id for item in scenarios] == [
        "loo_000_GW_A",
        "loo_001_GW_B",
    ]
    assert scenarios[0].drop_events == ("GW_A",)
    assert scenarios[0].category == "leave_one_out"


def test_stress_seed_and_dataset_identity_change_with_scenario():
    a = EventDropScenario("a", ("GW_A",), category="custom")
    b = EventDropScenario("b", ("GW_A",), category="custom")

    assert stress_seed(3, "a") == stress_seed(3, "a")
    assert stress_seed(3, "a") != stress_seed(3, "b")
    assert stress_dataset_identity("dataset", a) != stress_dataset_identity(
        "dataset",
        b,
    )


def test_edge_bayes_factor_comparison_uses_within_dataset_differences():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=4,
    )
    edge = graph.edges[0]
    reference = {
        edge.parent_hash: ModelEvidence(edge.parent_hash, -100.0, 0.1),
        edge.child_hash: ModelEvidence(edge.child_hash, -96.0, 0.1),
    }
    stressed = {
        edge.parent_hash: ModelEvidence(edge.parent_hash, -80.0, 0.1),
        edge.child_hash: ModelEvidence(edge.child_hash, -78.0, 0.1),
    }

    ref_edges = edge_log_bayes_factors(graph, reference)
    stress_edges = edge_log_bayes_factors(graph, stressed)
    comparison = compare_edge_bayes_factors(ref_edges, stress_edges)

    assert comparison["n_common_evidence_edges"] == 1
    row = comparison["edges"][0]
    assert row["reference_log_bayes_factor"] == 4.0
    assert row["stress_log_bayes_factor"] == 2.0
    assert row["delta_log_bayes_factor"] == -2.0
    assert comparison["max_abs_delta_log_bayes_factor"] == 2.0


def test_event_stress_plan_freezes_scenarios_and_limits():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=5,
    )
    scenarios = (
        EventDropScenario(
            "drop_A",
            ("GW_A",),
            category="loud_event",
            note="declared externally",
        ),
    )
    plan = build_event_stress_plan(
        campaign=_campaign(),
        base_dataset_identity="dataset-sha",
        graph=graph,
        scenarios=scenarios,
        config=EventStressConfig(
            max_gpu_hours_per_scenario=12.0,
            max_f3_models=5,
            max_f4_models=2,
        ),
    )

    assert plan["base_dataset_identity"] == "dataset-sha"
    assert plan["scenarios"][0]["scenario_id"] == "drop_A"
    assert plan["scenarios"][0]["category"] == "loud_event"
    assert plan["config"]["max_gpu_hours_per_scenario"] == 12.0


def test_event_drop_scenario_rejects_unsafe_id():
    import pytest

    with pytest.raises(ValueError, match="scenario_id"):
        EventDropScenario("../bad", ("GW_A",))



def test_event_stress_suite_spec_roundtrip(tmp_path):
    spec = EventStressSuiteSpec(
        scenarios=(
            EventDropScenario(
                "drop_A",
                ("GW_A",),
                category="loud_event",
            ),
        ),
        config=EventStressConfig(
            max_gpu_hours_per_scenario=11.0,
            max_f3_models=4,
            max_f4_models=2,
        ),
    )
    path = tmp_path / "stress.json"
    save_event_stress_suite_spec(path, spec)
    restored = load_event_stress_suite_spec(path)
    assert restored == spec
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-event-stress-suite-1.0"
    assert payload["config"]["stop_fidelity"] == "F3"
