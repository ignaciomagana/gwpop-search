import json

from gwpop_search.grammar import (
    DEFAULT_MUTATIONS,
    apply_mutation,
    baseline_model_spec,
    enumerate_model_graph,
)
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.search import ModelEvidence, SchedulerConfig
from gwpop_search.validation import (
    NearbyBaselineConfig,
    NearbyBaselineScenario,
    NearbyBaselineSuiteSpec,
    compare_mutation_support,
    load_nearby_baseline_suite_spec,
    mutation_log_bayes_factors,
    nearby_baseline_seed,
    nearby_dataset_identity,
    save_nearby_baseline_suite_spec,
)


def _mut(mutation_id):
    return next(
        item for item in DEFAULT_MUTATIONS
        if item.mutation_id == mutation_id
    )


def _campaign():
    return ProductionCampaignConfig(
        campaign_id="baseline-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash="b" * 64,
        model_graph_root_hash="c" * 64,
        git_commit="d" * 40,
        model_prior={
            "version": "axis-complexity-v1",
            "penalty_per_axis": 0.5,
        },
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=7),
        budget=SearchBudget(100.0, 10, 4, 20),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )


def test_nearby_baseline_suite_spec_roundtrip(tmp_path):
    root = apply_mutation(
        baseline_model_spec(),
        _mut("mass.family.broken_powerlaw"),
    )
    spec = NearbyBaselineSuiteSpec(
        scenarios=(
            NearbyBaselineScenario(
                "broken_mass_root",
                root,
                max_depth=1,
                max_models=12,
                note="nearby mass baseline",
            ),
        ),
        config=NearbyBaselineConfig(
            max_gpu_hours_per_scenario=20.0,
            max_f3_models=8,
            max_f4_models=3,
        ),
    )
    path = tmp_path / "nearby.json"
    save_nearby_baseline_suite_spec(path, spec)
    restored = load_nearby_baseline_suite_spec(path)

    assert restored == spec
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-nearby-baseline-suite-1.0"
    assert payload["scenarios"][0]["root_spec"]["blocks"]["mass"]["family"] == "broken_powerlaw"


def test_nearby_baseline_seed_and_dataset_identity_are_scenario_specific():
    a = NearbyBaselineScenario("a", baseline_model_spec())
    b = NearbyBaselineScenario(
        "b",
        apply_mutation(
            baseline_model_spec(),
            _mut("mass.family.powerlaw"),
        ),
    )
    assert nearby_baseline_seed(1, "a") == nearby_baseline_seed(1, "a")
    assert nearby_baseline_seed(1, "a") != nearby_baseline_seed(1, "b")
    assert nearby_dataset_identity("data", a) != nearby_dataset_identity(
        "data",
        b,
    )


def test_mutation_support_compares_edge_bfs_not_absolute_evidence():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=6,
    )
    edge = graph.edges[0]
    reference = {
        edge.parent_hash: ModelEvidence(edge.parent_hash, -100.0, 0.1),
        edge.child_hash: ModelEvidence(edge.child_hash, -95.0, 0.1),
    }
    alternate = {
        edge.parent_hash: ModelEvidence(edge.parent_hash, -30.0, 0.1),
        edge.child_hash: ModelEvidence(edge.child_hash, -28.0, 0.1),
    }
    ref = mutation_log_bayes_factors(graph, reference)
    alt = mutation_log_bayes_factors(graph, alternate)
    comparison = compare_mutation_support(ref, alt)

    mutation = edge.mutation_id
    assert ref[mutation] == 5.0
    assert alt[mutation] == 2.0
    row = next(
        item for item in comparison["mutations"]
        if item["mutation_id"] == mutation
    )
    assert row["delta_log_bayes_factor"] == -3.0
    assert comparison["max_abs_delta_log_bayes_factor"] == 3.0


def test_nearby_baseline_scenario_rejects_unsafe_id():
    import pytest

    with pytest.raises(ValueError, match="scenario_id"):
        NearbyBaselineScenario("../bad", baseline_model_spec())
