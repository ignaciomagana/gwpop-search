import json

import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.nulls import (
    ExactNullCampaignConfig,
    build_exact_null_campaign_plan,
    load_exact_null_campaign_config,
    null_replay_seed,
    null_search_seed,
    save_exact_null_campaign_config,
)
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.search import Fidelity, SchedulerConfig


def _campaign(max_nulls=20):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=5,
    )
    campaign = ProductionCampaignConfig(
        campaign_id="null-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash="b" * 64,
        model_graph_root_hash=graph.root_hash,
        git_commit="d" * 40,
        model_prior={
            "version": "axis-complexity-v1",
            "penalty_per_axis": 0.5,
        },
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=7),
        budget=SearchBudget(100.0, 10, 4, max_nulls),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )
    return graph, campaign


def test_exact_null_campaign_config_roundtrip(tmp_path):
    config = ExactNullCampaignConfig(
        n_nulls=12,
        root_seed=9,
        survey=SyntheticSurveyConfig(
            n_events=20,
            posterior_samples_per_event=64,
            n_injections=2000,
        ),
        stop_fidelity=Fidelity.F3_EVIDENCE,
        max_gpu_hours_per_null=20.0,
        max_f3_models=5,
        max_f4_models=2,
    )
    path = tmp_path / "nulls.json"
    save_exact_null_campaign_config(path, config)
    restored = load_exact_null_campaign_config(path)
    assert restored == config
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-exact-null-campaign-1.0"
    assert payload["stop_fidelity"] == "F3"
    assert payload["truth_hyperparameters"]["mmin"] == config.truth_hyperparameters["mmin"]


def test_null_data_and_search_seeds_are_independent_and_deterministic():
    assert null_replay_seed(5, 0) == null_replay_seed(5, 0)
    assert null_search_seed(5, 0) == null_search_seed(5, 0)
    assert null_replay_seed(5, 0) != null_search_seed(5, 0)
    assert null_search_seed(5, 0) != null_search_seed(5, 1)


def test_exact_null_plan_enforces_frozen_null_budget():
    graph, campaign = _campaign(max_nulls=3)
    with pytest.raises(ValueError, match="exceeds frozen campaign budget"):
        build_exact_null_campaign_plan(
            graph,
            campaign,
            ExactNullCampaignConfig(n_nulls=4),
        )


def test_exact_null_plan_pins_search_and_seed_policy():
    graph, campaign = _campaign(max_nulls=10)
    config = ExactNullCampaignConfig(
        n_nulls=3,
        root_seed=123,
        stop_fidelity=Fidelity.F3_EVIDENCE,
    )
    plan = build_exact_null_campaign_plan(graph, campaign, config)

    assert plan["graph_root_hash"] == graph.root_hash
    assert plan["null_config"]["n_nulls"] == 3
    assert len(plan["seed_policy"]) == 3
    assert plan["seed_policy"][0]["data_seed"] != plan["seed_policy"][0]["search_seed"]



def test_exact_null_config_rejects_missing_truth_parameter():
    config = ExactNullCampaignConfig()
    truth = dict(config.truth_hyperparameters)
    truth.pop("mmin")
    with pytest.raises(ValueError, match="missing"):
        ExactNullCampaignConfig(truth_hyperparameters=truth)
