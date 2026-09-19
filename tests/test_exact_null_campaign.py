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
from gwpop_search.search import SchedulerConfig


def _campaign(max_nulls=20, max_f3_models=10):
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
        budget=SearchBudget(100.0, max_f3_models, 4, max_nulls),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )
    return graph, campaign


def _noisy_survey(**kwargs):
    return SyntheticSurveyConfig(observation_model="noisy_observation", **kwargs)


def test_exact_null_campaign_config_roundtrip(tmp_path):
    config = ExactNullCampaignConfig(
        n_nulls=12,
        root_seed=9,
        survey=_noisy_survey(
            n_events=20,
            posterior_samples_per_event=64,
            n_injections=2000,
        ),
        max_gpu_hours_per_null=7.5,
    )
    path = tmp_path / "nulls.json"
    save_exact_null_campaign_config(path, config)
    restored = load_exact_null_campaign_config(path)
    assert restored == config
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-exact-null-campaign-1.4"
    assert payload["data_mode"] == "frozen_selection_resample"
    assert payload["min_resampling_ess"] == 200.0
    assert payload["max_gpu_hours_per_null"] == 7.5
    assert payload["statistic"] == "f3_completion"
    assert payload["survey"]["observation_model"] == "noisy_observation"
    assert "stop_fidelity" not in payload
    assert payload["truth_hyperparameters"]["mmin"] == config.truth_hyperparameters["mmin"]


def test_exact_null_config_refuses_truth_centred_pe_and_unknown_statistics():
    with pytest.raises(ValueError, match="noisy_observation"):
        ExactNullCampaignConfig(survey=SyntheticSurveyConfig())
    with pytest.raises(ValueError, match="noisy_observation"):
        ExactNullCampaignConfig(survey=SyntheticSurveyConfig(), data_mode="synthetic_survey")
    with pytest.raises(ValueError, match="unsupported null statistic"):
        ExactNullCampaignConfig(statistic="full_ladder")
    # The default survey is the noisy observation model; the ceiling is unset.
    default = ExactNullCampaignConfig()
    assert default.survey.observation_model == "noisy_observation"
    assert default.max_gpu_hours_per_null is None
    with pytest.raises(ValueError, match="must be frozen explicitly"):
        default.require_compute_ceiling()


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
            ExactNullCampaignConfig(n_nulls=4, max_gpu_hours_per_null=2.0),
        )


def test_exact_null_plan_requires_an_explicit_per_null_ceiling():
    graph, campaign = _campaign(max_nulls=10)
    with pytest.raises(ValueError, match="must be frozen explicitly"):
        build_exact_null_campaign_plan(graph, campaign, ExactNullCampaignConfig(n_nulls=2))


def test_exact_null_plan_pins_search_statistic_and_seed_policy():
    from gwpop_search.inference.fidelity import fidelity_config_sha256

    graph, campaign = _campaign(max_nulls=10)
    config = ExactNullCampaignConfig(
        n_nulls=3,
        root_seed=123,
        max_gpu_hours_per_null=6.0,
    )
    plan = build_exact_null_campaign_plan(graph, campaign, config)

    assert plan["format_version"] == "gwpop-search-exact-null-plan-1.2"
    assert plan["graph_root_hash"] == graph.root_hash
    assert plan["null_config"]["n_nulls"] == 3
    assert len(plan["seed_policy"]) == 3
    assert plan["seed_policy"][0]["data_seed"] != plan["seed_policy"][0]["search_seed"]
    statistic = plan["statistic"]
    assert statistic["mode"] == "f3_completion"
    assert statistic["evidence_fidelities"] == ["F3"]
    assert statistic["replay_ladder_stop"] == "F3"
    assert plan["sampler_backend"] == {"name": "dynesty", "version": "3.1.0"}
    assert plan["fidelity_config_sha256"] == fidelity_config_sha256(campaign.fidelity)
    replayed = plan["replayed_production_search"]
    assert replayed["stop_fidelity"] == "F3"
    assert replayed["production_ladder"] == ["F0", "F3", "F4"]
    assert replayed["null_ladder"] == ["F0", "F3"]
    assert replayed["max_gpu_hours"] == 6.0
    assert (
        replayed["source_production_max_gpu_hours"]
        == campaign.budget.max_gpu_hours
    )
    assert replayed["max_f3_models"] == campaign.budget.max_f3_models
    assert replayed["evidence_completion_required"] is True



def test_exact_null_config_rejects_missing_truth_parameter():
    config = ExactNullCampaignConfig()
    truth = dict(config.truth_hyperparameters)
    truth.pop("mmin")
    with pytest.raises(ValueError, match="missing"):
        ExactNullCampaignConfig(truth_hyperparameters=truth)



def test_exact_null_plan_requires_production_full_graph_f3_budget():
    graph, campaign = _campaign(max_nulls=10, max_f3_models=1)
    with pytest.raises(ValueError, match="full-graph F3"):
        build_exact_null_campaign_plan(
            graph,
            campaign,
            ExactNullCampaignConfig(n_nulls=2, max_gpu_hours_per_null=2.0),
        )


def test_old_null_campaign_format_is_rejected():
    config = ExactNullCampaignConfig(max_gpu_hours_per_null=2.0)
    payload = config.to_dict()
    for legacy in (
        "gwpop-search-exact-null-campaign-1.2",
        "gwpop-search-exact-null-campaign-1.3",
    ):
        payload["format_version"] = legacy
        with pytest.raises(ValueError, match="unsupported exact null"):
            ExactNullCampaignConfig.from_dict(payload)
    with pytest.raises(ValueError, match="re-freeze"):
        ExactNullCampaignConfig(format_version="gwpop-search-exact-null-campaign-1.3")



def test_exact_null_config_can_explicitly_select_engineering_synthetic_mode():
    config = ExactNullCampaignConfig(
        n_nulls=2,
        data_mode="synthetic_survey",
        survey=_noisy_survey(injection_draw="population_proxy"),
    )
    assert config.data_mode == "synthetic_survey"


def test_exact_null_config_rejects_unknown_data_mode():
    with pytest.raises(ValueError, match="unsupported null data mode"):
        ExactNullCampaignConfig(data_mode="not-a-mode")


def test_exact_null_plan_pins_production_dataset_for_frozen_selection_mode():
    graph, campaign = _campaign(max_nulls=10)
    plan = build_exact_null_campaign_plan(
        graph,
        campaign,
        ExactNullCampaignConfig(n_nulls=2, max_gpu_hours_per_null=2.0),
    )
    assert (
        plan["production_dataset_manifest_hash"]
        == campaign.dataset_manifest_hash
    )


def test_synthetic_null_plan_does_not_claim_production_dataset_resampling():
    graph, campaign = _campaign(max_nulls=10)
    plan = build_exact_null_campaign_plan(
        graph,
        campaign,
        ExactNullCampaignConfig(
            n_nulls=2,
            data_mode="synthetic_survey",
            max_gpu_hours_per_null=2.0,
        ),
    )
    assert plan["production_dataset_manifest_hash"] is None



def test_exact_null_config_rejects_invalid_per_null_compute_cap():
    with pytest.raises(ValueError, match="max_gpu_hours_per_null"):
        ExactNullCampaignConfig(max_gpu_hours_per_null=0.0)


def test_exact_null_plan_records_per_null_compute_cap():
    graph, campaign = _campaign(max_nulls=10)
    config = ExactNullCampaignConfig(
        n_nulls=2,
        max_gpu_hours_per_null=3.5,
    )
    plan = build_exact_null_campaign_plan(graph, campaign, config)
    assert plan["format_version"] == "gwpop-search-exact-null-plan-1.2"
    assert plan["null_config"]["max_gpu_hours_per_null"] == 3.5
    assert plan["replayed_production_search"]["max_gpu_hours"] == 3.5
