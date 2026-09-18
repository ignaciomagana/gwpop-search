import json

import pytest

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference.fidelity import FidelityRunConfig
from gwpop_search.inference.synthetic import SyntheticSurveyConfig
from gwpop_search.nulls import (
    ExactNullCampaignConfig,
    SearchReplayResult,
    finalize_exact_null_campaign,
    null_replay_seed,
    prepare_exact_null_campaign,
    run_exact_null_campaign,
    run_exact_null_index,
)
from gwpop_search.production import (
    ProductionCampaignConfig,
    SearchBudget,
    SeedPolicy,
)
from gwpop_search.search import SchedulerConfig


def _graph():
    return enumerate_model_graph(
        baseline_model_spec(),
        max_depth=0,
        max_models=1,
    )


def _campaign(graph):
    return ProductionCampaignConfig(
        campaign_id="indexed-null-test",
        dataset_manifest_hash="a" * 64,
        model_graph_hash="b" * 64,
        model_graph_root_hash=graph.root_hash,
        git_commit="c" * 40,
        model_prior={"version": "uniform-v1"},
        fidelity=FidelityRunConfig(),
        scheduler=SchedulerConfig(),
        seed_policy=SeedPolicy(root_seed=101),
        budget=SearchBudget(
            max_gpu_hours=50.0,
            max_f3_models=1,
            max_f4_models=1,
            max_null_replays=10,
        ),
        artifact_root="runs",
        state_database="runs/state.sqlite",
    )


def _config():
    return ExactNullCampaignConfig(
        n_nulls=2,
        root_seed=202,
        survey=SyntheticSurveyConfig(
            n_events=4,
            posterior_samples_per_event=8,
            n_injections=200,
        ),
        data_mode="synthetic_survey",
        max_gpu_hours_per_null=3.0,
    )


def _fake_replay_factory(calls):
    def fake(
        null_index,
        seed,
        **kwargs,
    ):
        calls.append((int(null_index), int(seed)))
        return SearchReplayResult(
            null_index=int(null_index),
            seed=int(seed),
            max_log_bayes_factor=float(null_index + 1),
            max_log_posterior_odds=float(null_index + 0.5),
            n_models_evaluated=1,
            best_model_hash=kwargs["graph"].root_hash,
            metadata={"fake": True},
        )
    return fake


def test_indexed_null_replay_requires_prepared_plan(
    monkeypatch,
    tmp_path,
):
    graph = _graph()
    campaign = _campaign(graph)
    config = _config()
    calls = []
    monkeypatch.setattr(
        "gwpop_search.nulls.campaign.run_baseline_null_search_replay",
        _fake_replay_factory(calls),
    )

    with pytest.raises(ValueError, match="prepare-null-search-calibration"):
        run_exact_null_index(
            tmp_path,
            graph,
            campaign,
            config,
            null_index=0,
        )
    assert calls == []


def test_indexed_null_replay_is_deterministic_and_resumable(
    monkeypatch,
    tmp_path,
):
    graph = _graph()
    campaign = _campaign(graph)
    config = _config()
    prepare_exact_null_campaign(tmp_path, graph, campaign, config)

    calls = []
    monkeypatch.setattr(
        "gwpop_search.nulls.campaign.run_baseline_null_search_replay",
        _fake_replay_factory(calls),
    )
    first = run_exact_null_index(
        tmp_path,
        graph,
        campaign,
        config,
        null_index=1,
    )
    second = run_exact_null_index(
        tmp_path,
        graph,
        campaign,
        config,
        null_index=1,
    )

    expected_seed = null_replay_seed(config.root_seed, 1)
    assert first == second
    assert first.seed == expected_seed
    assert calls == [(1, expected_seed)]
    payload = json.loads((tmp_path / "null_00001.json").read_text())
    assert payload["null_index"] == 1
    assert payload["seed"] == expected_seed


def test_indexed_null_replay_rejects_out_of_range_index(tmp_path):
    graph = _graph()
    campaign = _campaign(graph)
    config = _config()
    prepare_exact_null_campaign(tmp_path, graph, campaign, config)

    with pytest.raises(ValueError, match="outside"):
        run_exact_null_index(
            tmp_path,
            graph,
            campaign,
            config,
            null_index=2,
        )


def test_finalize_requires_every_declared_index(
    monkeypatch,
    tmp_path,
):
    graph = _graph()
    campaign = _campaign(graph)
    config = _config()
    prepare_exact_null_campaign(tmp_path, graph, campaign, config)
    monkeypatch.setattr(
        "gwpop_search.nulls.campaign.run_baseline_null_search_replay",
        _fake_replay_factory([]),
    )
    run_exact_null_index(
        tmp_path,
        graph,
        campaign,
        config,
        null_index=0,
    )

    with pytest.raises(ValueError, match="missing indices"):
        finalize_exact_null_campaign(
            tmp_path,
            graph,
            campaign,
            config,
        )


def test_indexed_and_serial_null_campaigns_finalize_identically(
    monkeypatch,
    tmp_path,
):
    graph = _graph()
    campaign = _campaign(graph)
    config = _config()

    indexed_root = tmp_path / "indexed"
    prepare_exact_null_campaign(
        indexed_root,
        graph,
        campaign,
        config,
    )
    indexed_calls = []
    monkeypatch.setattr(
        "gwpop_search.nulls.campaign.run_baseline_null_search_replay",
        _fake_replay_factory(indexed_calls),
    )
    for index in range(config.n_nulls):
        run_exact_null_index(
            indexed_root,
            graph,
            campaign,
            config,
            null_index=index,
        )
    indexed = finalize_exact_null_campaign(
        indexed_root,
        graph,
        campaign,
        config,
    )

    serial_root = tmp_path / "serial"
    serial_calls = []
    monkeypatch.setattr(
        "gwpop_search.nulls.campaign.run_baseline_null_search_replay",
        _fake_replay_factory(serial_calls),
    )
    serial = run_exact_null_campaign(
        serial_root,
        graph,
        campaign,
        config,
    )

    assert indexed["calibration"] == serial["calibration"]
    assert indexed["format_version"] == serial["format_version"]
    assert indexed["calibration"]["n_null_replays"] == 2
    assert indexed_calls == serial_calls
