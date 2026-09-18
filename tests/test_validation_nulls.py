import numpy as np

from gwpop_search.data.fixtures import (
    make_toy_posterior_catalog,
    make_toy_selection_catalog,
)
from gwpop_search.nulls import (
    SearchReplayResult,
    calibrate_search_replays,
    empirical_tail_probability,
    null_replay_seed,
    run_null_replay_campaign,
)
from gwpop_search.validation import (
    compare_holdout_models,
    deterministic_event_folds,
    heldout_detected_log_predictive,
)


def _toy_density(samples, hp):
    return (
        -0.02 * samples["m1_source"]
        + hp["q_slope"] * samples["q"]
        + hp["chi_slope"] * samples["chi_eff"]
        - 0.1 * samples["z"]
    )


def test_event_folds_are_deterministic_and_seed_dependent():
    names = ("A", "B", "C", "D", "E", "F")
    a = deterministic_event_folds(names, n_folds=3, seed=1)
    b = deterministic_event_folds(names, n_folds=3, seed=1)
    c = deterministic_event_folds(names, n_folds=3, seed=2)

    assert a == b
    assert a != c
    assert set(a) == set(names)
    assert all(0 <= fold < 3 for fold in a.values())


def test_heldout_detected_predictive_is_finite_and_repeatable():
    posterior = make_toy_posterior_catalog()
    selection = make_toy_selection_catalog()
    hyperposterior = {
        "q_slope": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "chi_slope": np.asarray([-0.2, 0.0, 0.1, 0.2]),
    }
    heldout = posterior.event_names[:2]

    a = heldout_detected_log_predictive(
        posterior,
        selection,
        _toy_density,
        hyperposterior,
        heldout_events=heldout,
    )
    b = heldout_detected_log_predictive(
        posterior,
        selection,
        _toy_density,
        hyperposterior,
        heldout_events=heldout,
    )

    assert a == b
    assert set(a) == set(heldout)
    assert all(np.isfinite(value) for value in a.values())


def test_holdout_model_comparison_returns_totals_without_ranking_label():
    scores = {
        "model-a": {"E1": -1.0, "E2": -2.0},
        "model-b": {"E1": -0.5, "E2": -2.2},
    }
    totals = compare_holdout_models(scores)
    assert totals == {"model-a": -3.0, "model-b": -2.7}


def test_empirical_tail_probability_uses_finite_sample_correction():
    result = empirical_tail_probability([1.0, 2.0, 3.0, 4.0], 4.5)
    assert result["n_exceed"] == 0
    assert result["tail_probability"] == 0.2
    assert result["minimum_resolvable_tail_probability"] == 0.2

    result2 = empirical_tail_probability([1.0, 2.0, 3.0, 4.0], 3.0)
    assert result2["n_exceed"] == 2
    assert result2["tail_probability"] == 0.6


def _replay(index, seed):
    return SearchReplayResult(
        null_index=index,
        seed=seed,
        max_log_bayes_factor=0.5 * index,
        max_log_posterior_odds=0.25 * index,
        n_models_evaluated=20 + index,
        best_model_hash=f"{index + 1:064x}",
        metadata={"pipeline": "full-search-test"},
    )


def test_null_replay_campaign_is_deterministic_and_resumable(tmp_path):
    calls = []

    def callback(index, seed):
        calls.append((index, seed))
        return _replay(index, seed)

    first = run_null_replay_campaign(
        tmp_path,
        n_nulls=5,
        root_seed=123,
        replay=callback,
    )
    assert len(calls) == 5
    assert first["n_null_replays"] == 5

    calls.clear()
    second = run_null_replay_campaign(
        tmp_path,
        n_nulls=5,
        root_seed=123,
        replay=callback,
    )
    assert calls == []
    assert second == first

    seeds = [null_replay_seed(123, index) for index in range(5)]
    assert len(set(seeds)) == 5


def test_search_replay_calibration_uses_maximum_search_statistic():
    results = tuple(_replay(index, 10 + index) for index in range(10))
    calibrated = calibrate_search_replays(
        results,
        observed_max_log_bayes_factor=4.2,
        observed_max_log_posterior_odds=2.1,
    )

    assert calibrated["n_null_replays"] == 10
    assert calibrated["max_log_bayes_factor"]["q90"] > 3.0
    assert calibrated["observed_log_bf_calibration"]["n_exceed"] == 1
    assert calibrated["observed_log_posterior_odds_calibration"]["n_exceed"] == 1
