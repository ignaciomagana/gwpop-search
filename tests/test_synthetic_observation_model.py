"""Phase-3 synthetic survey: DAG-consistent noisy observations (Essick & Fishbach 2023)."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr
from scipy.stats import kstest

from gwpop_search.hbi.numpy_backend import evaluate_selection
from gwpop_search.inference.priors import BASELINE_SYNTHETIC_PRIORS
from gwpop_search.inference.synthetic import (
    OBSERVATION_MODELS,
    SyntheticSurveyConfig,
    _detector_prior_bounds,
    _draw_population,
    _make_posterior_catalog,
    _noisy_posterior_draws,
    _observe,
    _uniform_detector_log_density,
    detection_mask,
    generate_baseline_synthetic_dataset,
    observed_detection_mask,
    pe_truth_quantiles,
)
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS, GwcatChiEffBBHModel

MODEL = GwcatChiEffBBHModel()
TRUTH = DEFAULT_BASELINE_HYPERPARAMETERS
PRIOR_MEDIAN = {
    name: (
        0.5 * (spec.low + spec.high)
        if spec.family == "uniform"
        else float(np.sqrt(spec.low * spec.high))
    )
    for name, spec in BASELINE_SYNTHETIC_PRIORS.items()
}


def _config(**overrides):
    values = {
        "n_events": 64,
        "posterior_samples_per_event": 16,
        "n_injections": 5_000,
        "population_batch_size": 512,
        "observation_model": "noisy_observation",
    }
    values.update(overrides)
    return SyntheticSurveyConfig(**values)


def _randomized_truth_pit(posterior, truths, name, rng):
    """Rank of the truth among n PE draws, spread uniformly: exactly U(0,1) under SBC."""
    n = int(np.diff(posterior.offsets)[0])
    rank = np.round(pe_truth_quantiles(posterior, truths, name) * n)
    return (rank + rng.random(rank.size)) / (n + 1)


def test_default_observation_model_is_legacy_and_omitted_from_manifests():
    assert OBSERVATION_MODELS == ("truth_centered", "noisy_observation")
    legacy = SyntheticSurveyConfig()
    assert legacy.observation_model == "truth_centered"
    assert "observation_model" not in legacy.to_dict()
    assert not legacy.uses_v2_options

    noisy = SyntheticSurveyConfig(observation_model="noisy_observation")
    payload = noisy.to_dict()
    assert payload["observation_model"] == "noisy_observation"
    assert tuple(payload)[-1] == "observation_model"
    assert "injection_draw" not in payload
    assert noisy.uses_v2_options
    assert SyntheticSurveyConfig.from_dict(payload) == noisy


def test_noisy_events_are_detected_on_their_observed_data():
    config = _config()
    dataset = generate_baseline_synthetic_dataset(seed=21, config=config, model=MODEL)
    observed = dataset.event_observations
    assert set(observed) == {"log_m1_detector", "q", "log_luminosity_distance", "chi_eff"}
    assert all(values.shape == (config.n_events,) for values in observed.values())

    bounds = _detector_prior_bounds(MODEL, config, TRUTH)
    assert np.all(observed_detection_mask(observed, config, bounds))
    # Detection acts on data: upward noise fluctuations detect systems whose
    # true parameters lie beyond the reach (Malmquist), which a truth-side cut
    # can never do.
    assert np.count_nonzero(~detection_mask(dataset.event_truths, config)) > 0
    # One noise realisation per event: the data are not the truths.
    offset = observed["chi_eff"] - dataset.event_truths["chi_eff"]
    assert 0.08 < np.std(offset) < 0.16

    assert dataset.posterior.metadata["observation_model"] == "noisy_observation"
    assert dataset.selection.metadata["observation_model"] == "noisy_observation"
    assert dataset.selection.campaigns[0].metadata == {
        "detection_rule": "chirp_mass_scaled_reach",
        "detection_applied_to": "observed_data",
    }


def test_observed_detection_rule_equals_the_truth_rule_on_noise_free_data():
    config = _config()
    rng = np.random.default_rng(4)
    draw = _draw_population(rng, 50_000, TRUTH, MODEL, config)
    bounds = _detector_prior_bounds(MODEL, config, TRUTH)
    noise_free = {
        "log_m1_detector": np.log(draw["m1_detector"]),
        "q": np.asarray(draw["q"]),
        "log_luminosity_distance": np.log(draw["luminosity_distance"]),
        "chi_eff": np.asarray(draw["chi_eff"]),
    }
    same = observed_detection_mask(noise_free, config, bounds) == detection_mask(draw, config)
    # Only exact ties at the reach can differ by round-off.
    assert same.mean() > 1.0 - 1e-4


def test_noisy_pe_draws_are_the_exact_posterior_given_the_data():
    """KS of each coordinate against the posterior computed on a fine grid."""
    config = _config()
    bounds = _detector_prior_bounds(MODEL, config, TRUTH)
    s_m, s_q, s_d, s_c = (
        config.pe_m1_fractional_sigma,
        config.pe_q_sigma,
        config.pe_d_l_fractional_sigma,
        config.pe_chi_eff_sigma,
    )
    observed = {
        "log_m1_detector": np.log(2.6),  # close to the m1_detector_min = 2 edge
        "q": 0.97,  # close to the q = 1 edge
        "log_luminosity_distance": np.log(900.0),
        "chi_eff": 0.9,  # close to the chi_eff = 1 edge
    }
    draws = _noisy_posterior_draws(np.random.default_rng(8), observed, config, bounds, 200_000)

    def check(values, grid, log_likelihood):
        # Posterior under the uniform box prior: likelihood x 1 on the grid.
        density = np.exp(log_likelihood - log_likelihood.max())
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))))
        cdf /= cdf[-1]
        assert kstest(values, lambda x: np.interp(x, grid, cdf)).pvalue > 1e-3

    m = np.linspace(bounds["m1_min"], 6.0, 400_001)
    check(draws["m1_detector"], m, -0.5 * ((observed["log_m1_detector"] - np.log(m)) / s_m) ** 2)
    q = np.linspace(bounds["q_min"], bounds["q_max"], 400_001)
    check(draws["q"], q, -0.5 * ((observed["q"] - q) / s_q) ** 2)
    d = np.linspace(1e-3, 3_000.0, 400_001)
    check(
        draws["luminosity_distance"],
        d,
        -0.5 * ((observed["log_luminosity_distance"] - np.log(d)) / s_d) ** 2,
    )
    c = np.linspace(-1.0, 1.0, 400_001)
    check(draws["chi_eff"], c, -0.5 * ((observed["chi_eff"] - c) / s_c) ** 2)
    # A PE sampler that drops the uniform-in-m1 Jacobian shift is rejected.
    shifted = np.exp(np.log(draws["m1_detector"]) - s_m**2)
    density = np.exp(-0.5 * ((observed["log_m1_detector"] - np.log(m)) / s_m) ** 2)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(m))))
    assert kstest(shifted, lambda x: np.interp(x, m, cdf / cdf[-1])).pvalue < 1e-6


def test_noisy_pe_prior_box_covers_every_hyperprior_population():
    config = _config()
    bounds = _detector_prior_bounds(MODEL, config, TRUTH)
    sup_mmax = BASELINE_SYNTHETIC_PRIORS["mmax"].high
    assert bounds["m1_max"] >= sup_mmax * (1.0 + MODEL.zmax)
    assert bounds["m1_min"] <= BASELINE_SYNTHETIC_PRIORS["mmin"].low
    # Truth-independent: the same box for any truth inside the hyperprior.
    assert _detector_prior_bounds(MODEL, config, PRIOR_MEDIAN) == bounds
    dataset = generate_baseline_synthetic_dataset(seed=2, config=config, model=MODEL)
    np.testing.assert_array_equal(
        dataset.posterior.log_ref_density, _uniform_detector_log_density(bounds)
    )
    legacy = _detector_prior_bounds(MODEL, replace(config, observation_model="truth_centered"), TRUTH)
    assert legacy["m1_max"] == 400.0


def test_truth_quantile_gate_separates_noisy_from_truth_centered_pe():
    """The truth's rank in each event's chi_eff PE is uniform only with noisy data."""
    rng = np.random.default_rng(0)
    pvalues = {}
    for observation in OBSERVATION_MODELS:
        config = _config(n_events=300, posterior_samples_per_event=128, observation_model=observation)
        dataset = generate_baseline_synthetic_dataset(seed=3, config=config, model=MODEL)
        pit = _randomized_truth_pit(dataset.posterior, dataset.event_truths, "chi_eff", rng)
        pvalues[observation] = kstest(pit, "uniform").pvalue
    assert pvalues["noisy_observation"] > 1e-3
    assert pvalues["truth_centered"] < 1e-10


def test_noisy_observation_removes_the_chi_eff_width_bias():
    """Conditional chi_sigma profile of a 400-event catalog peaks at the truth.

    Truth-centered PE (zero noise) makes the observed chi_eff spread equal to
    the population spread, so the model subtracts the PE width and settles
    near sqrt(0.20**2 - 0.12**2) = 0.16. chi_eff does not affect detection, so
    the selection term is independent of chi_sigma and the event terms decide.
    """
    grid = np.round(np.arange(0.10, 0.301, 0.01), 3)
    peaks = {}
    for observation in OBSERVATION_MODELS:
        config = _config(n_events=400, posterior_samples_per_event=64, observation_model=observation)
        dataset = generate_baseline_synthetic_dataset(seed=3, config=config, model=MODEL)
        pe = dataset.posterior
        profile = []
        for sigma in grid:
            log_w = np.asarray(
                MODEL(pe.samples, dict(TRUTH, chi_sigma=float(sigma))), dtype=float
            ) - pe.log_ref_density
            profile.append(logsumexp(log_w.reshape(config.n_events, -1), axis=1).sum())
        peaks[observation] = float(grid[int(np.argmax(profile))])
    assert abs(peaks["noisy_observation"] - TRUTH["chi_sigma"]) <= 0.02
    assert peaks["truth_centered"] <= 0.17


@pytest.mark.parametrize("injection_draw", ["population_proxy", "uniform_detector_box"])
@pytest.mark.parametrize("point", ["truth", "median"])
def test_noisy_selection_matches_brute_force_detection_on_observed_data(injection_draw, point):
    """Injections are detected with the event statistic, so A = P(detected data)."""
    hp = TRUTH if point == "truth" else PRIOR_MEDIAN
    n_injections = 300_000 if injection_draw == "population_proxy" else 1_000_000
    config = _config(
        n_events=8,
        n_injections=n_injections,
        injection_draw=injection_draw,
    )
    dataset = generate_baseline_synthetic_dataset(seed=99, config=config, model=MODEL)
    result = evaluate_selection(dataset.selection, MODEL, hp)
    estimate = float(np.exp(result.log_exposure))
    estimate_error = estimate * float(np.sqrt(result.variance_log_exposure))

    rng = np.random.default_rng(7)
    m = 600_000
    direct = _draw_population(rng, m, hp, MODEL, config)
    bounds = _detector_prior_bounds(MODEL, config, hp)
    detected = observed_detection_mask(_observe(rng, direct, config), config, bounds)
    fraction = float(detected.mean())
    fraction_error = float(np.sqrt(fraction * (1.0 - fraction) / m))
    # The truth-side rule gives a measurably different fraction.
    truth_side = float(detection_mask(direct, config).mean())

    combined = float(np.hypot(estimate_error, fraction_error))
    assert combined < 0.04 * fraction
    assert abs(estimate - fraction) < 4.0 * combined, (estimate, fraction, combined)
    assert abs(truth_side - fraction) > 5.0 * fraction_error


def test_posterior_catalog_requires_observations_exactly_for_noisy_pe():
    noisy = _config()
    legacy = replace(noisy, observation_model="truth_centered")
    rng = np.random.default_rng(1)
    draw = _draw_population(rng, noisy.n_events, TRUTH, MODEL, noisy)
    with pytest.raises(ValueError, match="pass the event observations"):
        _make_posterior_catalog(rng, draw, MODEL, noisy, TRUTH)
    observed = _observe(rng, draw, noisy)
    with pytest.raises(ValueError, match="only used by"):
        _make_posterior_catalog(rng, draw, MODEL, legacy, TRUTH, observed)
    catalog = _make_posterior_catalog(rng, draw, MODEL, noisy, TRUTH, observed)
    assert catalog.n_events == noisy.n_events


def test_structured_scout_datasets_follow_the_observation_model():
    from gwpop_search.scouts import StructuredScoutInjection, generate_structured_scout_dataset

    config = _config(n_events=16, n_injections=2_000)
    dataset = generate_structured_scout_dataset(
        seed=8,
        injection=StructuredScoutInjection("chieff.mean.linear_q", 0.4),
        survey_config=config,
    )
    bounds = _detector_prior_bounds(MODEL, config, TRUTH)
    assert dataset.event_observations is not None
    assert np.all(observed_detection_mask(dataset.event_observations, config, bounds))
    assert dataset.posterior.metadata["observation_model"] == "noisy_observation"
    assert dataset.selection.metadata["observation_model"] == "noisy_observation"
    legacy = generate_structured_scout_dataset(
        seed=8,
        injection=StructuredScoutInjection("chieff.mean.linear_q", 0.4),
        survey_config=replace(config, observation_model="truth_centered"),
    )
    assert legacy.event_observations is None
    assert np.all(detection_mask(legacy.event_truths, config))


def test_noisy_observation_draws_the_same_truth_population():
    """Only the observation layer changes: event truths still follow the model."""
    config = _config(n_events=2_000, posterior_samples_per_event=4, n_injections=1_000)
    dataset = generate_baseline_synthetic_dataset(seed=12, config=config, model=MODEL)
    chi = dataset.event_truths["chi_eff"]
    mu, sigma = TRUTH["chi_mu"], TRUTH["chi_sigma"]
    a, b = (-1.0 - mu) / sigma, (1.0 - mu) / sigma
    cdf = lambda x: (ndtr((x - mu) / sigma) - ndtr(a)) / (ndtr(b) - ndtr(a))
    assert kstest(chi, cdf).pvalue > 1e-3
