"""Phase-3 synthetic survey: selection injections drawn from a population proxy."""

from dataclasses import asdict

import numpy as np
import pytest
from scipy.stats import kstest

from gwpop_search.cli import build_parser
from gwpop_search.data import SelectionMode
from gwpop_search.hbi.numpy_backend import evaluate_selection
from gwpop_search.inference.campaign import RecoveryAcceptanceCriteria, build_campaign_plan
from gwpop_search.inference.numpyro import NUTSConfig
from gwpop_search.inference.priors import BASELINE_SYNTHETIC_PRIORS, PriorSpec
from gwpop_search.inference.synthetic import (
    DEFAULT_POPULATION_PROXY_HYPERPARAMETERS,
    INJECTION_DRAWS,
    SyntheticSurveyConfig,
    _draw_population,
    detection_mask,
    generate_baseline_synthetic_dataset,
    population_proxy_log_draw_density,
    population_proxy_support_violations,
)
from gwpop_search.models import (
    DEFAULT_BASELINE_HYPERPARAMETERS,
    GwcatChiEffBBHModel,
    chi_eff_logpdf,
    mass_ratio_logpdf,
    primary_mass_powerlaw_peak_logpdf,
    redshift_rate_logpdf,
)

MODEL = GwcatChiEffBBHModel()
FIELDS = GwcatChiEffBBHModel.required_fields
LEGACY_FIELDS = (
    "n_events",
    "posterior_samples_per_event",
    "n_injections",
    "observing_time_yr",
    "reference_chirp_mass",
    "reference_horizon_mpc",
    "m1_detector_min",
    "m1_detector_max",
    "pe_m1_fractional_sigma",
    "pe_q_sigma",
    "pe_d_l_fractional_sigma",
    "pe_chi_eff_sigma",
    "population_batch_size",
    "redshift_sampling_grid",
)
PRIOR_MEDIAN = {
    name: (
        0.5 * (spec.low + spec.high)
        if spec.family == "uniform"
        else float(np.sqrt(spec.low * spec.high))
    )
    for name, spec in BASELINE_SYNTHETIC_PRIORS.items()
}


def _small(**overrides):
    values = {
        "n_events": 8,
        "posterior_samples_per_event": 16,
        "n_injections": 5_000,
        "population_batch_size": 512,
    }
    values.update(overrides)
    return SyntheticSurveyConfig(**values)


def _prior_draw(rng):
    out = {}
    for name, spec in BASELINE_SYNTHETIC_PRIORS.items():
        if spec.family == "uniform":
            out[name] = float(rng.uniform(spec.low, spec.high))
        else:
            out[name] = float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))
    return out


def _columns(draw):
    return {name: np.asarray(draw[name], dtype=float) for name in FIELDS}


def test_default_config_keeps_legacy_uniform_box_and_manifest_payload():
    config = SyntheticSurveyConfig()
    assert config.injection_draw == "uniform_detector_box"
    assert config.injection_draw_hyperparameters is None

    payload = config.to_dict()
    assert tuple(payload) == LEGACY_FIELDS
    legacy = {name: value for name, value in asdict(config).items() if name in LEGACY_FIELDS}
    assert payload == legacy
    assert SyntheticSurveyConfig.from_dict(payload) == config

    plan = build_campaign_plan(
        n_runs=4,
        root_seed=20260917,
        survey_config=config,
        nuts_config=NUTSConfig(),
        selection_chunk_size=4096,
        criteria=RecoveryAcceptanceCriteria(),
    )
    assert tuple(plan["survey_config"]) == LEGACY_FIELDS


def test_explicit_uniform_box_is_identical_to_the_default():
    default = generate_baseline_synthetic_dataset(seed=77, config=_small())
    explicit = generate_baseline_synthetic_dataset(
        seed=77,
        config=_small(injection_draw="uniform_detector_box"),
    )
    for name in default.selection.field_names:
        np.testing.assert_array_equal(
            default.selection.samples[name], explicit.selection.samples[name]
        )
    np.testing.assert_array_equal(
        default.selection.log_draw_density, explicit.selection.log_draw_density
    )
    assert np.unique(default.selection.log_draw_density).size == 1
    assert default.selection.metadata == {
        "fixture": "phase3-synthetic-selection",
        "n_detected": default.selection.n_selected,
    }
    assert default.selection.campaigns[0].metadata == {
        "detection_rule": "chirp_mass_scaled_reach"
    }


def test_population_proxy_config_resolves_default_and_round_trips():
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    assert config.injection_draw_hyperparameters == DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    assert population_proxy_support_violations(config.injection_draw_hyperparameters) == ()

    payload = config.to_dict()
    assert payload["injection_draw"] == "population_proxy"
    assert payload["injection_draw_hyperparameters"] == DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    assert SyntheticSurveyConfig.from_dict(payload) == config

    custom = dict(DEFAULT_POPULATION_PROXY_HYPERPARAMETERS, alpha=2.0, kappa=0.0)
    explicit = SyntheticSurveyConfig(
        injection_draw="population_proxy",
        injection_draw_hyperparameters=custom,
    )
    assert explicit.injection_draw_hyperparameters == custom


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"injection_draw": "log_uniform_box"}, "injection_draw must be one of"),
        (
            {"injection_draw_hyperparameters": dict(DEFAULT_POPULATION_PROXY_HYPERPARAMETERS)},
            "only used by",
        ),
        (
            {
                "injection_draw": "population_proxy",
                "injection_draw_hyperparameters": {"alpha": 1.0},
            },
            "exactly the baseline",
        ),
        (
            {
                "injection_draw": "population_proxy",
                "injection_draw_hyperparameters": dict(
                    DEFAULT_POPULATION_PROXY_HYPERPARAMETERS, mmin=3.0
                ),
            },
            "smallest prior mmin",
        ),
        (
            {
                "injection_draw": "population_proxy",
                "injection_draw_hyperparameters": dict(
                    DEFAULT_POPULATION_PROXY_HYPERPARAMETERS, mmax=100.0
                ),
            },
            "largest prior mmax",
        ),
        (
            {
                "injection_draw": "population_proxy",
                "injection_draw_hyperparameters": dict(
                    DEFAULT_POPULATION_PROXY_HYPERPARAMETERS, chi_sigma=float("nan")
                ),
            },
            "finite",
        ),
        (
            {
                "injection_draw": "population_proxy",
                "injection_draw_hyperparameters": dict(
                    DEFAULT_POPULATION_PROXY_HYPERPARAMETERS, peak_sigma=0.0
                ),
            },
            "positive",
        ),
    ],
)
def test_injection_draw_configuration_is_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SyntheticSurveyConfig(**kwargs)


def test_support_violations_require_bounded_mass_priors():
    proxy = DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    priors = dict(BASELINE_SYNTHETIC_PRIORS)
    priors["mmin"] = PriorSpec("normal", loc=5.0, scale=1.0)
    assert any("unbounded" in item for item in population_proxy_support_violations(proxy, priors))
    del priors["mmax"]
    assert len(population_proxy_support_violations(proxy, priors)) == 2


def test_population_proxy_selection_catalog_contract():
    config = _small(n_injections=20_000, injection_draw="population_proxy")
    dataset = generate_baseline_synthetic_dataset(seed=4321, config=config, model=MODEL)
    selection = dataset.selection
    proxy = config.injection_draw_hyperparameters

    assert selection.mode is SelectionMode.RAW_DRAW
    assert len(selection.campaigns) == 1
    assert selection.campaigns[0].n_draw == 20_000
    assert selection.campaigns[0].observing_time_yr == config.observing_time_yr
    assert selection.campaigns[0].metadata["injection_draw"] == "population_proxy"
    assert selection.metadata["injection_draw"] == "population_proxy"
    assert selection.metadata["injection_draw_hyperparameters"] == proxy
    assert selection.metadata["draw_density_model"] == MODEL.to_config()
    assert 0 < selection.n_selected < 20_000
    assert np.all(detection_mask(selection.samples, config))

    # The stored raw-draw density is exactly the model density at the proxy.
    expected = np.asarray(MODEL(_columns(selection.samples), proxy), dtype=float)
    np.testing.assert_array_equal(selection.log_draw_density, expected)
    assert np.all(np.isfinite(selection.log_draw_density))

    # Selection is drawn after PE, so the PE catalog does not depend on the draw.
    legacy = generate_baseline_synthetic_dataset(
        seed=4321, config=_small(n_injections=20_000), model=MODEL
    )
    for name in dataset.posterior.field_names:
        np.testing.assert_array_equal(
            dataset.posterior.samples[name], legacy.posterior.samples[name]
        )


def _grid_cdf(grid, log_density):
    density = np.exp(np.asarray(log_density, dtype=float))
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))))
    return cdf


def test_population_proxy_draws_follow_the_declared_model_density():
    """KS tests of the proxy sampler against the model's own normalized factors.

    The detector-frame proxy density is the product of these source-frame
    factors and the model's explicit Jacobian, so each marginal/conditional
    probability-integral transform must be uniform.
    """
    proxy = DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    draw = _draw_population(np.random.default_rng(11), 100_000, proxy, MODEL, config)
    z = np.asarray(MODEL.cosmology.z_of_dL(draw["luminosity_distance"]), dtype=float)
    m1 = np.asarray(draw["m1_detector"], dtype=float) / (1.0 + z)
    q = np.asarray(draw["q"], dtype=float)
    chi = np.asarray(draw["chi_eff"], dtype=float)
    p_min = 1e-3

    z_grid = np.linspace(0.0, MODEL.zmax, 50_001)[1:]
    z_log = redshift_rate_logpdf(
        z_grid,
        kappa=proxy["kappa"],
        zmax=MODEL.zmax,
        cosmology=MODEL.cosmology,
        quadrature_order=MODEL.redshift_quadrature_order,
    )
    z_cdf = _grid_cdf(z_grid, z_log)
    assert abs(z_cdf[-1] - 1.0) < 1e-4
    assert kstest(z, lambda x: np.interp(x, z_grid, z_cdf / z_cdf[-1])).pvalue > p_min

    m_grid = np.linspace(proxy["mmin"], proxy["mmax"], 400_001)
    m_log = primary_mass_powerlaw_peak_logpdf(
        m_grid,
        alpha=proxy["alpha"],
        mmin=proxy["mmin"],
        mmax=proxy["mmax"],
        peak_fraction=proxy["peak_fraction"],
        peak_mu=proxy["peak_mu"],
        peak_sigma=proxy["peak_sigma"],
    )
    m_cdf = _grid_cdf(m_grid, m_log)
    assert abs(m_cdf[-1] - 1.0) < 1e-4
    assert kstest(m1, lambda x: np.interp(x, m_grid, m_cdf / m_cdf[-1])).pvalue > p_min

    c_grid = np.linspace(-1.0, 1.0, 40_001)
    c_cdf = _grid_cdf(c_grid, chi_eff_logpdf(c_grid, mu=proxy["chi_mu"], sigma=proxy["chi_sigma"]))
    assert abs(c_cdf[-1] - 1.0) < 1e-6
    assert kstest(chi, lambda x: np.interp(x, c_grid, c_cdf / c_cdf[-1])).pvalue > p_min

    # q | m1: the model density is q**beta / norm(m1) on [qmin(m1), 1], so the
    # conditional CDF is (q**(beta+1) - qmin**(beta+1)) / ((beta+1) norm(m1)),
    # with norm(m1) read back from the model's own log density.
    beta = proxy["beta_q"]
    log_q = np.asarray(
        mass_ratio_logpdf(q, m1, beta=beta, mmin=proxy["mmin"], q_floor=MODEL.q_floor),
        dtype=float,
    )
    assert np.all(np.isfinite(log_q))
    inv_norm = np.exp(log_q - beta * np.log(q))
    q_min = np.maximum(MODEL.q_floor, proxy["mmin"] / m1)
    pit = (q ** (beta + 1.0) - q_min ** (beta + 1.0)) / (beta + 1.0) * inv_norm
    assert kstest(pit, "uniform").pvalue > p_min

    assert kstest(np.asarray(draw["ra"]) / (2.0 * np.pi), "uniform").pvalue > p_min
    assert kstest(0.5 * (np.sin(np.asarray(draw["dec"])) + 1.0), "uniform").pvalue > p_min


def test_population_proxy_covers_every_hyperprior_population():
    proxy = DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    rng = np.random.default_rng(20260919)

    points = [_prior_draw(rng) for _ in range(40)]
    corners = {
        "alpha": 8.0,
        "peak_fraction": 0.5,
        "peak_sigma": 15.0,
        "beta_q": 12.0,
        "kappa": 12.0,
        "chi_mu": 0.3,
        "chi_sigma": 0.5,
    }
    for mmin in (2.0, 10.0):
        for mmax in (60.0, 120.0):
            points.append(dict(PRIOR_MEDIAN, mmin=mmin, mmax=mmax, **corners))
            points.append(
                dict(PRIOR_MEDIAN, mmin=mmin, mmax=mmax, alpha=0.0, beta_q=-4.0, kappa=-6.0)
            )

    for hp in points:
        samples = _columns(_draw_population(rng, 1_000, hp, MODEL, config))
        log_pop = np.asarray(MODEL(samples, hp), dtype=float)
        log_draw = population_proxy_log_draw_density(MODEL, samples, proxy)
        supported = np.isfinite(log_pop)
        assert supported.mean() > 0.999, hp
        assert np.all(np.isfinite(log_draw[supported])), hp


def test_population_proxy_importance_weights_are_unbiased_over_all_draws():
    proxy = DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    rng = np.random.default_rng(0)
    n = 200_000
    samples = _columns(_draw_population(rng, n, proxy, MODEL, config))
    log_draw = population_proxy_log_draw_density(MODEL, samples, proxy)
    assert np.all(np.isfinite(log_draw))

    for hp in (DEFAULT_BASELINE_HYPERPARAMETERS, PRIOR_MEDIAN):
        assert hp != proxy
        weights = np.exp(np.asarray(MODEL(samples, hp), dtype=float) - log_draw)
        mean = weights.mean()
        standard_error = weights.std() / np.sqrt(n)
        assert standard_error < 0.05
        assert abs(mean - 1.0) < 4.0 * standard_error, (hp, mean, standard_error)


@pytest.mark.parametrize("point", ["truth", "median"])
def test_population_proxy_selection_matches_brute_force_detected_fraction(point):
    hp = DEFAULT_BASELINE_HYPERPARAMETERS if point == "truth" else PRIOR_MEDIAN
    config = _small(n_injections=300_000, injection_draw="population_proxy")
    dataset = generate_baseline_synthetic_dataset(seed=99, config=config, model=MODEL)
    result = evaluate_selection(dataset.selection, MODEL, hp)
    # raw-draw exposure is T/N_draw * sum_detected p_pop/p_draw with T = 1 yr.
    assert config.observing_time_yr == 1.0
    estimate = float(np.exp(result.log_exposure))
    estimate_error = estimate * float(np.sqrt(result.variance_log_exposure))

    rng = np.random.default_rng(7)
    m = 500_000
    direct = detection_mask(_draw_population(rng, m, hp, MODEL, config), config)
    fraction = float(direct.mean())
    fraction_error = float(np.sqrt(fraction * (1.0 - fraction) / m))

    combined = float(np.hypot(estimate_error, fraction_error))
    assert combined < 0.03 * fraction
    assert abs(estimate - fraction) < 4.0 * combined, (estimate, fraction, combined)


def test_synthetic_cli_exposes_injection_draw():
    parser = build_parser()
    for command in (
        ["synthetic-recovery", "--run-dir", "runs/x"],
        ["synthetic-campaign", "--root", "runs/y"],
    ):
        default = parser.parse_args(command)
        assert default.injection_draw == "uniform_detector_box"
        for draw in INJECTION_DRAWS:
            chosen = parser.parse_args(command + ["--injection-draw", draw])
            assert chosen.injection_draw == draw
        with pytest.raises(SystemExit):
            parser.parse_args(command + ["--injection-draw", "uniform"])

    from gwpop_search.cli import _survey_config

    config = _survey_config(
        parser.parse_args(
            ["synthetic-campaign", "--root", "runs/y", "--injection-draw", "population_proxy"]
        )
    )
    assert config.injection_draw == "population_proxy"
    assert config.injection_draw_hyperparameters == DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
