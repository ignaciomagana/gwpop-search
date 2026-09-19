"""Phase-3 synthetic survey: selection injections drawn from a population proxy."""

import json
import math
import pickle
from dataclasses import asdict, replace

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
    DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION,
    DEFAULT_POPULATION_PROXY_HYPERPARAMETERS,
    INJECTION_DRAWS,
    POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX,
    POPULATION_PROXY_DEFENSIVE_U_RATIO,
    POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID,
    RECOMMENDED_POPULATION_PROXY_N_INJECTIONS,
    FrozenHyperparameters,
    SyntheticSurveyConfig,
    _apply_defensive_pairing,
    _draw_population,
    detection_mask,
    generate_baseline_synthetic_dataset,
    population_proxy_log_draw_density,
    population_proxy_point_violations,
    population_proxy_support_violations,
    require_population_proxy_coverage,
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


def _proxy_draw(rng, n, config):
    """Draw exactly like the population_proxy selection campaign (before detection)."""
    proxy = config.injection_draw_hyperparameters
    draw = _draw_population(rng, n, proxy, MODEL, config)
    return _apply_defensive_pairing(
        rng, draw, proxy, MODEL.q_floor, config.injection_draw_defensive_fraction
    )


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
    assert (
        config.injection_draw_defensive_fraction
        == DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION
    )
    assert population_proxy_support_violations(config.injection_draw_hyperparameters) == ()

    payload = config.to_dict()
    assert tuple(payload) == LEGACY_FIELDS + (
        "injection_draw",
        "injection_draw_hyperparameters",
        "injection_draw_defensive_fraction",
    )
    assert payload["injection_draw"] == "population_proxy"
    assert payload["injection_draw_hyperparameters"] == DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    assert type(payload["injection_draw_hyperparameters"]) is dict
    assert payload["injection_draw_defensive_fraction"] == DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION
    json.dumps(payload)
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
        ({"injection_draw_defensive_fraction": 0.1}, "only used by"),
        (
            {"injection_draw": "population_proxy", "injection_draw_defensive_fraction": 1.0},
            r"\[0, 1\)",
        ),
        (
            {"injection_draw": "population_proxy", "injection_draw_defensive_fraction": -0.1},
            r"\[0, 1\)",
        ),
        (
            {
                "injection_draw": "population_proxy",
                "injection_draw_defensive_fraction": float("nan"),
            },
            r"\[0, 1\)",
        ),
        (
            {"injection_draw": "population_proxy", "redshift_sampling_grid": 64},
            "redshift_sampling_grid >= 4096",
        ),
        ({"observation_model": "noisy"}, "observation_model must be one of"),
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

    assert selection.metadata["injection_draw_defensive_fraction"] == pytest.approx(
        DEFAULT_POPULATION_PROXY_DEFENSIVE_FRACTION
    )
    assert selection.metadata["defensive_pairing"] == {
        "u_ratio": POPULATION_PROXY_DEFENSIVE_U_RATIO,
        "m1_source_max": POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX,
    }

    # The stored raw-draw density is exactly the defensive mixture: the model
    # density at the proxy with p_q replaced by (1 - eps) p_q + eps g, where g
    # is log-uniform in 1 - q on [r u_max, u_max] for m1_source below the cap.
    columns = _columns(selection.samples)
    log_model = np.asarray(MODEL(columns, proxy), dtype=float)
    z = np.asarray(MODEL.cosmology.z_of_dL(columns["luminosity_distance"]), dtype=float)
    m1 = columns["m1_detector"] / (1.0 + z)
    q = columns["q"]
    log_pq = np.asarray(
        mass_ratio_logpdf(q, m1, beta=proxy["beta_q"], mmin=proxy["mmin"], q_floor=MODEL.q_floor),
        dtype=float,
    )
    eps = config.injection_draw_defensive_fraction
    r = POPULATION_PROXY_DEFENSIVE_U_RATIO
    u = 1.0 - q
    u_max = 1.0 - np.maximum(MODEL.q_floor, proxy["mmin"] / m1)
    active = m1 < POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX
    g = np.where(active & (u >= r * u_max) & (u <= u_max), 1.0 / (u * math.log(1.0 / r)), 0.0)
    mixture_ratio = np.where(active, (1.0 - eps) + eps * g / np.exp(log_pq), 1.0)
    expected = log_model + np.log(mixture_ratio)
    np.testing.assert_allclose(selection.log_draw_density, expected, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(selection.log_draw_density))
    # The defensive component really was sampled (rows with q -> 1 at low mass).
    assert np.count_nonzero(active & (g > 0.0) & (u < 1e-3)) > 0

    # eps = 0 is exactly the pure proxy: the stored density is the model's.
    pure = generate_baseline_synthetic_dataset(
        seed=4321,
        config=replace(config, injection_draw_defensive_fraction=0.0),
        model=MODEL,
    ).selection
    np.testing.assert_array_equal(
        pure.log_draw_density,
        np.asarray(MODEL(_columns(pure.samples), proxy), dtype=float),
    )

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
        log_draw = population_proxy_log_draw_density(MODEL, samples, config)
        supported = np.isfinite(log_pop)
        assert supported.mean() > 0.999, hp
        assert np.all(np.isfinite(log_draw[supported])), hp


def test_population_proxy_importance_weights_are_unbiased_over_all_draws():
    proxy = DEFAULT_POPULATION_PROXY_HYPERPARAMETERS
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    rng = np.random.default_rng(0)
    n = 200_000
    samples = _columns(_proxy_draw(rng, n, config))
    log_draw = population_proxy_log_draw_density(MODEL, samples, config)
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
        assert default.observation_model == "truth_centered"
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


# --- Defensive pairing component (review finding: Pareto-2 weight tail) -----


def _mixture_q_cdf(q, m1, proxy, eps):
    """CDF of the population_proxy q | m1 conditional (defensive mixture)."""
    beta = proxy["beta_q"]
    q_min = np.maximum(MODEL.q_floor, proxy["mmin"] / m1)
    log_q = np.asarray(
        mass_ratio_logpdf(q, m1, beta=beta, mmin=proxy["mmin"], q_floor=MODEL.q_floor),
        dtype=float,
    )
    inv_norm = np.exp(log_q - beta * np.log(q))
    pure = (q ** (beta + 1.0) - q_min ** (beta + 1.0)) / (beta + 1.0) * inv_norm
    r = POPULATION_PROXY_DEFENSIVE_U_RATIO
    u_max = 1.0 - q_min
    u = np.clip(1.0 - q, r * u_max, u_max)
    defensive = np.log(u_max / u) / math.log(1.0 / r)
    eps_eff = np.where(m1 < POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX, eps, 0.0)
    return (1.0 - eps_eff) * pure + eps_eff * defensive


def test_population_proxy_q_draws_follow_the_declared_mixture():
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    proxy = config.injection_draw_hyperparameters
    eps = config.injection_draw_defensive_fraction
    draw = _proxy_draw(np.random.default_rng(12), 200_000, config)
    m1 = np.asarray(draw["m1_source"], dtype=float)
    q = np.asarray(draw["q"], dtype=float)
    low = m1 < POPULATION_PROXY_DEFENSIVE_M1_SOURCE_MAX
    assert low.sum() > 20_000 and (~low).sum() > 20_000
    for rows in (low, ~low):
        pit = _mixture_q_cdf(q[rows], m1[rows], proxy, eps)
        assert kstest(pit, "uniform").pvalue > 1e-3
    # A sampler that ignored the defensive component is rejected.
    wrong = _mixture_q_cdf(q[low], m1[low], proxy, 0.0)
    assert kstest(wrong, "uniform").pvalue < 1e-6


@pytest.mark.parametrize("target", ["truth", "median", "mmin_9.9"])
def test_defensive_component_bounds_pairing_corner_weights(target):
    """The p_q(target)/p_q(draw) ratio stays bounded as m1 -> mmin_target.

    For a target with mmin > mmin_proxy the pairing density q**beta/Z(m1)
    grows like 1/(1 - mmin/m1) at the support edge, which the pure proxy
    cannot follow (its weights have an infinite second moment). The defensive
    component bounds the ratio by ln(1/r)/eps.
    """
    hp = {
        "truth": DEFAULT_BASELINE_HYPERPARAMETERS,
        "median": PRIOR_MEDIAN,
        "mmin_9.9": dict(DEFAULT_BASELINE_HYPERPARAMETERS, mmin=9.9),
    }[target]
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    pure = replace(config, injection_draw_defensive_fraction=0.0)
    proxy = config.injection_draw_hyperparameters

    delta = np.logspace(-2.0, -5.5, 36)  # 1 - mmin/m1 at the support edge
    m1_source = hp["mmin"] / (1.0 - delta)
    z = np.full(delta.size, 0.2)
    samples = {
        "m1_detector": m1_source * 1.2,
        "q": hp["mmin"] / m1_source,  # the target's lowest allowed q
        "luminosity_distance": np.asarray(MODEL.cosmology.dL_of_z(z), dtype=float),
        "ra": np.full(delta.size, 1.0),
        "dec": np.full(delta.size, 0.1),
        "chi_eff": np.full(delta.size, 0.05),
    }
    z_back = np.asarray(MODEL.cosmology.z_of_dL(samples["luminosity_distance"]), dtype=float)
    m1_back = samples["m1_detector"] / (1.0 + z_back)
    q = samples["q"]
    log_target_q = np.asarray(
        mass_ratio_logpdf(q, m1_back, beta=hp["beta_q"], mmin=hp["mmin"], q_floor=MODEL.q_floor),
        dtype=float,
    )
    log_proxy_q = np.asarray(
        mass_ratio_logpdf(q, m1_back, beta=proxy["beta_q"], mmin=proxy["mmin"], q_floor=MODEL.q_floor),
        dtype=float,
    )
    assert np.all(np.isfinite(log_target_q))
    pure_ratio = np.exp(log_target_q - log_proxy_q)
    # log p_draw(pure) - log p_draw(mixture) = log p_q - log p_mix,q exactly.
    shift = population_proxy_log_draw_density(MODEL, samples, pure) - (
        population_proxy_log_draw_density(MODEL, samples, config)
    )
    mixture_ratio = pure_ratio * np.exp(shift)

    bound = math.log(1.0 / POPULATION_PROXY_DEFENSIVE_U_RATIO) / config.injection_draw_defensive_fraction
    assert pure_ratio.max() > 1.0e4
    assert mixture_ratio.max() <= 1.05 * bound
    assert np.all(mixture_ratio <= pure_ratio * (1.0 + 1e-12))


# --- Draw density evaluated with the baseline family (review finding) -------


def test_population_proxy_draw_density_is_the_baseline_density_for_any_model():
    from gwpop_search.grammar import baseline_model_spec
    from gwpop_search.grammar.mutations import DEFAULT_MUTATIONS, apply_mutation
    from gwpop_search.hbi.numpy_backend import evaluate_selection
    from gwpop_search.models import compile_model_spec

    mutation = {m.mutation_id: m for m in DEFAULT_MUTATIONS}["mass.family.powerlaw"]
    child = compile_model_spec(apply_mutation(baseline_model_spec(), mutation))
    config = _small(n_injections=20_000, injection_draw="population_proxy")
    with_child = generate_baseline_synthetic_dataset(seed=11, model=child, config=config)
    with_baseline = generate_baseline_synthetic_dataset(seed=11, model=MODEL, config=config)
    for name in with_child.selection.field_names:
        np.testing.assert_array_equal(
            with_child.selection.samples[name], with_baseline.selection.samples[name]
        )
    np.testing.assert_array_equal(
        with_child.selection.log_draw_density, with_baseline.selection.log_draw_density
    )
    assert with_child.selection.metadata["draw_density_model"] == MODEL.to_config()
    a = evaluate_selection(with_child.selection, MODEL, DEFAULT_BASELINE_HYPERPARAMETERS)
    b = evaluate_selection(with_baseline.selection, MODEL, DEFAULT_BASELINE_HYPERPARAMETERS)
    assert a.log_exposure == b.log_exposure


# --- Redshift sampler exactness floor (review finding) -----------------------


def test_redshift_grid_floor_keeps_the_sampler_bias_negligible():
    """max |E_sampler[w_z] - 1| over the kappa prior at the minimum accepted grid."""
    config = SyntheticSurveyConfig(
        injection_draw="population_proxy",
        redshift_sampling_grid=POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID,
    )
    with pytest.raises(ValueError, match="redshift_sampling_grid"):
        replace(config, redshift_sampling_grid=POPULATION_PROXY_MIN_REDSHIFT_SAMPLING_GRID - 1)
    kappa_proxy = config.injection_draw_hyperparameters["kappa"]

    def log_z(z, kappa):
        return np.asarray(
            redshift_rate_logpdf(
                z.ravel(),
                kappa=kappa,
                zmax=MODEL.zmax,
                cosmology=MODEL.cosmology,
                quadrature_order=MODEL.redshift_quadrature_order,
            ),
            dtype=float,
        ).reshape(z.shape)

    # The sampler inverts a trapezoid CDF linearly: piecewise-constant density.
    grid = np.linspace(1e-7, MODEL.zmax, config.redshift_sampling_grid)
    shape = np.asarray(MODEL.cosmology.dVc_dz(grid), dtype=float) * (1.0 + grid) ** (
        kappa_proxy - 1.0
    )
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (shape[1:] + shape[:-1]) * np.diff(grid))))
    cdf /= cdf[-1]
    low, high = grid[:-1], grid[1:]
    sub = 8
    edges = low[:, None] + (high - low)[:, None] * np.linspace(0.0, 1.0, sub + 1)[None, :]
    a, b = edges[:, :-1].ravel(), edges[:, 1:].ravel()
    nodes, weights = np.polynomial.legendre.leggauss(16)
    zq = (0.5 * (b - a))[:, None] * (nodes[None, :] + 1.0) + a[:, None]
    wq = (0.5 * (b - a))[:, None] * weights[None, :]
    density = np.repeat(np.diff(cdf) / (high - low), sub)[:, None]
    log_proxy = log_z(zq, kappa_proxy)
    bias = [
        abs(float(np.sum(density * np.exp(log_z(zq, kappa) - log_proxy) * wq)) - 1.0)
        for kappa in np.linspace(-6.0, 12.0, 10)
    ]
    assert max(bias) < 1.0e-6


# --- Immutable, hashable resolved proxy (review finding) ---------------------


def test_resolved_proxy_is_read_only_hashable_and_serializable():
    config = SyntheticSurveyConfig(injection_draw="population_proxy")
    proxy = config.injection_draw_hyperparameters
    assert isinstance(proxy, FrozenHyperparameters)
    for mutate in (
        lambda: proxy.__setitem__("mmin", 6.0),
        lambda: proxy.update(mmin=6.0),
        lambda: proxy.pop("mmin"),
        lambda: proxy.clear(),
        lambda: proxy.setdefault("x", 1.0),
        lambda: DEFAULT_POPULATION_PROXY_HYPERPARAMETERS.__setitem__("mmin", 6.0),
    ):
        with pytest.raises(TypeError, match="read-only"):
            mutate()
    assert config.injection_draw_hyperparameters["mmin"] == 2.0
    assert hash(config) == hash(SyntheticSurveyConfig(injection_draw="population_proxy"))
    assert pickle.loads(pickle.dumps(config)) == config
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()
    assert asdict(config)["injection_draw_hyperparameters"] == proxy


def test_pre_defensive_population_proxy_payloads_are_rejected():
    payload = SyntheticSurveyConfig(injection_draw="population_proxy").to_dict()
    payload.pop("injection_draw_defensive_fraction")
    with pytest.raises(ValueError, match="predates the defensive"):
        SyntheticSurveyConfig.from_dict(payload)
    payload["injection_draw_defensive_fraction"] = 0.0
    assert SyntheticSurveyConfig.from_dict(payload).injection_draw_defensive_fraction == 0.0


# --- Coverage hook for every consumer (review finding) -----------------------


def _proxy_selection():
    config = _small(n_injections=2_000, injection_draw="population_proxy")
    return generate_baseline_synthetic_dataset(seed=5, config=config, model=MODEL).selection


def test_require_population_proxy_coverage_checks_priors_and_points():
    selection = _proxy_selection()
    require_population_proxy_coverage(selection, priors=BASELINE_SYNTHETIC_PRIORS)
    require_population_proxy_coverage(selection, hyperparameters=DEFAULT_BASELINE_HYPERPARAMETERS)
    wide = dict(BASELINE_SYNTHETIC_PRIORS, mmin=PriorSpec("uniform", low=1.5, high=10.0))
    with pytest.raises(ValueError, match="smallest prior mmin"):
        require_population_proxy_coverage(selection, priors=wide, context="unit test")
    with pytest.raises(ValueError, match="below the proxy mmin"):
        require_population_proxy_coverage(
            selection, hyperparameters=dict(DEFAULT_BASELINE_HYPERPARAMETERS, mmin=1.8)
        )
    with pytest.raises(ValueError, match="exceeds the proxy mmax"):
        require_population_proxy_coverage(
            selection, hyperparameters=dict(DEFAULT_BASELINE_HYPERPARAMETERS, mmax=150.0)
        )
    with pytest.raises(ValueError, match="needs the priors"):
        require_population_proxy_coverage(selection)
    assert population_proxy_point_violations(
        DEFAULT_POPULATION_PROXY_HYPERPARAMETERS, {"alpha": 1.0}
    ) == (
        "hyperparameters have no 'mmin' entry; support cannot be verified",
        "hyperparameters have no 'mmax' entry; support cannot be verified",
    )
    # Uniform-box and real-data selections are not population proxies.
    box = generate_baseline_synthetic_dataset(seed=5, config=_small(), model=MODEL).selection
    require_population_proxy_coverage(box, priors=wide)


def test_population_proxy_rejects_an_uncovered_synthetic_truth():
    config = _small(injection_draw="population_proxy")
    with pytest.raises(ValueError, match="do not cover the synthetic truth"):
        generate_baseline_synthetic_dataset(
            seed=1,
            config=config,
            hyperparameters=dict(DEFAULT_BASELINE_HYPERPARAMETERS, mmin=1.5),
        )


def _wide_mmin_spec():
    from gwpop_search.grammar import PriorConfig, baseline_model_spec

    spec = baseline_model_spec()
    priors = dict(spec.priors)
    priors["mmin"] = PriorConfig("uniform", {"low": 1.0, "high": 10.0})
    return replace(spec, priors=priors)


def test_fidelity_evaluator_and_evidence_refuse_uncovered_models(tmp_path):
    from gwpop_search.inference.evidence_campaign import (
        EvidenceCampaignConfig,
        run_model_evidence_repeats,
    )
    from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
    from gwpop_search.search.scheduler import Fidelity

    dataset = generate_baseline_synthetic_dataset(
        seed=5, config=_small(n_injections=2_000, injection_draw="population_proxy"), model=MODEL
    )
    spec = _wide_mmin_spec()
    evaluator = DeterministicHBIEvaluator(dataset.posterior, dataset.selection)
    with pytest.raises(ValueError, match="smallest prior mmin"):
        evaluator.evaluate(spec, Fidelity.F0_SANITY, seed=1, run_dir=tmp_path / "f0")
    with pytest.raises(ValueError, match="smallest prior mmin"):
        run_model_evidence_repeats(
            tmp_path / "evidence",
            spec,
            dataset.posterior,
            dataset.selection,
            root_seed=1,
            config=EvidenceCampaignConfig(),
        )
    assert not (tmp_path / "evidence").exists()


def test_structured_scout_campaign_refuses_uncovered_base_point(tmp_path):
    from gwpop_search.scouts import (
        StructuredScoutInjection,
        default_scout_campaign_config,
        run_structured_scout_campaign,
    )

    with pytest.raises(ValueError, match="scout base hyperparameters"):
        run_structured_scout_campaign(
            tmp_path / "scouts",
            n_runs=8,
            root_seed=3,
            injection=StructuredScoutInjection("null", 0.0),
            survey_config=_small(injection_draw="population_proxy"),
            scout_config=default_scout_campaign_config("chi_eff", "q"),
            base_hyperparameters=dict(DEFAULT_BASELINE_HYPERPARAMETERS, mmin=1.5),
        )
    assert not (tmp_path / "scouts" / "run_000").exists()


def test_conditional_scout_refuses_uncovered_base_point(tmp_path):
    from gwpop_search.grammar import baseline_model_spec
    from gwpop_search.scouts import default_scout_campaign_config
    from gwpop_search.scouts.inference import run_conditional_hsgp_scout

    dataset = generate_baseline_synthetic_dataset(
        seed=5, config=_small(n_injections=2_000, injection_draw="population_proxy"), model=MODEL
    )
    scout = default_scout_campaign_config("chi_eff", "q")
    with pytest.raises(ValueError, match="scout base hyperparameters"):
        run_conditional_hsgp_scout(
            tmp_path / "scout",
            dataset.posterior,
            dataset.selection,
            base_spec=baseline_model_spec(),
            base_hyperparameters=dict(DEFAULT_BASELINE_HYPERPARAMETERS, mmin=1.5),
            hsgp_config=scout.hsgp,
            seed=1,
            config=scout.run,
        )
    assert not (tmp_path / "scout").exists()


# --- CLI injection count default (review finding) ----------------------------


def test_cli_injection_count_default_depends_on_the_draw(capsys):
    from gwpop_search.cli import _survey_config

    parser = build_parser()
    base = ["synthetic-campaign", "--root", "runs/y"]
    assert _survey_config(parser.parse_args(base)).n_injections == 20_000
    proxy = _survey_config(parser.parse_args(base + ["--injection-draw", "population_proxy"]))
    assert proxy.n_injections == RECOMMENDED_POPULATION_PROXY_N_INJECTIONS
    assert capsys.readouterr().err == ""
    explicit = _survey_config(
        parser.parse_args(
            base + ["--injection-draw", "population_proxy", "--n-injections", "20000"]
        )
    )
    assert explicit.n_injections == 20_000
    assert "below the recommended" in capsys.readouterr().err
    legacy = _survey_config(parser.parse_args(base + ["--n-injections", "5000"]))
    assert legacy.n_injections == 5_000
    assert capsys.readouterr().err == ""
    noisy = _survey_config(parser.parse_args(base + ["--observation-model", "noisy_observation"]))
    assert noisy.observation_model == "noisy_observation"


# --- Null identities (review finding) ----------------------------------------


def test_survey_v2_options_enter_the_synthetic_null_dataset_identity(monkeypatch, tmp_path):
    from gwpop_search.grammar import ModelGraph, baseline_model_spec
    from gwpop_search.nulls import search_replay

    legacy = _small()
    proxy = _small(injection_draw="population_proxy")
    proxy_eps0 = replace(proxy, injection_draw_defensive_fraction=0.0)
    noisy = _small(observation_model="noisy_observation")
    assert legacy.dataset_identity_suffix() == ""
    suffixes = {cfg.dataset_identity_suffix() for cfg in (proxy, proxy_eps0, noisy)}
    assert len(suffixes) == 3 and all(s.startswith(":survey-") for s in suffixes)

    captured = []

    class _Stop(Exception):
        pass

    class _Evaluator:
        def __init__(self, posterior, selection, *, config, dataset_identity):
            captured.append(dataset_identity)
            raise _Stop

    monkeypatch.setattr(search_replay, "DeterministicHBIEvaluator", _Evaluator)
    root = baseline_model_spec()
    graph = ModelGraph(root.model_hash, (root,), (), {root.model_hash: 0})
    for cfg in (legacy, proxy):
        with pytest.raises(_Stop):
            search_replay.run_baseline_null_search_replay(
                0,
                17,
                root=tmp_path,
                graph=graph,
                model_prior=None,
                execution_config=None,
                fidelity_config=None,
                survey_config=cfg,
            )
    assert captured == ["baseline-null:17", "baseline-null:17" + proxy.dataset_identity_suffix()]


def test_frozen_selection_nulls_reject_survey_v2_options():
    from gwpop_search.nulls.campaign import ExactNullCampaignConfig

    for survey in (
        SyntheticSurveyConfig(injection_draw="population_proxy"),
        SyntheticSurveyConfig(observation_model="noisy_observation"),
    ):
        with pytest.raises(ValueError, match="do not apply"):
            ExactNullCampaignConfig(survey=survey, data_mode="frozen_selection_resample")
        config = ExactNullCampaignConfig(survey=survey, data_mode="synthetic_survey")
        assert ExactNullCampaignConfig.from_dict(config.to_dict()) == config
