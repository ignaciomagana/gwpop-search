import numpy as np
from scipy.integrate import quad

from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.inference import prior_specs_from_model_spec
from gwpop_search.models import (
    DEFAULT_BASELINE_HYPERPARAMETERS,
    DeclarativeGwcatChiEffModel,
    FlatLambdaCDM,
    GwcatChiEffBBHModel,
    chi_eff_mixture_logpdf,
    mass_ratio_truncated_normal_logpdf,
    primary_mass_powerlaw_two_peak_logpdf,
    redshift_madau_dickinson_logpdf,
)


def _integral(logpdf, low, high, epsabs=2e-6):
    return quad(
        lambda x: float(np.exp(logpdf(x))),
        low,
        high,
        epsabs=epsabs,
        limit=250,
    )[0]


def _prior_center(prior):
    p = prior.parameters
    if prior.family == "uniform":
        return 0.5 * (p["low"] + p["high"])
    if prior.family == "log_uniform":
        return np.sqrt(p["low"] * p["high"])
    if prior.family == "normal":
        return p["loc"]
    raise AssertionError(prior.family)


def test_two_peak_primary_mass_normalizes():
    value = _integral(
        lambda x: primary_mass_powerlaw_two_peak_logpdf(
            x,
            alpha=3.0,
            mmin=5.0,
            mmax=90.0,
            peak1_fraction=0.2,
            peak1_mu=30.0,
            peak1_sigma=3.0,
            peak2_fraction=0.3,
            peak2_mu=55.0,
            peak2_sigma=6.0,
        ),
        5.0,
        90.0,
    )
    assert np.isclose(value, 1.0, rtol=5e-5)


def test_truncated_gaussian_q_normalizes_conditionally():
    for m1 in (8.0, 20.0, 60.0):
        qmin = max(0.05, 5.0 / m1)
        value = _integral(
            lambda q: mass_ratio_truncated_normal_logpdf(
                q,
                m1,
                mu=0.75,
                sigma=0.15,
                mmin=5.0,
                q_floor=0.05,
            ),
            qmin,
            1.0,
        )
        assert np.isclose(value, 1.0, rtol=5e-5)


def test_chieff_mixture_normalizes():
    value = _integral(
        lambda x: chi_eff_mixture_logpdf(
            x,
            mu1=-0.1,
            sigma1=0.15,
            mu2=0.3,
            sigma2=0.25,
            fraction=0.35,
        ),
        -1.0,
        1.0,
    )
    assert np.isclose(value, 1.0, rtol=5e-5)


def test_madau_dickinson_redshift_density_normalizes():
    cosmology = FlatLambdaCDM()
    value = quad(
        lambda z: float(
            np.exp(
                redshift_madau_dickinson_logpdf(
                    z,
                    a=3.0,
                    b=4.0,
                    z_turnover=1.5,
                    zmax=2.5,
                    cosmology=cosmology,
                    quadrature_order=128,
                )
            )
        ),
        1e-6,
        2.5,
        epsabs=2e-5,
        limit=250,
    )[0]
    assert np.isclose(value, 1.0, rtol=1e-4)


def test_declarative_baseline_exactly_matches_phase3_baseline_density():
    spec = baseline_model_spec()
    declarative = DeclarativeGwcatChiEffModel(spec)
    baseline = GwcatChiEffBBHModel()

    z = 0.4
    m1src = 35.0
    samples = {
        "m1_detector": np.asarray([m1src * (1.0 + z)]),
        "q": np.asarray([0.8]),
        "luminosity_distance": np.asarray(
            [float(baseline.cosmology.dL_of_z(z))]
        ),
        "ra": np.asarray([1.0]),
        "dec": np.asarray([0.2]),
        "chi_eff": np.asarray([0.1]),
    }
    got = np.asarray(declarative(samples, DEFAULT_BASELINE_HYPERPARAMETERS))
    expected = np.asarray(baseline(samples, DEFAULT_BASELINE_HYPERPARAMETERS))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-10)


def test_every_initial_graph_node_compiles_and_is_finite_at_prior_center():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=2,
        max_models=40,
    )
    cosmology = FlatLambdaCDM()
    z = 0.25
    m1src = 35.0
    samples = {
        "m1_detector": np.asarray([m1src * (1.0 + z)]),
        "q": np.asarray([0.8]),
        "luminosity_distance": np.asarray([float(cosmology.dL_of_z(z))]),
        "ra": np.asarray([1.0]),
        "dec": np.asarray([0.1]),
        "chi_eff": np.asarray([0.05]),
    }

    for spec in graph.nodes:
        model = DeclarativeGwcatChiEffModel(
            spec,
            cosmology=cosmology,
            redshift_quadrature_order=32,
        )
        hp = {
            name: _prior_center(prior)
            for name, prior in spec.priors.items()
        }
        value = float(model(samples, hp)[0])
        assert np.isfinite(value), spec.model_hash


def test_model_spec_priors_bridge_without_changing_prior_semantics():
    spec = baseline_model_spec()
    priors = prior_specs_from_model_spec(spec)

    assert set(priors) == set(spec.priors)
    assert priors["alpha"].family == "uniform"
    assert priors["alpha"].low == 0.0
    assert priors["alpha"].high == 8.0
    assert priors["chi_sigma"].family == "log_uniform"
