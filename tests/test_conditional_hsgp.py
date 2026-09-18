import numpy as np

from gwpop_search.grammar import baseline_model_spec
from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS
from gwpop_search.scouts.conditional import (
    ConditionalHSGPConfig,
    ConditionalHSGPResidualModel,
    coefficient_priors,
)
from gwpop_search.scouts.hsgp import HSGPAxis


def _q_scout():
    config = ConditionalHSGPConfig(
        target="q",
        covariate="m1_source",
        target_axis=HSGPAxis("q", 0.05, 1.0, modes=4),
        covariate_axis=HSGPAxis("m1_source", 5.0, 90.0, modes=3),
        amplitude=0.7,
        target_length_scale=0.20,
        covariate_length_scale=18.0,
        quadrature_order=64,
    )
    return ConditionalHSGPResidualModel(
        baseline_model_spec(),
        DEFAULT_BASELINE_HYPERPARAMETERS,
        config,
    )


def _chi_q_scout():
    config = ConditionalHSGPConfig(
        target="chi_eff",
        covariate="q",
        target_axis=HSGPAxis("chi_eff", -1.0, 1.0, modes=5),
        covariate_axis=HSGPAxis("q", 0.05, 1.0, modes=3),
        amplitude=0.6,
        target_length_scale=0.30,
        covariate_length_scale=0.20,
        quadrature_order=64,
    )
    return ConditionalHSGPResidualModel(
        baseline_model_spec(),
        DEFAULT_BASELINE_HYPERPARAMETERS,
        config,
    )


def _coefficients(model):
    return {
        name: 0.20 * np.sin(index + 1.0)
        for index, name in enumerate(model.coefficient_names)
    }


def test_q_hsgp_conditional_normalizes_for_multiple_masses():
    model = _q_scout()
    coeffs = _coefficients(model)

    for m1 in (8.0, 20.0, 60.0):
        qmin = max(model.base_model.q_floor, DEFAULT_BASELINE_HYPERPARAMETERS["mmin"] / m1)
        q = np.linspace(qmin, 1.0, 5001)
        logp = np.asarray(
            model.conditional_logpdf_source(
                q,
                m1_source=np.full_like(q, m1),
                q=q,
                z=np.full_like(q, 0.25),
                hyperparameters=coeffs,
            )
        )
        integral = np.trapezoid(np.exp(logp), q)
        assert np.isclose(integral, 1.0, rtol=5e-4, atol=5e-4)


def test_chieff_q_hsgp_conditional_normalizes_for_multiple_q():
    model = _chi_q_scout()
    coeffs = _coefficients(model)

    chi = np.linspace(-1.0, 1.0, 5001)
    for q_value in (0.25, 0.6, 0.95):
        logp = np.asarray(
            model.conditional_logpdf_source(
                chi,
                m1_source=np.full_like(chi, 35.0),
                q=np.full_like(chi, q_value),
                z=np.full_like(chi, 0.25),
                hyperparameters=coeffs,
            )
        )
        integral = np.trapezoid(np.exp(logp), chi)
        assert np.isclose(integral, 1.0, rtol=5e-4, atol=5e-4)


def test_zero_hsgp_coefficients_recover_baseline_density():
    from gwpop_search.models import DeclarativeGwcatChiEffModel

    scout = _chi_q_scout()
    baseline = DeclarativeGwcatChiEffModel(baseline_model_spec())
    zero = {name: 0.0 for name in scout.coefficient_names}

    z = np.asarray([0.15, 0.35, 0.6])
    m1_source = np.asarray([20.0, 35.0, 55.0])
    d_l = np.asarray(baseline.cosmology.dL_of_z(z))
    samples = {
        "m1_detector": m1_source * (1.0 + z),
        "q": np.asarray([0.5, 0.75, 0.9]),
        "luminosity_distance": d_l,
        "ra": np.asarray([0.3, 1.2, 2.0]),
        "dec": np.asarray([0.1, -0.2, 0.4]),
        "chi_eff": np.asarray([-0.1, 0.05, 0.2]),
    }
    got = np.asarray(scout(samples, zero))
    expected = np.asarray(baseline(samples, DEFAULT_BASELINE_HYPERPARAMETERS))
    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=2e-7)


def test_hsgp_coefficients_have_standard_normal_inference_priors():
    model = _q_scout()
    priors = coefficient_priors(model.config)

    assert tuple(priors) == model.coefficient_names
    assert all(item.family == "normal" for item in priors.values())
    assert all(item.loc == 0.0 and item.scale == 1.0 for item in priors.values())


def test_hsgp_scout_has_finite_standardized_hbi_gradient():
    import jax
    import jax.numpy as jnp

    from gwpop_search.hbi import HBIConfig, build_jax_shape_log_likelihood
    from gwpop_search.inference.synthetic import (
        SyntheticSurveyConfig,
        generate_baseline_synthetic_dataset,
    )

    dataset = generate_baseline_synthetic_dataset(
        seed=19,
        config=SyntheticSurveyConfig(
            n_events=3,
            posterior_samples_per_event=16,
            n_injections=600,
            population_batch_size=128,
            redshift_sampling_grid=512,
        ),
    )
    model = _chi_q_scout()
    likelihood = build_jax_shape_log_likelihood(
        dataset.posterior,
        dataset.selection,
        model,
        config=HBIConfig(selection_chunk_size=128),
    )
    names = model.coefficient_names

    def objective(vector):
        hp = {name: vector[index] for index, name in enumerate(names)}
        return likelihood(hp)

    x = jnp.zeros(len(names))
    value = objective(x)
    gradient = jax.grad(objective)(x)

    assert np.isfinite(float(value))
    assert gradient.shape == x.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
