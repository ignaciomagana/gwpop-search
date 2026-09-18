import numpy as np
from scipy.integrate import quad

from gwpop_search.models import (
    DEFAULT_BASELINE_HYPERPARAMETERS,
    FlatLambdaCDM,
    GwcatChiEffBBHModel,
    chi_eff_logpdf,
    mass_ratio_logpdf,
    powerlaw_logpdf,
    primary_mass_broken_powerlaw_logpdf,
    primary_mass_powerlaw_peak_logpdf,
    redshift_rate_logpdf,
)


def _integral(logpdf, low, high, *, epsabs=2e-6):
    return quad(
        lambda x: float(np.exp(logpdf(x))),
        low,
        high,
        epsabs=epsabs,
        limit=200,
    )[0]


def test_powerlaw_normalizes_including_alpha_one():
    for alpha in (0.2, 1.0, 3.5):
        value = _integral(
            lambda x: powerlaw_logpdf(x, alpha=alpha, xmin=5.0, xmax=90.0),
            5.0,
            90.0,
        )
        assert np.isclose(value, 1.0, rtol=3e-5)


def test_powerlaw_peak_normalizes_at_mixture_edges_and_interior():
    for fraction in (0.0, 0.1, 0.7, 1.0):
        value = _integral(
            lambda x: primary_mass_powerlaw_peak_logpdf(
                x,
                alpha=3.0,
                mmin=5.0,
                mmax=90.0,
                peak_fraction=fraction,
                peak_mu=35.0,
                peak_sigma=4.0,
            ),
            5.0,
            90.0,
        )
        assert np.isclose(value, 1.0, rtol=3e-5)


def test_broken_powerlaw_normalizes():
    for alpha1, alpha2, break_fraction in ((1.0, 3.0, 0.4), (3.0, 1.0, 0.7)):
        value = _integral(
            lambda x: primary_mass_broken_powerlaw_logpdf(
                x,
                alpha1=alpha1,
                alpha2=alpha2,
                mmin=5.0,
                mmax=90.0,
                break_fraction=break_fraction,
            ),
            5.0,
            90.0,
        )
        assert np.isclose(value, 1.0, rtol=4e-5)


def test_mass_ratio_conditional_normalizes_for_multiple_primary_masses():
    for m1 in (6.0, 10.0, 30.0, 80.0):
        qmin = max(0.05, 5.0 / m1)
        for beta in (-1.0, 0.0, 3.0):
            value = _integral(
                lambda q: mass_ratio_logpdf(
                    q,
                    m1,
                    beta=beta,
                    mmin=5.0,
                    q_floor=0.05,
                ),
                qmin,
                1.0,
            )
            assert np.isclose(value, 1.0, rtol=4e-5)


def test_chi_eff_truncated_gaussian_normalizes():
    for mu, sigma in ((0.0, 0.2), (0.5, 0.3), (-0.8, 0.1)):
        value = _integral(
            lambda x: chi_eff_logpdf(x, mu=mu, sigma=sigma),
            -1.0,
            1.0,
        )
        assert np.isclose(value, 1.0, rtol=4e-5)


def test_redshift_rate_density_normalizes():
    cosmology = FlatLambdaCDM()
    for kappa in (-2.0, 0.0, 2.0, 5.0):
        value = quad(
            lambda z: float(
                np.exp(
                    redshift_rate_logpdf(
                        z,
                        kappa=kappa,
                        zmax=2.5,
                        cosmology=cosmology,
                        quadrature_order=128,
                    )
                )
            ),
            1e-6,
            2.5,
            epsabs=2e-5,
        )[0]
        assert np.isclose(value, 1.0, rtol=8e-5)


def test_cosmology_inverse_and_luminosity_distance_derivative():
    cosmology = FlatLambdaCDM()
    for z in (0.01, 0.1, 1.0, 2.5):
        d_l = float(cosmology.dL_of_z(z))
        assert np.isclose(float(cosmology.z_of_dL(d_l)), z, rtol=2e-5, atol=2e-6)

        eps = 1e-4
        finite_difference = (
            float(cosmology.dL_of_z(z + eps)) - float(cosmology.dL_of_z(z - eps))
        ) / (2.0 * eps)
        assert np.isclose(
            float(cosmology.ddL_dz(z)),
            finite_difference,
            rtol=5e-3,
        )


def test_baseline_transform_matches_explicit_source_to_detector_jacobian():
    model = GwcatChiEffBBHModel()
    hp = DEFAULT_BASELINE_HYPERPARAMETERS

    z = 0.35
    m1_source = 32.0
    q = 0.8
    chi_eff = 0.12
    d_l = float(model.cosmology.dL_of_z(z))
    m1_detector = m1_source * (1.0 + z)

    samples = {
        "m1_detector": np.asarray([m1_detector]),
        "q": np.asarray([q]),
        "luminosity_distance": np.asarray([d_l]),
        "ra": np.asarray([1.0]),
        "dec": np.asarray([0.2]),
        "chi_eff": np.asarray([chi_eff]),
    }
    got = float(model(samples, hp)[0])

    source_logp = (
        float(
            primary_mass_powerlaw_peak_logpdf(
                m1_source,
                alpha=hp["alpha"],
                mmin=hp["mmin"],
                mmax=hp["mmax"],
                peak_fraction=hp["peak_fraction"],
                peak_mu=hp["peak_mu"],
                peak_sigma=hp["peak_sigma"],
            )
        )
        + float(
            mass_ratio_logpdf(
                q,
                m1_source,
                beta=hp["beta_q"],
                mmin=hp["mmin"],
                q_floor=model.q_floor,
            )
        )
        + float(
            redshift_rate_logpdf(
                z,
                kappa=hp["kappa"],
                zmax=model.zmax,
                cosmology=model.cosmology,
                quadrature_order=model.redshift_quadrature_order,
            )
        )
        + float(chi_eff_logpdf(chi_eff, mu=hp["chi_mu"], sigma=hp["chi_sigma"]))
    )
    expected = (
        source_logp
        - np.log1p(z)
        - np.log(float(model.cosmology.ddL_dz(z)))
        - np.log(4.0 * np.pi)
    )
    assert np.isclose(got, expected, rtol=0.0, atol=2e-5)
