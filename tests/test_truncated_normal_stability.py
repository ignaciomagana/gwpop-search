"""Stability of the truncated-normal normalization far in a tail (finding F8)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

from gwpop_search.models.components import (
    mass_ratio_truncated_normal_logpdf,
    truncated_normal_logpdf,
)


def _scipy_logpdf(x, mu, sigma, low, high):
    return truncnorm.logpdf(
        x, (low - mu) / sigma, (high - mu) / sigma, loc=mu, scale=sigma
    )


@pytest.mark.parametrize("a", [5.0, 8.0, 10.0, 15.0, 30.0])
def test_far_tail_interval_is_normalized_not_zero(a):
    # ndtr(b) - ndtr(a) underflows to exactly 0 for a >~ 9, which used to make
    # the density -inf (spurious zero support) instead of small but finite.
    low, high, x = a, a + 3.0, a + 0.7
    value = float(truncated_normal_logpdf(jnp.asarray(x), mu=0.0, sigma=1.0, low=low, high=high))
    assert np.isfinite(value)
    assert value == pytest.approx(_scipy_logpdf(x, 0.0, 1.0, low, high), abs=1e-9)


def test_regular_cases_match_scipy():
    rng = np.random.default_rng(20260919)
    mu = rng.uniform(-2.0, 2.0, 2000)
    sigma = np.exp(rng.uniform(-3.0, 1.0, 2000))
    low = mu - rng.uniform(0.5, 5.0, 2000) * sigma
    high = mu + rng.uniform(0.5, 5.0, 2000) * sigma
    x = low + (high - low) * rng.uniform(0.0, 1.0, 2000)
    got = np.asarray(
        truncated_normal_logpdf(
            jnp.asarray(x), mu=jnp.asarray(mu), sigma=jnp.asarray(sigma),
            low=jnp.asarray(low), high=jnp.asarray(high),
        )
    )
    assert np.max(np.abs(got - _scipy_logpdf(x, mu, sigma, low, high))) < 1e-12


def test_density_integrates_to_one_in_the_far_tail():
    low, high, mu, sigma = 12.0, 15.0, 0.0, 1.0
    grid = np.linspace(low, high, 20001)
    logp = np.asarray(truncated_normal_logpdf(jnp.asarray(grid), mu=mu, sigma=sigma, low=low, high=high))
    assert np.trapezoid(np.exp(logp), grid) == pytest.approx(1.0, rel=1e-6)


def test_gradients_are_finite_in_the_far_tail():
    def f(params):
        mu, sigma = params
        return truncated_normal_logpdf(jnp.asarray(12.5), mu=mu, sigma=sigma, low=12.0, high=15.0)

    grad = np.asarray(jax.grad(f)(jnp.asarray([0.0, 1.0])))
    assert np.all(np.isfinite(grad))


def test_support_and_invalid_hyperparameters_still_give_minus_inf():
    assert truncated_normal_logpdf(jnp.asarray(11.0), mu=0.0, sigma=1.0, low=0.0, high=10.0) == -jnp.inf
    assert truncated_normal_logpdf(jnp.asarray(1.0), mu=0.0, sigma=-1.0, low=0.0, high=10.0) == -jnp.inf
    assert truncated_normal_logpdf(jnp.asarray(1.0), mu=0.0, sigma=1.0, low=1.0, high=1.0) == -jnp.inf


def test_truncated_gaussian_pairing_stays_finite_where_it_used_to_underflow():
    # q_mu far below the m2 >= mmin edge with a narrow q_sigma: (qmin - q_mu)/q_sigma = 15.
    value = float(
        mass_ratio_truncated_normal_logpdf(
            jnp.asarray(0.6), jnp.asarray(20.0), mu=0.2, sigma=0.02, mmin=10.0, q_floor=0.05
        )
    )
    assert np.isfinite(value)
    expected = _scipy_logpdf(0.6, 0.2, 0.02, 0.5, 1.0)
    assert value == pytest.approx(expected, abs=1e-9)
