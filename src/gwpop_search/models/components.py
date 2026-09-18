"""Normalized JAX population components for the Phase-3 BBH baseline."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp, ndtr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "baseline population models require JAX: install gwpop-search[inference]"
    ) from exc


LOG2PI = float(np.log(2.0 * np.pi))


def _exprel(x):
    """Stable expm1(x)/x, including the x -> 0 limit."""
    x = jnp.asarray(x)
    small = jnp.abs(x) < 1e-5
    series = 1.0 + 0.5 * x + x**2 / 6.0 + x**3 / 24.0
    safe_x = jnp.where(small, 1.0, x)
    safe = jnp.expm1(safe_x) / safe_x
    return jnp.where(small, series, safe)


def _power_integral(lo, hi, exponent):
    """Integral of x**exponent from lo to hi, stable near exponent=-1."""
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)
    exponent = jnp.asarray(exponent)
    a = exponent + 1.0
    log_ratio = jnp.log(hi / lo)
    return jnp.power(lo, a) * log_ratio * _exprel(a * log_ratio)


def powerlaw_logpdf(x, *, alpha, xmin, xmax):
    """Normalized p(x) proportional to x**(-alpha) on [xmin, xmax]."""
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    norm = _power_integral(xmin, xmax, -alpha)
    valid_hyper = (xmin > 0.0) & (xmax > xmin) & jnp.isfinite(norm) & (norm > 0.0)
    valid = valid_hyper & (x >= xmin) & (x <= xmax)
    logp = -alpha * jnp.log(x) - jnp.log(norm)
    return jnp.where(valid, logp, -jnp.inf)


def truncated_normal_logpdf(x, *, mu, sigma, low, high):
    """Normalized normal distribution truncated to [low, high]."""
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    a = (low - mu) / sigma
    b = (high - mu) / sigma
    z = (x - mu) / sigma
    norm = ndtr(b) - ndtr(a)
    valid_hyper = (sigma > 0.0) & (high > low) & jnp.isfinite(norm) & (norm > 0.0)
    valid = valid_hyper & (x >= low) & (x <= high)
    logp = -0.5 * z**2 - jnp.log(sigma) - 0.5 * LOG2PI - jnp.log(norm)
    return jnp.where(valid, logp, -jnp.inf)


def primary_mass_powerlaw_peak_logpdf(
    m1,
    *,
    alpha,
    mmin,
    mmax,
    peak_fraction,
    peak_mu,
    peak_sigma,
):
    """Normalized power law + truncated Gaussian peak on [mmin, mmax]."""
    m1 = jnp.asarray(m1)
    f = jnp.asarray(peak_fraction)
    log_pl = powerlaw_logpdf(m1, alpha=alpha, xmin=mmin, xmax=mmax)
    log_peak = truncated_normal_logpdf(
        m1, mu=peak_mu, sigma=peak_sigma, low=mmin, high=mmax
    )
    terms = jnp.stack(
        (jnp.log1p(-f) + log_pl, jnp.log(f) + log_peak),
        axis=0,
    )
    mixture = logsumexp(terms, axis=0)
    valid_f = (f >= 0.0) & (f <= 1.0)
    return jnp.where(valid_f, mixture, -jnp.inf)


def primary_mass_broken_powerlaw_logpdf(
    m1,
    *,
    alpha1,
    alpha2,
    mmin,
    mmax,
    break_fraction,
):
    """Continuous normalized broken power law with a fractional break location."""
    m1 = jnp.asarray(m1)
    bf = jnp.asarray(break_fraction)
    mb = mmin + bf * (mmax - mmin)

    i1 = _power_integral(mmin, mb, -alpha1)
    continuity = jnp.power(mb, alpha2 - alpha1)
    i2 = continuity * _power_integral(mb, mmax, -alpha2)
    norm = i1 + i2

    low = jnp.power(m1, -alpha1)
    high = continuity * jnp.power(m1, -alpha2)
    shape = jnp.where(m1 <= mb, low, high)

    valid_hyper = (
        (mmin > 0.0)
        & (mmax > mmin)
        & (bf > 0.0)
        & (bf < 1.0)
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    valid = valid_hyper & (m1 >= mmin) & (m1 <= mmax)
    return jnp.where(valid, jnp.log(shape) - jnp.log(norm), -jnp.inf)


def mass_ratio_logpdf(q, m1, *, beta, mmin, q_floor=0.05):
    """Normalized q**beta conditional with m2=q*m1 >= mmin."""
    q = jnp.asarray(q)
    m1 = jnp.asarray(m1)
    qmin = jnp.maximum(jnp.asarray(q_floor), jnp.asarray(mmin) / m1)
    norm = _power_integral(qmin, 1.0, beta)

    valid_hyper = (
        (q_floor > 0.0)
        & (q_floor < 1.0)
        & (mmin > 0.0)
        & (qmin < 1.0)
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    valid = valid_hyper & (q >= qmin) & (q <= 1.0)
    logp = beta * jnp.log(q) - jnp.log(norm)
    return jnp.where(valid, logp, -jnp.inf)


@lru_cache(maxsize=16)
def _legendre_nodes(order: int):
    if order < 16:
        raise ValueError("quadrature_order must be >= 16")
    return np.polynomial.legendre.leggauss(int(order))


def redshift_rate_logpdf(z, *, kappa, zmax, cosmology, quadrature_order=96):
    """Normalized source redshift density proportional to dVc/dz*(1+z)^(kappa-1)."""
    z = jnp.asarray(z)
    nodes, weights = _legendre_nodes(int(quadrature_order))
    zq = 0.5 * zmax * (jnp.asarray(nodes) + 1.0)
    wq = 0.5 * zmax * jnp.asarray(weights)

    shape_q = cosmology.dVc_dz(zq) * jnp.power(1.0 + zq, kappa - 1.0)
    norm = jnp.sum(wq * shape_q)
    shape = cosmology.dVc_dz(z) * jnp.power(1.0 + z, kappa - 1.0)

    valid_hyper = (zmax > 0.0) & jnp.isfinite(norm) & (norm > 0.0)
    valid = valid_hyper & (z >= 0.0) & (z <= zmax) & (shape > 0.0)
    return jnp.where(valid, jnp.log(shape) - jnp.log(norm), -jnp.inf)


def chi_eff_logpdf(chi_eff, *, mu, sigma):
    """Truncated Gaussian effective-spin distribution on [-1, 1]."""
    return truncated_normal_logpdf(
        chi_eff,
        mu=mu,
        sigma=sigma,
        low=-1.0,
        high=1.0,
    )
