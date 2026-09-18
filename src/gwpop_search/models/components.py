"""Normalized, autodiff-safe JAX population components for the BBH baseline."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    import jax.numpy as jnp
    from jax.scipy.special import ndtr
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
    safe_lo = jnp.where(lo > 0.0, lo, 1.0)
    safe_hi = jnp.where(hi > 0.0, hi, 1.0)
    a = exponent + 1.0
    log_ratio = jnp.log(safe_hi / safe_lo)
    return jnp.power(safe_lo, a) * log_ratio * _exprel(a * log_ratio)


def powerlaw_logpdf(x, *, alpha, xmin, xmax):
    """Normalized p(x) proportional to x**(-alpha) on [xmin, xmax]."""
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    norm = _power_integral(xmin, xmax, -alpha)
    valid_hyper = (xmin > 0.0) & (xmax > xmin) & jnp.isfinite(norm) & (norm > 0.0)
    valid = valid_hyper & (x >= xmin) & (x <= xmax) & (x > 0.0)

    safe_x = jnp.where(x > 0.0, x, 1.0)
    safe_norm = jnp.where(valid_hyper, norm, 1.0)
    logp = -alpha * jnp.log(safe_x) - jnp.log(safe_norm)
    return jnp.where(valid, logp, -jnp.inf)


def truncated_normal_logpdf(x, *, mu, sigma, low, high):
    """Normalized normal distribution truncated to [low, high]."""
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)

    valid_sigma = jnp.isfinite(sigma) & (sigma > 0.0)
    safe_sigma = jnp.where(valid_sigma, sigma, 1.0)
    a = (low - mu) / safe_sigma
    b = (high - mu) / safe_sigma
    z = (x - mu) / safe_sigma
    norm = ndtr(b) - ndtr(a)
    valid_hyper = (
        valid_sigma
        & (high > low)
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    safe_norm = jnp.where(valid_hyper, norm, 1.0)
    valid = valid_hyper & (x >= low) & (x <= high)
    logp = (
        -0.5 * z**2
        - jnp.log(safe_sigma)
        - 0.5 * LOG2PI
        - jnp.log(safe_norm)
    )
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
        m1,
        mu=peak_mu,
        sigma=peak_sigma,
        low=mmin,
        high=mmax,
    )

    # Outside common mass support both component log densities are -inf. Feeding
    # an all--inf vector into logsumexp/logaddexp has undefined derivatives even
    # if a later where masks it. Evaluate a finite surrogate first, then apply
    # the exact support mask at the end.
    safe_pl = jnp.where(jnp.isfinite(log_pl), log_pl, 0.0)
    safe_peak = jnp.where(jnp.isfinite(log_peak), log_peak, 0.0)
    safe_f = jnp.clip(f, 1e-12, 1.0 - 1e-12)
    mixture = jnp.logaddexp(
        jnp.log1p(-safe_f) + safe_pl,
        jnp.log(safe_f) + safe_peak,
    )
    mixture = jnp.where(f <= 0.0, safe_pl, mixture)
    mixture = jnp.where(f >= 1.0, safe_peak, mixture)

    valid_f = jnp.isfinite(f) & (f >= 0.0) & (f <= 1.0)
    valid_mass = (
        (mmin > 0.0)
        & (mmax > mmin)
        & (m1 >= mmin)
        & (m1 <= mmax)
        & (m1 > 0.0)
    )
    return jnp.where(valid_f & valid_mass, mixture, -jnp.inf)


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
    continuity = jnp.power(jnp.where(mb > 0.0, mb, 1.0), alpha2 - alpha1)
    i2 = continuity * _power_integral(mb, mmax, -alpha2)
    norm = i1 + i2

    safe_m1 = jnp.where(m1 > 0.0, m1, 1.0)
    low = jnp.power(safe_m1, -alpha1)
    high = continuity * jnp.power(safe_m1, -alpha2)
    shape = jnp.where(m1 <= mb, low, high)

    valid_hyper = (
        (mmin > 0.0)
        & (mmax > mmin)
        & (bf > 0.0)
        & (bf < 1.0)
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    safe_norm = jnp.where(valid_hyper, norm, 1.0)
    valid = valid_hyper & (m1 >= mmin) & (m1 <= mmax) & (m1 > 0.0)
    logp = jnp.log(jnp.where(shape > 0.0, shape, 1.0)) - jnp.log(safe_norm)
    return jnp.where(valid, logp, -jnp.inf)


def mass_ratio_logpdf(q, m1, *, beta, mmin, q_floor=0.05):
    """Normalized q**beta conditional with m2=q*m1 >= mmin."""
    q = jnp.asarray(q)
    m1 = jnp.asarray(m1)

    safe_m1 = jnp.where(m1 > 0.0, m1, 1.0)
    qmin = jnp.maximum(jnp.asarray(q_floor), jnp.asarray(mmin) / safe_m1)
    norm = _power_integral(qmin, 1.0, beta)

    valid_hyper = (
        (q_floor > 0.0)
        & (q_floor < 1.0)
        & (mmin > 0.0)
        & (m1 > 0.0)
        & (qmin < 1.0)
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    safe_norm = jnp.where(valid_hyper, norm, 1.0)
    safe_q = jnp.where(q > 0.0, q, 1.0)
    valid = valid_hyper & (q >= qmin) & (q <= 1.0) & (q > 0.0)
    logp = beta * jnp.log(safe_q) - jnp.log(safe_norm)
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

    safe_z = jnp.where(z >= 0.0, z, 0.0)
    shape = cosmology.dVc_dz(safe_z) * jnp.power(
        1.0 + safe_z,
        kappa - 1.0,
    )

    valid_hyper = (zmax > 0.0) & jnp.isfinite(norm) & (norm > 0.0)
    safe_norm = jnp.where(valid_hyper, norm, 1.0)
    safe_shape = jnp.where(
        jnp.isfinite(shape) & (shape > 0.0),
        shape,
        1.0,
    )
    valid = (
        valid_hyper
        & (z >= 0.0)
        & (z <= zmax)
        & jnp.isfinite(shape)
        & (shape > 0.0)
    )
    logp = jnp.log(safe_shape) - jnp.log(safe_norm)
    return jnp.where(valid, logp, -jnp.inf)


def chi_eff_logpdf(chi_eff, *, mu, sigma):
    """Truncated Gaussian effective-spin distribution on [-1, 1]."""
    return truncated_normal_logpdf(
        chi_eff,
        mu=mu,
        sigma=sigma,
        low=-1.0,
        high=1.0,
    )



def primary_mass_powerlaw_two_peak_logpdf(
    m1,
    *,
    alpha,
    mmin,
    mmax,
    peak1_fraction,
    peak1_mu,
    peak1_sigma,
    peak2_fraction,
    peak2_mu,
    peak2_sigma,
):
    """Normalized power law plus two truncated Gaussian peaks."""
    m1 = jnp.asarray(m1)
    f1 = jnp.asarray(peak1_fraction)
    f2 = jnp.asarray(peak2_fraction)
    f0 = 1.0 - f1 - f2

    log_pl = powerlaw_logpdf(m1, alpha=alpha, xmin=mmin, xmax=mmax)
    log_p1 = truncated_normal_logpdf(
        m1, mu=peak1_mu, sigma=peak1_sigma, low=mmin, high=mmax
    )
    log_p2 = truncated_normal_logpdf(
        m1, mu=peak2_mu, sigma=peak2_sigma, low=mmin, high=mmax
    )

    safe_pl = jnp.where(jnp.isfinite(log_pl), log_pl, 0.0)
    safe_p1 = jnp.where(jnp.isfinite(log_p1), log_p1, 0.0)
    safe_p2 = jnp.where(jnp.isfinite(log_p2), log_p2, 0.0)

    sf0 = jnp.clip(f0, 1e-12, 1.0)
    sf1 = jnp.clip(f1, 1e-12, 1.0)
    sf2 = jnp.clip(f2, 1e-12, 1.0)
    mixture = jnp.logaddexp(
        jnp.log(sf0) + safe_pl,
        jnp.log(sf1) + safe_p1,
    )
    mixture = jnp.logaddexp(
        mixture,
        jnp.log(sf2) + safe_p2,
    )

    valid_f = (
        jnp.isfinite(f1)
        & jnp.isfinite(f2)
        & (f1 >= 0.0)
        & (f2 >= 0.0)
        & (f0 >= 0.0)
    )
    valid_mass = (
        (mmin > 0.0)
        & (mmax > mmin)
        & (m1 >= mmin)
        & (m1 <= mmax)
        & (m1 > 0.0)
    )
    return jnp.where(valid_f & valid_mass, mixture, -jnp.inf)


def mass_ratio_truncated_normal_logpdf(
    q,
    m1,
    *,
    mu,
    sigma,
    mmin,
    q_floor=0.05,
):
    """Truncated-Gaussian q conditional with the m2 >= mmin support."""
    q = jnp.asarray(q)
    m1 = jnp.asarray(m1)
    safe_m1 = jnp.where(m1 > 0.0, m1, 1.0)
    qmin = jnp.maximum(jnp.asarray(q_floor), jnp.asarray(mmin) / safe_m1)
    logp = truncated_normal_logpdf(
        q,
        mu=mu,
        sigma=sigma,
        low=qmin,
        high=1.0,
    )
    valid = (
        (m1 > 0.0)
        & (qmin < 1.0)
        & (q >= qmin)
        & (q <= 1.0)
    )
    return jnp.where(valid, logp, -jnp.inf)


def chi_eff_mixture_logpdf(
    chi_eff,
    *,
    mu1,
    sigma1,
    mu2,
    sigma2,
    fraction,
):
    """Two-component normalized truncated-Gaussian chi_eff mixture."""
    f = jnp.asarray(fraction)
    log1 = chi_eff_logpdf(chi_eff, mu=mu1, sigma=sigma1)
    log2 = chi_eff_logpdf(chi_eff, mu=mu2, sigma=sigma2)
    safe1 = jnp.where(jnp.isfinite(log1), log1, 0.0)
    safe2 = jnp.where(jnp.isfinite(log2), log2, 0.0)
    sf = jnp.clip(f, 1e-12, 1.0 - 1e-12)
    mixture = jnp.logaddexp(
        jnp.log1p(-sf) + safe1,
        jnp.log(sf) + safe2,
    )
    mixture = jnp.where(f <= 0.0, safe1, mixture)
    mixture = jnp.where(f >= 1.0, safe2, mixture)
    valid = (
        jnp.isfinite(f)
        & (f >= 0.0)
        & (f <= 1.0)
        & (chi_eff >= -1.0)
        & (chi_eff <= 1.0)
    )
    return jnp.where(valid, mixture, -jnp.inf)


def redshift_madau_dickinson_logpdf(
    z,
    *,
    a,
    b,
    z_turnover,
    zmax,
    cosmology,
    quadrature_order=96,
):
    """Normalized dVc/dz/(1+z) times a Madau-Dickinson-like rate history."""
    z = jnp.asarray(z)
    nodes, weights = _legendre_nodes(int(quadrature_order))
    zq = 0.5 * zmax * (jnp.asarray(nodes) + 1.0)
    wq = 0.5 * zmax * jnp.asarray(weights)

    safe_turnover = jnp.where(z_turnover > 0.0, z_turnover, 1.0)

    def rate_shape(x):
        one_plus = 1.0 + x
        scale = 1.0 + safe_turnover
        numerator = jnp.power(one_plus, a)
        denominator = 1.0 + jnp.power(one_plus / scale, a + b)
        return numerator / denominator

    shape_q = cosmology.dVc_dz(zq) * rate_shape(zq) / (1.0 + zq)
    norm = jnp.sum(wq * shape_q)

    safe_z = jnp.where(z >= 0.0, z, 0.0)
    shape = cosmology.dVc_dz(safe_z) * rate_shape(safe_z) / (1.0 + safe_z)

    valid_hyper = (
        (zmax > 0.0)
        & (a >= 0.0)
        & (b >= 0.0)
        & (z_turnover > 0.0)
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    safe_norm = jnp.where(valid_hyper, norm, 1.0)
    safe_shape = jnp.where(
        jnp.isfinite(shape) & (shape > 0.0),
        shape,
        1.0,
    )
    valid = (
        valid_hyper
        & (z >= 0.0)
        & (z <= zmax)
        & jnp.isfinite(shape)
        & (shape > 0.0)
    )
    return jnp.where(
        valid,
        jnp.log(safe_shape) - jnp.log(safe_norm),
        -jnp.inf,
    )
