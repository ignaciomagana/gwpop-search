"""Normalized, autodiff-safe JAX population components for the BBH baseline."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import betainc, gammaln, log_ndtr, ndtr
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


def _log_normal_interval_mass(a, b):
    """Stable ``log(Phi(b) - Phi(a))`` for standardized bounds ``a < b``.

    ``ndtr(b) - ndtr(a)`` cancels catastrophically when the interval lies far in
    a tail: at ``a = 10`` it returns exactly zero (``-inf`` in the log) where the
    true value is about ``-21.9``, which turns a finite population density into
    spurious zero support. Both branches below are the same quantity written on
    the side where no cancellation occurs: for ``a >= 0`` the right tails
    ``Phi(-a) - Phi(-b)``, otherwise the left tails ``Phi(b) - Phi(a)``. The
    subtraction is done in log space with ``log1p(-exp(delta))`` and
    ``delta <= 0``; a degenerate interval gives ``-inf`` and is rejected by the
    caller's support mask rather than floored.
    """
    right = a >= 0.0
    hi = jnp.where(right, log_ndtr(-a), log_ndtr(b))
    lo = jnp.where(right, log_ndtr(-b), log_ndtr(a))
    delta = jnp.minimum(lo - hi, 0.0)
    return hi + jnp.log1p(-jnp.exp(delta))


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
    log_norm = _log_normal_interval_mass(a, b)
    valid_hyper = (
        valid_sigma
        & (high > low)
        & jnp.isfinite(log_norm)
    )
    safe_log_norm = jnp.where(valid_hyper, log_norm, 0.0)
    valid = valid_hyper & (x >= low) & (x <= high)
    logp = (
        -0.5 * z**2
        - jnp.log(safe_sigma)
        - 0.5 * LOG2PI
        - safe_log_norm
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


# ---------------------------------------------------------------------------
# Follow-up chi_eff components (opt-in; not used by any DEFAULT_MUTATIONS model).
# ---------------------------------------------------------------------------


def _student_t_log_kernel(z, nu):
    """Log density of the standard Student-t with ``nu`` degrees of freedom."""
    return (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * jnp.log(nu * jnp.pi)
        - 0.5 * (nu + 1.0) * jnp.log1p(z * z / nu)
    )


def _student_t_central(z, nu):
    """``P(0 < T < |z|)`` for a standard Student-t, via I_x(1/2, nu/2)."""
    z2 = z * z
    return 0.5 * betainc(0.5, 0.5 * nu, z2 / (nu + z2))


def _student_t_tail(z, nu):
    """``P(T > |z|)`` for a standard Student-t, via I_x(nu/2, 1/2)."""
    return 0.5 * betainc(0.5 * nu, 0.5, nu / (nu + z * z))


def _log_student_t_interval_mass(a, b, nu):
    """``log(F(b) - F(a))`` of a standard Student-t CDF for ``a < b``.

    Written without catastrophic cancellation from the regularized incomplete
    beta function: an interval straddling zero is the *sum* of two central
    masses ``P(0 < T < |a|) + P(0 < T < b)``; a one-sided interval is reflected
    to ``0 <= lo < hi`` and taken as the difference of whichever representation
    (central masses or upper tails) has the smaller minuend. Float64 accuracy
    of ``jax.scipy.special.betainc`` is ~1e-13 relative for ``nu <= 100``.
    """
    straddle = (a < 0.0) & (b > 0.0)
    central_sum = _student_t_central(a, nu) + _student_t_central(b, nu)

    lo = jnp.maximum(jnp.where(a >= 0.0, a, -b), 0.0)
    hi = jnp.maximum(jnp.where(a >= 0.0, b, -a), 0.0)
    # Keep the unselected one-sided branch away from z = 0, where d I_x/dx is
    # singular, so a masked branch cannot turn a location/scale gradient into NaN.
    lo = jnp.where(straddle, 1.0, lo)
    hi = jnp.where(straddle, 2.0, hi)
    c_lo = _student_t_central(lo, nu)
    c_hi = _student_t_central(hi, nu)
    t_lo = _student_t_tail(lo, nu)
    t_hi = _student_t_tail(hi, nu)
    one_sided = jnp.where(c_hi <= t_lo, c_hi - c_lo, t_lo - t_hi)

    mass = jnp.where(straddle, central_sum, one_sided)
    safe_mass = jnp.where(mass > 0.0, mass, 1.0)
    return jnp.where(mass > 0.0, jnp.log(safe_mass), -jnp.inf)


def truncated_student_t_logpdf(x, *, mu, sigma, nu, low, high):
    """Normalized location-scale Student-t truncated to [low, high].

    Gradient-free use only: ``jax.scipy.special.betainc`` has no derivative
    with respect to its shape parameter ``nu/2`` (the dynesty ladder does not
    differentiate the likelihood).
    """
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    nu = jnp.asarray(nu)

    valid_sigma = jnp.isfinite(sigma) & (sigma > 0.0)
    valid_nu = jnp.isfinite(nu) & (nu > 0.0)
    safe_sigma = jnp.where(valid_sigma, sigma, 1.0)
    safe_nu = jnp.where(valid_nu, nu, 1.0)
    a = (low - mu) / safe_sigma
    b = (high - mu) / safe_sigma
    z = (x - mu) / safe_sigma
    log_norm = _log_student_t_interval_mass(a, b, safe_nu)
    valid_hyper = valid_sigma & valid_nu & (high > low) & jnp.isfinite(log_norm)
    safe_log_norm = jnp.where(valid_hyper, log_norm, 0.0)
    valid = valid_hyper & (x >= low) & (x <= high)
    logp = _student_t_log_kernel(z, safe_nu) - jnp.log(safe_sigma) - safe_log_norm
    return jnp.where(valid, logp, -jnp.inf)


def chi_eff_student_t_logpdf(chi_eff, *, mu, sigma, nu):
    """Truncated Student-t effective-spin distribution on [-1, 1]."""
    return truncated_student_t_logpdf(
        chi_eff,
        mu=mu,
        sigma=sigma,
        nu=nu,
        low=-1.0,
        high=1.0,
    )


def chi_eff_logistic_mixture_logpdf(
    chi_eff,
    *,
    mu1,
    sigma1,
    mu2,
    sigma2,
    fraction_logit,
):
    """Two-component truncated-Gaussian chi_eff mixture with a per-sample weight.

    The weight of component 2 is ``sigmoid(fraction_logit)`` (broadcast against
    ``chi_eff``); log weights are formed with ``log_sigmoid`` so no clipping is
    needed. With a constant logit this equals :func:`chi_eff_mixture_logpdf`
    with ``fraction = sigmoid(fraction_logit)``.
    """
    eta = jnp.asarray(fraction_logit)
    log1 = chi_eff_logpdf(chi_eff, mu=mu1, sigma=sigma1)
    log2 = chi_eff_logpdf(chi_eff, mu=mu2, sigma=sigma2)
    safe1 = jnp.where(jnp.isfinite(log1), log1, 0.0)
    safe2 = jnp.where(jnp.isfinite(log2), log2, 0.0)
    safe_eta = jnp.where(jnp.isfinite(eta), eta, 0.0)
    mixture = jnp.logaddexp(
        jax.nn.log_sigmoid(-safe_eta) + safe1,
        jax.nn.log_sigmoid(safe_eta) + safe2,
    )
    valid = (
        jnp.isfinite(eta)
        & jnp.isfinite(log1)
        & jnp.isfinite(log2)
        & (chi_eff >= -1.0)
        & (chi_eff <= 1.0)
    )
    return jnp.where(valid, mixture, -jnp.inf)


# ===========================================================================
# v2 components: the LVK GWTC-4/5 default BBH family and its atoms.
#
# Ported from the gate-verified popres implementation
# (populations-research/popres/models_core.py, models_ext.py; gate G2a,
# gates/G2a_grids.md: dR/dm1 and dR/dq reproduce the GWTC-5.0 popsummary
# release to 4.4e-14). Conventions kept exactly:
#
# * Planck low-mass taper of gwpopulation (clip of the scaled mass to
#   [1e-6, 1 - 1e-6], hard window, S = 1 when delta = 0);
# * primary mixture [lam_pl BPL + sum_k lam_k N_[mlow_1, mmax](mu_k, sigma_k)]
#   S(m1; mlow_1, delta_m_1) / Z_m1, BPL continuous at the break with an
#   analytic normalisation on [mlow_1, mmax]; Z_m1 by trapz on the m1 nodes
#   (1 when delta_m_1 == 0);
# * p(q | m1) = PL(q; beta, qmin(m1), 1) S(q m1; mlow_2, delta_m_2) / Z_q(m1), Z_q
#   by trapz over the q nodes at every m1 node, interpolated linearly in ln m1
#   (LVK "geomspace" grid; index clipped to the end segments), evaluated in log
#   space (finite where Z_q underflows); Z_q = 1 when delta_m_2 == 0.
#   Known, inherited accuracy limit (accepted as LVK-faithful; review of
#   fe32657): near m1 -> mlow_2 the q support spans only a few q nodes and the
#   linear-in-Z interpolation error of Z_q is large pointwise; weighted by
#   p(m1) over prior draws it is small (median 7e-6, max 2.6% when mlow_1 ~
#   mlow_2 ~ 3 and delta_m_1 is small). The pilot reports it at the posterior.
#
# Generalisation (the only one): a population floor ``q_floor``. The pairing
# lower edge is qmin(m1) = max(q_floor, mlow_2 / m1) and the q nodes are
# linspace(q_floor, 1, n_q). With q_floor = 0.001 this is the LVK grid
# linspace(0.001, 1, 500) exactly, and qmin = mlow_2/m1 whenever
# mlow_2/m1 >= 0.001 (always for mlow_2 >= 3, m1 <= 300).
# ===========================================================================

NEG_INF = -jnp.inf
LOG_4PI = float(np.log(4.0 * np.pi))
_SQRT2 = float(np.sqrt(2.0))
_LOG_SQRT_PI_OVER_2 = float(0.5 * np.log(np.pi / 2.0))


def _log_powerlaw_integral_lvk(a1, log_low, log_high):
    """log of int_low^high x^(a1-1) dx = (high^a1 - low^a1)/a1 (popres, stable)."""
    span = log_high - log_low
    x = a1 * span
    small = jnp.abs(x) < 1e-8
    a1_safe = jnp.where(small, 1.0, a1)
    ratio = jnp.where(small, span * (1.0 + 0.5 * x), jnp.expm1(a1_safe * span) / a1_safe)
    return a1 * log_low + jnp.log(ratio)


def log_planck_taper(m, mmin, mmax, delta_m):
    """log of gwpopulation's smoothing window S(m; mmin, mmax, delta_m).

    S = expit(-(1/s - 1/(1-s))), s = clip((m - mmin)/delta_m, 1e-6, 1 - 1e-6),
    times the hard window mmin <= m <= mmax; S = 1 inside the window when
    delta_m == 0.
    """
    positive = delta_m > 0
    dm_safe = jnp.where(positive, delta_m, 1.0)
    s = jnp.clip((m - mmin) / dm_safe, 1e-6, 1.0 - 1e-6)
    exponent = 1.0 / s - 1.0 / (1.0 - s)
    log_window = jnp.where(positive, -jax.nn.softplus(exponent), 0.0)
    inside = (m >= mmin) & (m <= mmax)
    return jnp.where(inside, log_window, NEG_INF)


def trapz_weights(x) -> np.ndarray:
    """Trapezoid weights: ``trapz(y, x) == sum(w * y)`` (NumPy)."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def _trapz_w(y, w, axis=-1):
    shape = [1] * y.ndim
    shape[axis] = -1
    return jnp.sum(y * jnp.reshape(w, shape), axis=axis)


def _log_z_from_linear(z):
    pos = z > 0
    return jnp.where(pos, jnp.log(jnp.where(pos, z, 1.0)), NEG_INF)


def _log_trapz_shifted(lp, shift, w):
    """ln trapz over axis 0 of exp(lp) as c + ln sum_i w_i exp(lp_i - c) (popres)."""
    shift = jax.lax.stop_gradient(shift)
    none = jnp.isneginf(shift)
    shift = jnp.where(none, 0.0, shift)
    tot = _trapz_w(jnp.exp(lp - shift[None, :]), w, axis=0)
    pos = tot > 0
    return jnp.where(pos, shift + jnp.log(jnp.where(pos, tot, 1.0)), NEG_INF)


def _log_interp_regular(x, x0, dx, n, lz):
    """ln of the linear interpolant (regular grid, end segments extrapolated) of exp(lz)."""
    la, lb = lz[:-1], lz[1:]
    shift = jax.lax.stop_gradient(jnp.maximum(la, lb))
    shift = jnp.where(jnp.isneginf(shift), 0.0, shift)
    za = jnp.exp(la - shift)
    table = jnp.stack([za, (jnp.exp(lb - shift) - za) / dx, shift], axis=-1)
    idx = jnp.clip(jnp.floor((x - x0) / dx).astype(jnp.int32), 0, n - 2)
    seg = table[idx]
    u = seg[..., 0] + (x - (x0 + idx * dx)) * seg[..., 1]
    pos = u > 0
    return jnp.where(pos, seg[..., 2] + jnp.log(jnp.where(pos, u, 1.0)), NEG_INF)


def log_mix(log_terms):
    """log sum_k exp(t_k) of a list of equal-shape arrays; -inf where all are -inf."""
    terms = [jnp.asarray(t) for t in log_terms]
    mx = terms[0]
    for t in terms[1:]:
        mx = jnp.maximum(mx, t)
    none = jnp.isneginf(mx)
    ms = jnp.where(none, 0.0, mx)
    tot = sum(jnp.exp(t - ms) for t in terms)
    return jnp.where(none, NEG_INF, ms + jnp.log(jnp.where(none, 1.0, tot)))


def log_broken_power_law(m, log_m, alpha_1, alpha_2, break_mass, low, high):
    """gwpopulation double power law with the break at ``break_mass`` (popres).

    (m/m_b)^-alpha_1 on [low, m_b), (m/m_b)^-alpha_2 on [m_b, high], continuous,
    normalised analytically on [low, high].
    """
    log_mb = jnp.log(break_mass)
    log_i1 = log_mb + _log_powerlaw_integral_lvk(1.0 - alpha_1, jnp.log(low) - log_mb, 0.0)
    log_i2 = log_mb + _log_powerlaw_integral_lvk(1.0 - alpha_2, 0.0, jnp.log(high) - log_mb)
    log_norm = jnp.logaddexp(log_i1, log_i2)
    slope = jnp.where(m < break_mass, alpha_1, alpha_2)
    inside = (m >= low) & (m <= high)
    return jnp.where(inside, -slope * (log_m - log_mb) - log_norm, NEG_INF)


def log_single_power_law(m, log_m, alpha, low, high):
    """m^-alpha normalised on [low, high] (gwpopulation ``powerlaw`` with slope -alpha)."""
    ok = high > low
    low_s = jnp.where(ok, low, 0.5 * high)
    log_int = _log_powerlaw_integral_lvk(1.0 - alpha, jnp.log(low_s), jnp.log(high))
    inside = ok & (m >= low) & (m <= high)
    return jnp.where(inside, -alpha * log_m - log_int, NEG_INF)


class LVKMassGrid:
    """Static normalisation nodes of the v2 mass and pairing blocks.

    ``m1_grid="geomspace"`` is the LVK GWTC-4/5 convention (log-uniform nodes,
    Z_q interpolated linearly in ln m1); ``"linspace"`` is gwpopulation's.
    q nodes are ``linspace(q_floor, 1, n_q)``.
    """

    def __init__(self, *, mmin: float, mmax: float, n_m1: int, n_q: int,
                 q_floor: float, m1_grid: str = "geomspace"):
        if m1_grid not in ("geomspace", "linspace"):
            raise ValueError("m1_grid must be 'geomspace' or 'linspace'")
        if not 0.0 < mmin < mmax:
            raise ValueError("need 0 < mmin < mmax")
        if not 0.0 < q_floor < 1.0:
            raise ValueError("q_floor must lie in (0, 1)")
        self.mmin, self.mmax = float(mmin), float(mmax)
        self.n_m1, self.n_q = int(n_m1), int(n_q)
        self.q_floor = float(q_floor)
        self.m1_grid = m1_grid
        if m1_grid == "geomspace":
            m1s = np.geomspace(self.mmin, self.mmax, self.n_m1)
            self.x0 = float(np.log(self.mmin))
            self.dx = float((np.log(self.mmax) - np.log(self.mmin)) / (self.n_m1 - 1))
        else:
            m1s = np.linspace(self.mmin, self.mmax, self.n_m1)
            self.x0 = self.mmin
            self.dx = float(m1s[1] - m1s[0])
        qs = np.linspace(self.q_floor, 1.0, self.n_q)
        # Nodes are stored as NumPy float64 and converted where they are used,
        # so a model compiled before jax_enable_x64 is switched on does not keep
        # float32 grids (review of fe32657).
        self._np = {
            "m1s": np.asarray(m1s, dtype=np.float64),
            "log_m1s": np.log(m1s).astype(np.float64),
            "w_m1": trapz_weights(m1s).astype(np.float64),
            "qs": np.asarray(qs, dtype=np.float64),
            "log_qs": np.log(qs).astype(np.float64),
            "w_q": trapz_weights(qs).astype(np.float64),
        }
        self.log_q_floor = float(np.log(self.q_floor))

    @property
    def m1s(self):
        return jnp.asarray(self._np["m1s"])

    @property
    def log_m1s(self):
        return jnp.asarray(self._np["log_m1s"])

    @property
    def w_m1(self):
        return jnp.asarray(self._np["w_m1"])

    @property
    def qs(self):
        return jnp.asarray(self._np["qs"])

    @property
    def log_qs(self):
        return jnp.asarray(self._np["log_qs"])

    @property
    def w_q(self):
        return jnp.asarray(self._np["w_q"])

    def log_interp(self, m1, log_m1, lz_nodes):
        x = log_m1 if self.m1_grid == "geomspace" else m1
        return _log_interp_regular(x, self.x0, self.dx, self.n_m1, lz_nodes)


class BrokenPowerLawPeaksMass:
    """Primary mass: (broken) power law + truncated-Gaussian peaks, Planck taper.

    ``continuum`` is ``"broken"`` (alpha_1, alpha_2, m_break; the LVK BP2P
    continuum) or ``"single"`` (alpha). ``peaks`` are the peak labels ``L``
    with parameters ``mu_L``, ``sigma_L``. Weights enter as ``log_lam_<c>``
    for the components ``("pl",) + peaks`` (physical parameters; the
    declarative layer maps the Dirichlet coordinates). The continuum is
    normalised on [mlow_1, grid.mmax] and the Gaussians are truncated to
    [mlow_1, grid.mmax] (the fixed mmax = maximum_mass of the LVK runs).
    """

    def __init__(self, continuum: str, peaks: tuple[str, ...], grid: LVKMassGrid,
                 *, continuum_high: float | None = None):
        if continuum not in ("broken", "single"):
            raise ValueError("continuum must be 'broken' or 'single'")
        self.continuum = continuum
        self.peaks = tuple(peaks)
        self.components = ("pl",) + self.peaks
        self.grid = grid
        # The LVK var_cut-4 run normalises the power law to mmax = 300 while its
        # grid (and Gaussian truncation / taper window) stops at 200.
        self.continuum_high = grid.mmax if continuum_high is None else float(continuum_high)

    def log_component_terms(self, m, log_m, p):
        """[log lam_c + log p_c(m)] for every component (untapered, normalised)."""
        lo = p["mlow_1"]
        hi = self.grid.mmax
        if self.continuum == "broken":
            lpl = log_broken_power_law(m, log_m, p["alpha_1"], p["alpha_2"], p["m_break"],
                                       lo, self.continuum_high)
        else:
            lpl = log_single_power_law(m, log_m, p["alpha"], lo, self.continuum_high)
        terms = [p["log_lam_pl"] + lpl]
        for peak in self.peaks:
            lg = truncated_normal_logpdf(m, mu=p[f"mu_{peak}"], sigma=p[f"sigma_{peak}"], low=lo, high=hi)
            terms.append(p[f"log_lam_{peak}"] + lg)
        return terms

    def log_taper(self, m, p):
        return log_planck_taper(m, p["mlow_1"], self.grid.mmax, p["delta_m_1"])

    def log_norm(self, p):
        """log Z_m1 = log trapz over the m1 nodes (0 when delta_m_1 == 0)."""
        g = self.grid
        lp = log_mix(self.log_component_terms(g.m1s, g.log_m1s, p)) + self.log_taper(g.m1s, p)
        z = _trapz_w(jnp.exp(lp), g.w_m1)
        return jnp.where(p["delta_m_1"] != 0, _log_z_from_linear(z), 0.0)

    def log_prob_from_terms(self, m, terms, p):
        return log_mix(terms) + self.log_taper(m, p) - self.log_norm(p)

    def log_prob(self, m, log_m, p):
        return self.log_prob_from_terms(m, self.log_component_terms(m, log_m, p), p)


class TaperedPowerLawPairing:
    """p(q | m1) = PL(q; beta, qmin(m1), 1) S(q m1; mlow_2, delta_m_2) / Z_q(m1).

    ``form``:

    * ``"constant"``: one ``beta`` (the LVK default);
    * ``"logistic_log_m1"``: beta(m1) = beta_low + (beta_high - beta_low)
      expit(ln(m1 / beta_m_t) / beta_width), Z_q at each node with beta(m1_k)
      (popres ``MassDependentBetaPairing`` "logistic");
    * ``"per_mass_component"``: one ``beta_c`` per mass component (the LVK
      "Extended" pairing): p(q | m1) = sum_c w_c(m1) p(q | m1; beta_c), with
      w_c(m1) = lam_c p_c(m1) / sum_k lam_k p_k(m1) the component membership at
      m1, so the m1 marginal is unchanged and p(q | m1) is normalised.
    """

    def __init__(self, form: str, grid: LVKMassGrid, components: tuple[str, ...] = ()):
        if form not in ("constant", "logistic_log_m1", "per_mass_component"):
            raise ValueError(f"unknown pairing form {form!r}")
        if form == "per_mass_component" and not components:
            raise ValueError("per_mass_component pairing needs the mass components")
        self.form = form
        self.grid = grid
        self.components = tuple(components)

    def beta(self, log_m1, p):
        if self.form == "constant":
            return p["beta"] + 0.0 * log_m1
        t = jax.nn.sigmoid((log_m1 - jnp.log(p["beta_m_t"])) / p["beta_width"])
        return p["beta_low"] + (p["beta_high"] - p["beta_low"]) * t

    def _log_num(self, q, log_q, m1, log_m1, beta, p):
        lo2 = p["mlow_2"]
        log_lo = jnp.maximum(self.grid.log_q_floor, jnp.log(lo2) - log_m1)
        ok = log_lo < 0.0
        log_int = _log_powerlaw_integral_lvk(1.0 + beta, jnp.where(ok, log_lo, -1.0), 0.0)
        inside = ok & (q * m1 >= lo2) & (q >= self.grid.q_floor) & (q <= 1.0)
        lp = jnp.where(inside, beta * log_q - log_int, NEG_INF)
        return lp + log_planck_taper(m1 * q, lo2, m1, p["delta_m_2"])

    def log_norm_nodes(self, beta_nodes, p):
        """ln Z_q at the m1 nodes (0 where delta_m_2 == 0; -inf without support)."""
        g = self.grid
        lp = self._log_num(g.qs[:, None], g.log_qs[:, None], g.m1s[None, :], g.log_m1s[None, :],
                           beta_nodes[None, :], p)
        top = self._log_num(jnp.ones_like(g.m1s), jnp.zeros_like(g.m1s), g.m1s, g.log_m1s, beta_nodes, p)
        lz = _log_trapz_shifted(lp, top - jnp.maximum(-beta_nodes, 0.0) * g.log_qs[0], g.w_q)
        return jnp.where(p["delta_m_2"] != 0, lz, 0.0)

    def _log_conditional(self, q, log_q, m1, log_m1, beta_sample, beta_nodes, p):
        num = self._log_num(q, log_q, m1, log_m1, beta_sample, p)
        log_zq = self.grid.log_interp(m1, log_m1, self.log_norm_nodes(beta_nodes, p))
        ok = jnp.isfinite(log_zq)
        return jnp.where(ok, num - jnp.where(ok, log_zq, 0.0), NEG_INF)

    def log_prob(self, q, log_q, m1, log_m1, p, *, mass_terms=None):
        g = self.grid
        if self.form != "per_mass_component":
            return self._log_conditional(q, log_q, m1, log_m1, self.beta(log_m1, p),
                                         self.beta(g.log_m1s, p), p)
        if mass_terms is None:
            raise ValueError("per_mass_component pairing needs the mass component terms")
        log_total = log_mix(mass_terms)
        safe_total = jnp.where(jnp.isneginf(log_total), 0.0, log_total)
        parts = []
        for component, term in zip(self.components, mass_terms):
            b = p[f"beta_{component}"]
            cond = self._log_conditional(q, log_q, m1, log_m1, b + 0.0 * log_m1, b + 0.0 * g.log_m1s, p)
            parts.append(term - safe_total + cond)
        return jnp.where(jnp.isneginf(log_total), NEG_INF, log_mix(parts))


def madau_dickinson_log_psi(log1p_z, gamma, kappa, z_peak):
    """gwpopulation MadauDickinsonRedshift psi(z) with psi(0) = 1 (popres M6)."""
    log1p_zp = jnp.log1p(z_peak)
    return (gamma * log1p_z - jnp.logaddexp(0.0, kappa * (log1p_z - log1p_zp))
            + jnp.logaddexp(0.0, -kappa * log1p_zp))


def redshift_madau_dickinson_psi_logpdf(z, *, gamma, kappa, z_peak, zmax, cosmology,
                                        quadrature_order=96):
    """p(z) = psi(z) / (1+z) dVc/dz / V on [0, zmax] (gwpopulation form, psi(0) = 1)."""
    z = jnp.asarray(z)
    nodes, weights = _legendre_nodes(int(quadrature_order))
    zq = 0.5 * zmax * (jnp.asarray(nodes) + 1.0)
    wq = 0.5 * zmax * jnp.asarray(weights)
    log_shape_q = (jnp.log(cosmology.dVc_dz(zq))
                   + madau_dickinson_log_psi(jnp.log1p(zq), gamma, kappa, z_peak) - jnp.log1p(zq))
    log_norm = jax.scipy.special.logsumexp(log_shape_q + jnp.log(wq))
    safe_z = jnp.where(z >= 0.0, z, 0.0)
    log1p_z = jnp.log1p(safe_z)
    log_shape = (jnp.log(cosmology.dVc_dz(safe_z))
                 + madau_dickinson_log_psi(log1p_z, gamma, kappa, z_peak) - log1p_z)
    valid = (z >= 0.0) & (z <= zmax) & (z_peak >= 0.0) & jnp.isfinite(log_shape) & jnp.isfinite(log_norm)
    return jnp.where(valid, log_shape - log_norm, NEG_INF)


def _dvc_dz_float64(cosmology, z) -> np.ndarray:
    """``dVc/dz`` in NumPy float64 (independent of the JAX x64 flag).

    Mirrors :meth:`FlatLambdaCDM.dVc_dz` on its precomputed float64 grid;
    other cosmology objects fall back to their own (JAX) method.
    """
    z = np.asarray(z, dtype=np.float64)
    grid_z = getattr(cosmology, "_z_grid", None)
    grid_dl = getattr(cosmology, "_dL_grid", None)
    if grid_z is None or grid_dl is None:
        return np.asarray(cosmology.dVc_dz(jnp.asarray(z)), dtype=np.float64)
    from .cosmology import C_KM_S

    dc = np.interp(z, grid_z, grid_dl) / (1.0 + z)
    e = np.sqrt(cosmology.Om0 * (1.0 + z) ** 3 + (1.0 - cosmology.Om0))
    return 4.0 * np.pi * (C_KM_S / cosmology.H0) * dc**2 / e


class PowerLawRedshiftNormTable:
    """ln N(kappa) = ln int_0^zmax dVc/dz (1+z)^(kappa-1) dz on a fixed kappa grid.

    Used by the kappa(m1) redshift model, whose normalisation differs per sample.
    Values and derivatives d ln N / d kappa = E[ln(1+z)] are tabulated once
    (Gauss-Legendre in z, float64) and interpolated with a cubic Hermite
    spline; outside [kappa_low, kappa_high] the density is -inf.
    """

    def __init__(self, cosmology, zmax: float, *, quadrature_order: int = 96,
                 kappa_low: float = -40.0, kappa_high: float = 40.0, n: int = 8001):
        nodes, weights = _legendre_nodes(int(quadrature_order))
        zq = 0.5 * float(zmax) * (np.asarray(nodes) + 1.0)
        wq = 0.5 * float(zmax) * np.asarray(weights)
        log_dvc = np.log(_dvc_dz_float64(cosmology, zq))
        l1pz = np.log1p(zq)
        kappas = np.linspace(kappa_low, kappa_high, int(n))
        a = log_dvc[None, :] + (kappas[:, None] - 1.0) * l1pz[None, :] + np.log(wq)[None, :]
        amax = a.max(axis=1, keepdims=True)
        e = np.exp(a - amax)
        # NumPy float64 storage, converted at use (see LVKMassGrid)
        self._log_norm = (amax[:, 0] + np.log(e.sum(axis=1))).astype(np.float64)
        self._dlog_norm = ((e * l1pz[None, :]).sum(axis=1) / e.sum(axis=1)).astype(np.float64)
        self.kappa_low, self.kappa_high = float(kappa_low), float(kappa_high)
        self.h = float(kappas[1] - kappas[0])
        self.n = int(n)

    @property
    def log_norm(self):
        return jnp.asarray(self._log_norm)

    @property
    def dlog_norm(self):
        return jnp.asarray(self._dlog_norm)

    def __call__(self, kappa):
        t_all = (kappa - self.kappa_low) / self.h
        idx = jnp.clip(jnp.floor(t_all).astype(jnp.int32), 0, self.n - 2)
        t = t_all - idx
        y0, y1 = self.log_norm[idx], self.log_norm[idx + 1]
        d0, d1 = self.dlog_norm[idx] * self.h, self.dlog_norm[idx + 1] * self.h
        t2, t3 = t * t, t * t * t
        value = ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * d0
                 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * d1)
        inside = (kappa >= self.kappa_low) & (kappa <= self.kappa_high)
        return jnp.where(inside, value, jnp.nan)


def redshift_powerlaw_conditional_logpdf(z, kappa, *, zmax, cosmology, norm_table):
    """p(z | kappa) = dVc/dz (1+z)^(kappa-1) / N(kappa) with per-sample kappa."""
    z = jnp.asarray(z)
    safe_z = jnp.where(z >= 0.0, z, 0.0)
    log_norm = norm_table(kappa)
    val = jnp.log(cosmology.dVc_dz(safe_z)) + (kappa - 1.0) * jnp.log1p(safe_z) - log_norm
    valid = (z >= 0.0) & (z <= zmax) & jnp.isfinite(log_norm) & jnp.isfinite(val)
    return jnp.where(valid, val, NEG_INF)


def log_eps_skewnorm(x, mu, sigma, eps, low=-1.0, high=1.0):
    """epsilon-skew-normal (Mudholkar & Hutson 2000) truncated to [low, high] (popres M7).

    Widths sigma (1 + eps) below mu and sigma (1 - eps) above, continuous at
    mu, analytic normalisation; the form of the LVK 260512 release. Needs
    sigma > 0 and |eps| < 1. The normalisation

        Z = sqrt(2 pi) [s_L (Phi((min(high, mu) - mu)/s_L) - Phi((low - mu)/s_L))
                        + s_R (Phi((high - mu)/s_R) - Phi((max(low, mu) - mu)/s_R))]

    is popres's erf expression written with the tail-stable
    ``_log_normal_interval_mass``, so it stays finite when mu leaves [low, high]
    and equals the truncated Gaussian exactly at eps = 0.
    """
    s_l = sigma * (1.0 + eps)
    s_r = sigma * (1.0 - eps)
    ok = (sigma > 0) & (s_l > 0) & (s_r > 0)
    s_l = jnp.where(ok, s_l, 1.0)
    s_r = jnp.where(ok, s_r, 1.0)
    left_hi = jnp.minimum(high, mu)
    right_lo = jnp.maximum(low, mu)

    def side(a, b, s):
        has = b > a
        a_s = jnp.where(has, a, 0.0)
        b_s = jnp.where(has, b, 1.0)
        return jnp.where(has, jnp.log(s) + _log_normal_interval_mass(a_s, b_s), NEG_INF)

    log_left = side((low - mu) / s_l, (left_hi - mu) / s_l, s_l)
    log_right = side((right_lo - mu) / s_r, (high - mu) / s_r, s_r)
    log_z = 0.5 * LOG2PI + log_mix([log_left, log_right])
    s = jnp.where(x < mu, s_l, s_r)
    val = -0.5 * ((x - mu) / s) ** 2 - jnp.where(jnp.isfinite(log_z), log_z, 0.0)
    inside = ok & (x >= low) & (x <= high) & jnp.isfinite(log_z)
    return jnp.where(inside, val, NEG_INF)
