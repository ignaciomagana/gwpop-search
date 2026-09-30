"""JAX implementation of the HBI inner likelihood with NumPy-reference parity.

With ``HBIConfig.variance_taper`` set, the shape likelihood is multiplied by
the variance taper of :mod:`gwpop_search.hbi.taper` evaluated at the
Monte-Carlo variance of the shape log-likelihood estimator::

    sigma^2 = sum_i Var[ln I_i] + N^2 Var[xi] / xi^2

computed in the same pass as the event terms and the exposure, with the
estimator of the importance diagnostics
(:func:`gwpop_search.inference.dynesty_backend.build_importance_diagnostics`):
``Var[ln I_i] = S2_i / S1_i^2 - 1/n_i`` from the event weights and, for the
exposure ``xi = sum_k (T_k / N_k) sum_{j in k} w_j`` over campaigns ``k``,
``Var[xi]/xi^2 = sum_rows (e^f w)^2 / xi^2 - sum_k xi_k^2 / (N_k xi^2)``
(equal to ``sum_k (xi_k/xi)^2 Var[ln xi_k]``). The per-campaign sums use a
constant one-hot matrix product (no scatter), with one shared shift per chunk.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from ..data import validate_pair
from .common import density_required_fields, selection_log_factors
from .types import HBIConfig, PopulationLogDensity, RateTreatment

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.scipy.special import logsumexp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "JAX backend requires the 'inference' extra: pip install gwpop-search[inference]"
    ) from exc

@dataclass(frozen=True)
class PreparedPosterior:
    samples: dict[str, Any]
    log_ref_density: Any
    mask: Any
    counts: Any

@dataclass(frozen=True)
class PreparedSelection:
    samples: dict[str, Any]
    log_draw_density: Any
    log_factor: Any
    mask: Any
    # Only for the variance: one-hot campaign membership [n_chunks, size, K]
    # (zero on padding rows) and 1 / N_draw,k per campaign [K].
    campaign_onehot: Any = None
    campaign_inv_n_draw: Any = None


def _pad_events(posterior, fields) -> PreparedPosterior:
    counts = np.diff(posterior.offsets).astype(np.int64)
    n_events = posterior.n_events
    max_n = int(counts.max())
    mask = np.arange(max_n)[None, :] < counts[:, None]
    padded = {}
    for name in fields:
        values = posterior.samples[name]
        arr = np.empty((n_events, max_n), dtype=float)
        for i in range(n_events):
            sl = posterior.event_slice(i)
            v = np.asarray(values[sl], dtype=float)
            arr[i, : v.size] = v
            arr[i, v.size :] = v[0]
        padded[name] = jnp.asarray(arr)
    ref = np.zeros((n_events, max_n), dtype=float)
    for i in range(n_events):
        sl = posterior.event_slice(i)
        v = np.asarray(posterior.log_ref_density[sl], dtype=float)
        ref[i, : v.size] = v
        ref[i, v.size :] = v[0]
    return PreparedPosterior(
        samples=padded,
        log_ref_density=jnp.asarray(ref),
        mask=jnp.asarray(mask),
        counts=jnp.asarray(counts),
    )


def _campaign_membership(selection, total: int):
    """One-hot campaign membership of every (padded) row and ``1/N_draw,k``.

    ``N_draw,k`` is the campaign's ``n_draw`` or, when absent
    (estimator-ready products), its number of retained rows, as in the
    importance diagnostics. Campaigns without rows are dropped.
    """
    columns = []
    inv_n = []
    for campaign in selection.campaigns:
        rows = selection.rows_for_campaign(campaign.campaign_id)
        if rows.size == 0:
            continue
        column = np.zeros(total, dtype=float)
        column[rows] = 1.0
        columns.append(column)
        n_draw = rows.size if campaign.n_draw is None else int(campaign.n_draw)
        inv_n.append(1.0 / float(n_draw))
    return np.stack(columns, axis=1), np.asarray(inv_n, dtype=float)


def _pad_selection(
    selection,
    fields,
    *,
    chunk_size: int | None,
    use_observing_time: bool,
    with_variance: bool = False,
) -> PreparedSelection:
    n = selection.n_selected
    size = n if chunk_size is None else int(chunk_size)
    if size <= 0:
        raise ValueError("selection chunk_size must be positive")
    n_chunks = (n + size - 1) // size
    total = n_chunks * size
    mask = np.arange(total) < n
    factors = selection_log_factors(selection, use_observing_time=use_observing_time)
    padded = {}
    for name in fields:
        values = np.asarray(selection.samples[name], dtype=float)
        arr = np.empty(total, dtype=float)
        arr[:n] = values
        arr[n:] = values[0]
        padded[name] = jnp.asarray(arr.reshape(n_chunks, size))
    draw = np.empty(total, dtype=float)
    draw[:n] = selection.log_draw_density
    draw[n:] = selection.log_draw_density[0]
    fac = np.zeros(total, dtype=float)
    fac[:n] = factors
    onehot = inv_n_draw = None
    if with_variance:
        onehot, inv_n = _campaign_membership(selection, total)
        onehot = jnp.asarray(onehot.reshape(n_chunks, size, onehot.shape[1]))
        inv_n_draw = jnp.asarray(inv_n)
    return PreparedSelection(
        samples=padded,
        log_draw_density=jnp.asarray(draw.reshape(n_chunks, size)),
        log_factor=jnp.asarray(fac.reshape(n_chunks, size)),
        mask=jnp.asarray(mask.reshape(n_chunks, size)),
        campaign_onehot=onehot,
        campaign_inv_n_draw=inv_n_draw,
    )


def _event_terms(
    prepared: PreparedPosterior, log_density: PopulationLogDensity, hyperparameters
):
    log_pop = log_density(prepared.samples, hyperparameters)
    logw = jnp.where(
        prepared.mask,
        log_pop - prepared.log_ref_density,
        -jnp.inf,
    )
    return logsumexp(logw, axis=1) - jnp.log(prepared.counts)


def _selection_log_exposure(
    prepared: PreparedSelection, log_density: PopulationLogDensity, hyperparameters
):
    def body(acc, xs):
        sample_chunk, log_draw, log_factor, mask = xs
        log_pop = log_density(sample_chunk, hyperparameters)
        logw = jnp.where(mask, log_pop - log_draw + log_factor, -jnp.inf)
        chunk_lse = logsumexp(logw)
        return jnp.logaddexp(acc, chunk_lse), None
    init = jnp.asarray(-jnp.inf)
    final, _ = lax.scan(
        body,
        init,
        (
            prepared.samples,
            prepared.log_draw_density,
            prepared.log_factor,
            prepared.mask,
        ),
    )
    return final


def _event_terms_and_variance(
    prepared: PreparedPosterior, log_density: PopulationLogDensity, hyperparameters
):
    """Event terms (as :func:`_event_terms`) and ``Var[ln I_i]`` per event."""
    log_pop = log_density(prepared.samples, hyperparameters)
    logw = jnp.where(
        prepared.mask,
        log_pop - prepared.log_ref_density,
        -jnp.inf,
    )
    lse = logsumexp(logw, axis=1)
    lse2 = logsumexp(2.0 * logw, axis=1)
    has = jnp.isfinite(lse)
    safe_lse = jnp.where(has, lse, 0.0)
    safe_lse2 = jnp.where(has, lse2, 0.0)
    inv_ess = jnp.exp(safe_lse2 - 2.0 * safe_lse)
    variance = jnp.where(has, jnp.maximum(inv_ess - 1.0 / prepared.counts, 0.0), jnp.inf)
    return lse - jnp.log(prepared.counts), variance


def _selection_log_exposure_and_variance(
    prepared: PreparedSelection, log_density: PopulationLogDensity, hyperparameters
):
    """``ln xi`` (as :func:`_selection_log_exposure`, up to rounding) and ``Var[xi]/xi^2``."""
    n_campaigns = prepared.campaign_inv_n_draw.shape[0]

    def body(acc, xs):
        lse_acc, peak_acc, s1_acc, s2_acc, sk_acc = acc
        sample_chunk, log_draw, log_factor, mask, onehot = xs
        log_pop = log_density(sample_chunk, hyperparameters)
        logw = jnp.where(mask, log_pop - log_draw + log_factor, -jnp.inf)
        lse_acc = jnp.logaddexp(lse_acc, logsumexp(logw))
        peak = jnp.max(logw)
        new_peak = jnp.maximum(peak_acc, peak)
        shift = jnp.where(jnp.isfinite(new_peak), new_peak, 0.0)
        scaled = jnp.where(jnp.isfinite(logw), jnp.exp(logw - shift), 0.0)
        rescale = jnp.where(jnp.isfinite(peak_acc), jnp.exp(peak_acc - shift), 0.0)
        s1 = s1_acc * rescale + jnp.sum(scaled)
        s2 = s2_acc * rescale * rescale + jnp.sum(scaled * scaled)
        sk = sk_acc * rescale + scaled @ onehot
        return (lse_acc, new_peak, s1, s2, sk), None

    init = (
        jnp.asarray(-jnp.inf),
        jnp.asarray(-jnp.inf),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.zeros(n_campaigns),
    )
    (log_exposure, _, s1, s2, sk), _ = lax.scan(
        body,
        init,
        (
            prepared.samples,
            prepared.log_draw_density,
            prepared.log_factor,
            prepared.mask,
            prepared.campaign_onehot,
        ),
    )
    has = s1 > 0.0
    safe_s1 = jnp.where(has, s1, 1.0)
    fractions = sk / safe_s1
    relative = s2 / (safe_s1 * safe_s1) - jnp.sum(
        fractions * fractions * prepared.campaign_inv_n_draw
    )
    variance = jnp.where(has, jnp.maximum(relative, 0.0), jnp.inf)
    return log_exposure, variance


def build_terms_and_variance_function(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    *,
    config: HBIConfig | None = None,
    jit: bool = True,
):
    """``f(hyper) -> (event_terms, log_exposure, event_variance, selection_variance)``.

    ``event_terms`` and ``log_exposure`` equal :func:`build_terms_function`
    up to rounding (XLA fuses the extra reductions into the same graph; the
    untapered likelihood without ``variance_taper`` never takes this path);
    ``event_variance[i] = Var[ln I_i]`` and
    ``selection_variance = Var[xi]/xi^2``. The shape log-likelihood variance
    is ``sum(event_variance) + N^2 selection_variance``.
    """
    cfg = HBIConfig() if config is None else config
    fields = density_required_fields(log_density, posterior.basis)
    validate_pair(posterior, selection, fields)
    pe = _pad_events(posterior, fields)
    sel = _pad_selection(
        selection,
        fields,
        chunk_size=cfg.selection_chunk_size,
        use_observing_time=cfg.raw_selection_use_observing_time,
        with_variance=True,
    )

    def terms(hyperparameters):
        event_terms, event_variance = _event_terms_and_variance(
            pe, log_density, hyperparameters
        )
        log_exposure, selection_variance = _selection_log_exposure_and_variance(
            sel, log_density, hyperparameters
        )
        return event_terms, log_exposure, event_variance, selection_variance

    return jax.jit(terms) if jit else terms


def tapered_shape_value(event_terms, log_exposure, event_variance, selection_variance, *, n_events, taper):
    """``(ln L T, ln L, sigma^2, ln T)`` from the terms (``taper=None``: ``ln T = 0``).

    ``ln L`` follows :func:`build_shape_log_likelihood` (``-inf`` when an
    event or the exposure has no support). ``sigma^2`` is NaN-free: an
    event without support gives ``inf``.
    """
    valid = jnp.all(jnp.isfinite(event_terms)) & jnp.isfinite(log_exposure)
    safe_events = jnp.where(jnp.isfinite(event_terms), event_terms, 0.0)
    safe_exposure = jnp.where(jnp.isfinite(log_exposure), log_exposure, 0.0)
    value = jnp.where(valid, jnp.sum(safe_events) - n_events * safe_exposure, -jnp.inf)
    variance = jnp.sum(event_variance) + (n_events * n_events) * selection_variance
    variance = jnp.where(jnp.isnan(variance), jnp.inf, variance)
    if taper is None:
        log_t = jnp.zeros_like(variance)
    else:
        log_t = taper.log_taper_jax(variance)
    tapered = jnp.where(valid, value + log_t, -jnp.inf)
    return tapered, value, variance, log_t


def build_shape_log_likelihood_components(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    *,
    config: HBIConfig | None = None,
    jit: bool = True,
):
    """``f(hyper) -> (ln L T, ln L, sigma^2, ln T)`` for the configured taper.

    The taper is ``config.variance_taper``; with none configured ``ln T = 0``
    and ``ln L T = ln L`` (``sigma^2`` is still computed).
    """
    cfg = HBIConfig() if config is None else config
    if cfg.rate_treatment is not RateTreatment.SHAPE:
        raise ValueError("the variance taper is defined for the shape likelihood only")
    terms = build_terms_and_variance_function(
        posterior, selection, log_density, config=cfg, jit=False
    )
    n_events = posterior.n_events
    taper = cfg.variance_taper

    def components(hyperparameters):
        return tapered_shape_value(
            *terms(hyperparameters), n_events=n_events, taper=taper
        )

    return jax.jit(components) if jit else components


def build_terms_function(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    *,
    config: HBIConfig | None = None,
    jit: bool = True,
):
    """Build a differentiable function returning event terms and log exposure."""
    cfg = HBIConfig() if config is None else config
    fields = density_required_fields(log_density, posterior.basis)
    validate_pair(posterior, selection, fields)
    pe = _pad_events(posterior, fields)
    sel = _pad_selection(
        selection,
        fields,
        chunk_size=cfg.selection_chunk_size,
        use_observing_time=cfg.raw_selection_use_observing_time,
    )
    def terms(hyperparameters):
        return _event_terms(pe, log_density, hyperparameters), _selection_log_exposure(
            sel, log_density, hyperparameters
        )
    return jax.jit(terms) if jit else terms


def build_shape_log_likelihood(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    *,
    config: HBIConfig | None = None,
    jit: bool = True,
):
    cfg = HBIConfig() if config is None else config
    if cfg.variance_taper is not None:
        components = build_shape_log_likelihood_components(
            posterior, selection, log_density, config=cfg, jit=False
        )

        def tapered(hyperparameters):
            return components(hyperparameters)[0]

        return jax.jit(tapered) if jit else tapered
    terms = build_terms_function(
        posterior, selection, log_density, config=config, jit=False
    )
    n_events = posterior.n_events
    def likelihood(hyperparameters):
        event_terms, log_exposure = terms(hyperparameters)
        valid = jnp.all(jnp.isfinite(event_terms)) & jnp.isfinite(log_exposure)
        safe_events = jnp.where(jnp.isfinite(event_terms), event_terms, 0.0)
        safe_exposure = jnp.where(jnp.isfinite(log_exposure), log_exposure, 0.0)
        value = jnp.sum(safe_events) - n_events * safe_exposure
        return jnp.where(valid, value, -jnp.inf)
    return jax.jit(likelihood) if jit else likelihood


def build_poisson_log_likelihood(
    posterior,
    selection,
    log_density: PopulationLogDensity,
    *,
    config: HBIConfig | None = None,
    jit: bool = True,
):
    cfg = HBIConfig() if config is None else config
    if cfg.variance_taper is not None:  # pragma: no cover - HBIConfig refuses it
        raise ValueError("the variance taper is defined for the shape likelihood only")
    terms = build_terms_function(
        posterior, selection, log_density, config=config, jit=False
    )
    n_events = posterior.n_events
    def likelihood(hyperparameters, rate):
        event_terms, log_exposure = terms(hyperparameters)
        valid_rate = jnp.isfinite(rate) & (rate > 0)
        valid = (
            valid_rate
            & jnp.all(jnp.isfinite(event_terms))
            & jnp.isfinite(log_exposure)
        )
        safe_rate = jnp.where(valid_rate, rate, 1.0)
        safe_events = jnp.where(jnp.isfinite(event_terms), event_terms, 0.0)
        safe_exposure = jnp.where(jnp.isfinite(log_exposure), log_exposure, 0.0)
        value = (
            jnp.sum(safe_events)
            + n_events * jnp.log(safe_rate)
            - safe_rate * jnp.exp(safe_exposure)
        )
        return jnp.where(valid, value, -jnp.inf)
    return jax.jit(likelihood) if jit else likelihood
