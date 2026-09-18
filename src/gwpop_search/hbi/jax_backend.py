"""JAX implementation of the HBI inner likelihood with NumPy-reference parity."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from ..data import validate_pair
from .common import density_required_fields, selection_log_factors
from .types import HBIConfig, PopulationLogDensity

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


def _pad_selection(
    selection, fields, *, chunk_size: int | None, use_observing_time: bool
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
    return PreparedSelection(
        samples=padded,
        log_draw_density=jnp.asarray(draw.reshape(n_chunks, size)),
        log_factor=jnp.asarray(fac.reshape(n_chunks, size)),
        mask=jnp.asarray(mask.reshape(n_chunks, size)),
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
