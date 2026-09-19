"""Batched evaluation of HBI event terms and per-sample importance weights.

Two evaluators share the standardized HBI estimator of
:mod:`gwpop_search.hbi` (nothing about the likelihood changes)::

    log L(Lambda) = sum_i log ell_i(Lambda) - N log A(Lambda)
    ell_i = (1/n_i) sum_j p_pop(theta_ij | Lambda) / pi_ij
    A     = sum_k (T_k / N_k) sum_{m in k} p_pop(theta_m | Lambda) / p_draw(theta_m)

* :class:`BatchedCatalogTerms` returns ``(log ell_i [m, N], log A [m])`` for a
  block of hyperparameter vectors. It is ``jax.jit(jax.vmap(...))`` of
  :func:`gwpop_search.hbi.jax_backend.build_terms_function`, the function
  the dynesty likelihood is built on, so ``sum_i log ell_i - N log A`` equals
  the sampled ``log L`` (parity is tested).
* :class:`CatalogWeightEvaluator` exposes the per-sample log importance
  weights ``log w_ij = log p_pop(theta_ij) - log pi_ij`` and the effective
  exposure weights ``log u_m = log p_pop(theta_m) - log p_draw(theta_m) +
  log(T_k/N_k)`` (``estimator_ready``: factor 0), with posterior-weighted
  accumulations used by :mod:`gwpop_search.analysis.edge_mc_error`. The data
  arrays are passed to the jitted functions as arguments (not captured as
  constants), so realizations of a catalog with equal padded shapes reuse one
  compilation.

Contract: ``-inf`` is zero population support and is propagated; NaN or
``+inf`` in a population density or importance weight raises
:class:`gwpop_search.hbi.PopulationDensityError` (never mapped to a value).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Mapping, Sequence

import numpy as np

from gwpop_search.hbi.common import PopulationDensityError

from ._common import as_int


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "the analysis evaluators require JAX: pip install 'gwpop-search[inference]'"
        ) from exc
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError(
            "the analysis evaluators require 64-bit JAX; set JAX_ENABLE_X64=true or call "
            "jax.config.update('jax_enable_x64', True) first"
        )
    return jax, jnp


def _pad_rows(X: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    m = X.shape[0]
    n_blocks = -(-m // batch_size)
    padded = np.empty((n_blocks * batch_size, X.shape[1]), dtype=np.float64)
    padded[:m] = X
    padded[m:] = X[0]
    return padded, n_blocks


def _check_block(X, ndim: int, names) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != ndim or X.shape[0] == 0:
        raise ValueError(f"hyperparameter block must have shape [m>0, {ndim}] ({names}); got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite hyperparameters passed to an analysis evaluator")
    return X


def _describe(names, X, rows, limit=4) -> str:
    parts = []
    for row in rows[:limit]:
        values = ", ".join(f"{name}={X[row, k]:.17g}" for k, name in enumerate(names))
        parts.append(f"row {int(row)}: {{{values}}}")
    return "; ".join(parts) + ("" if len(rows) <= limit else f" (+{len(rows) - limit} more)")


# ---------------------------------------------------------------------------
# Backend event terms
# ---------------------------------------------------------------------------


class BatchedCatalogTerms:
    """``f(X [m, ndim]) -> (event_log_likelihoods [m, N], log_exposure [m])``.

    ``jax.jit(jax.vmap)`` of the backend terms function
    (:func:`gwpop_search.hbi.jax_backend.build_terms_function`) evaluated in
    fixed ``[batch_size, ndim]`` blocks (rows are padded with copies of the
    first row). ``log L = event_log_likelihoods.sum(1) - N * log_exposure``
    reproduces the dynesty likelihood wherever both are finite.
    """

    def __init__(
        self,
        posterior,
        selection,
        population_model,
        names: Sequence[str],
        *,
        hbi_config=None,
        batch_size: int = 64,
    ):
        jax, jnp = _require_jax()
        from gwpop_search.hbi import HBIConfig, RateTreatment
        from gwpop_search.hbi.jax_backend import build_terms_function

        cfg = HBIConfig(selection_chunk_size=None) if hbi_config is None else hbi_config
        if cfg.rate_treatment is not RateTreatment.SHAPE:
            raise ValueError("analysis evaluators support the shape likelihood only")
        self.names = tuple(str(name) for name in names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("hyperparameter names must be unique and non-empty")
        self.ndim = len(self.names)
        self.batch_size = as_int("batch_size", batch_size, minimum=1)
        self.n_events = int(posterior.n_events)
        self.event_names = tuple(posterior.event_names)
        self.hbi_config = cfg
        terms = build_terms_function(posterior, selection, population_model, config=cfg, jit=False)
        names_ = self.names

        def single(x):
            hp = {name: x[k] for k, name in enumerate(names_)}
            event_terms, log_exposure = terms(hp)
            invalid = (
                jnp.any(jnp.isnan(event_terms) | (event_terms == jnp.inf))
                | jnp.isnan(log_exposure)
                | (log_exposure == jnp.inf)
            )
            return event_terms, log_exposure, invalid

        self._fn = jax.jit(jax.vmap(single))
        self._jax = jax
        self._jnp = jnp

    def __call__(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = _check_block(X, self.ndim, self.names)
        m = X.shape[0]
        padded, n_blocks = _pad_rows(X, self.batch_size)
        events = np.empty((padded.shape[0], self.n_events), dtype=np.float64)
        exposure = np.empty(padded.shape[0], dtype=np.float64)
        invalid = np.empty(padded.shape[0], dtype=bool)
        for block in range(n_blocks):
            sl = slice(block * self.batch_size, (block + 1) * self.batch_size)
            out = self._jax.device_get(self._fn(self._jnp.asarray(padded[sl])))
            events[sl] = np.asarray(out[0], dtype=np.float64)
            exposure[sl] = np.asarray(out[1], dtype=np.float64)
            invalid[sl] = np.asarray(out[2], dtype=bool)
        events, exposure, invalid = events[:m], exposure[:m], invalid[:m]
        if invalid.any():
            rows = np.flatnonzero(invalid)
            raise PopulationDensityError(
                "NaN/+inf event terms or selection exposure for "
                f"{rows.size} hyperparameter vector(s): {_describe(self.names, X, rows)}. "
                "-inf is allowed only for genuine zero population support."
            )
        return events, exposure

    def log_likelihood(self, X) -> np.ndarray:
        """``sum_i log ell_i - N log A`` with the backend's ``-inf`` convention."""
        events, exposure = self(X)
        return shape_log_likelihood_from_terms(events, exposure)


def shape_log_likelihood_from_terms(event_terms, log_exposure) -> np.ndarray:
    """``sum_i log ell_i - N log A``; ``-inf`` unless every term is finite."""
    event_terms = np.asarray(event_terms, dtype=np.float64)
    log_exposure = np.asarray(log_exposure, dtype=np.float64)
    n_events = event_terms.shape[-1]
    valid = np.all(np.isfinite(event_terms), axis=-1) & np.isfinite(log_exposure)
    with np.errstate(invalid="ignore"):
        value = event_terms.sum(axis=-1) - n_events * log_exposure
    return np.where(valid, value, -np.inf)


# ---------------------------------------------------------------------------
# Padded catalog arrays and per-sample weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PaddedCatalog:
    """PE and selection arrays laid out for per-sample weight evaluation.

    Events are ``[N, n_max]`` with a boolean mask (padding repeats the first
    sample and is masked out). Selection rows are ``[M_pad]`` with a mask and
    an integer campaign index; ``campaign_onehot`` is ``[M_pad, K]`` (zero rows
    for padding). ``campaign_n_draw[k]`` is the campaign's ``n_draw`` when
    known and otherwise its number of retained rows, which is the convention
    of :func:`gwpop_search.hbi.common.importance_diagnostics`.

    ``raw_selection_use_observing_time`` records the HBI-configuration field
    that shaped ``sel_log_factor`` (the per-campaign ``log(T_k / N_k)``), so a
    consumer can refuse a catalog padded under a different convention from the
    likelihood it is analysing.
    """

    fields: tuple[str, ...]
    event_names: tuple[str, ...]
    pe_samples: Mapping[str, np.ndarray]
    pe_log_ref: np.ndarray
    pe_mask: np.ndarray
    pe_counts: np.ndarray
    sel_samples: Mapping[str, np.ndarray]
    sel_log_draw: np.ndarray
    sel_log_factor: np.ndarray
    sel_mask: np.ndarray
    sel_campaign: np.ndarray
    campaign_onehot: np.ndarray
    campaign_ids: tuple[str, ...]
    campaign_n_draw: np.ndarray
    n_selected: int
    selection_rows_per_campaign: tuple[np.ndarray, ...]
    raw_selection_use_observing_time: bool = True

    @property
    def n_events(self) -> int:
        return len(self.event_names)

    @property
    def n_max(self) -> int:
        return int(self.pe_mask.shape[1])

    @property
    def m_pad(self) -> int:
        return int(self.sel_mask.shape[0])

    def device_arrays(self) -> dict[str, object]:
        """The arrays passed to the jitted functions."""
        return {
            "pe_samples": dict(self.pe_samples),
            "pe_log_ref": self.pe_log_ref,
            "pe_mask": self.pe_mask,
            "pe_counts": self.pe_counts.astype(np.float64),
            "sel_samples": dict(self.sel_samples),
            "sel_log_draw": self.sel_log_draw,
            "sel_log_factor": self.sel_log_factor,
            "sel_mask": self.sel_mask,
            "campaign_onehot": self.campaign_onehot,
            "campaign_n_draw": self.campaign_n_draw,
        }


def pad_catalog(
    posterior,
    selection,
    population_model,
    *,
    hbi_config=None,
    pe_capacity: int | None = None,
    selection_capacity: int | None = None,
) -> PaddedCatalog:
    """Lay out ``posterior``/``selection`` for :class:`CatalogWeightEvaluator`.

    ``pe_capacity``/``selection_capacity`` pad to fixed sizes (at least the
    data size) so that catalogs of different sizes share compiled functions.
    """
    from gwpop_search.data import validate_pair
    from gwpop_search.hbi import HBIConfig
    from gwpop_search.hbi.common import density_required_fields, selection_log_factors

    cfg = HBIConfig(selection_chunk_size=None) if hbi_config is None else hbi_config
    fields = density_required_fields(population_model, posterior.basis)
    validate_pair(posterior, selection, fields)
    counts = np.diff(posterior.offsets).astype(np.int64)
    n_events = posterior.n_events
    n_max = int(counts.max())
    if pe_capacity is not None:
        pe_capacity = as_int("pe_capacity", pe_capacity, minimum=1)
        if pe_capacity < n_max:
            raise ValueError(f"pe_capacity {pe_capacity} < largest event sample count {n_max}")
        n_max = pe_capacity
    mask = np.arange(n_max)[None, :] < counts[:, None]
    pe_samples = {}
    for name in fields:
        values = np.asarray(posterior.samples[name], dtype=np.float64)
        arr = np.empty((n_events, n_max), dtype=np.float64)
        for i in range(n_events):
            v = values[posterior.event_slice(i)]
            arr[i, : v.size] = v
            arr[i, v.size :] = v[0]
        pe_samples[name] = arr
    log_ref = np.zeros((n_events, n_max), dtype=np.float64)
    for i in range(n_events):
        v = np.asarray(posterior.log_ref_density[posterior.event_slice(i)], dtype=np.float64)
        log_ref[i, : v.size] = v
        log_ref[i, v.size :] = v[0]

    n = int(selection.n_selected)
    m_pad = n
    if selection_capacity is not None:
        selection_capacity = as_int("selection_capacity", selection_capacity, minimum=1)
        if selection_capacity < n:
            raise ValueError(f"selection_capacity {selection_capacity} < selected rows {n}")
        m_pad = selection_capacity
    sel_mask = np.arange(m_pad) < n
    sel_samples = {}
    for name in fields:
        values = np.asarray(selection.samples[name], dtype=np.float64)
        arr = np.empty(m_pad, dtype=np.float64)
        arr[:n] = values
        arr[n:] = values[0]
        sel_samples[name] = arr
    log_draw = np.empty(m_pad, dtype=np.float64)
    log_draw[:n] = selection.log_draw_density
    log_draw[n:] = selection.log_draw_density[0]
    factors = selection_log_factors(selection, use_observing_time=cfg.raw_selection_use_observing_time)
    log_factor = np.zeros(m_pad, dtype=np.float64)
    log_factor[:n] = factors
    campaign_ids, n_draw, rows_per_campaign = [], [], []
    campaign_index = np.zeros(m_pad, dtype=np.int64)
    for campaign in selection.campaigns:
        rows = selection.rows_for_campaign(campaign.campaign_id)
        if rows.size == 0:
            continue  # the NumPy reference skips empty campaigns
        k = len(campaign_ids)
        campaign_ids.append(campaign.campaign_id)
        n_draw.append(float(rows.size if campaign.n_draw is None else int(campaign.n_draw)))
        campaign_index[rows] = k
        rows_per_campaign.append(np.asarray(rows, dtype=np.int64))
    onehot = np.zeros((m_pad, len(campaign_ids)), dtype=np.float64)
    onehot[np.arange(n), campaign_index[:n]] = 1.0
    return PaddedCatalog(
        fields=tuple(fields),
        event_names=tuple(posterior.event_names),
        pe_samples=pe_samples,
        pe_log_ref=log_ref,
        pe_mask=mask,
        pe_counts=counts,
        sel_samples=sel_samples,
        sel_log_draw=log_draw,
        sel_log_factor=log_factor,
        sel_mask=sel_mask,
        sel_campaign=campaign_index,
        campaign_onehot=onehot,
        campaign_ids=tuple(campaign_ids),
        campaign_n_draw=np.asarray(n_draw, dtype=np.float64),
        n_selected=n,
        selection_rows_per_campaign=tuple(rows_per_campaign),
        raw_selection_use_observing_time=bool(cfg.raw_selection_use_observing_time),
    )


def _single_log_weights(xp, population_model, names, x, data):
    """Per-sample log weights of one hyperparameter vector (``xp`` = numpy or jax.numpy)."""
    hp = {name: x[k] for k, name in enumerate(names)}
    raw_e = population_model(data["pe_samples"], hp) - data["pe_log_ref"]
    raw_s = (
        population_model(data["sel_samples"], hp) - data["sel_log_draw"] + data["sel_log_factor"]
    )
    invalid = xp.any(data["pe_mask"] & (xp.isnan(raw_e) | (raw_e == xp.inf))) | xp.any(
        data["sel_mask"] & (xp.isnan(raw_s) | (raw_s == xp.inf))
    )
    lw_e = xp.where(data["pe_mask"], raw_e, -xp.inf)
    lu = xp.where(data["sel_mask"], raw_s, -xp.inf)
    return lw_e, lu, invalid


def _logsumexp(xp, values, axis):
    peak = xp.max(values, axis=axis, keepdims=True)
    safe = xp.where(xp.isfinite(peak), peak, 0.0)
    total = xp.sum(xp.exp(values - safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = xp.log(total) + safe
    return xp.squeeze(out, axis=axis)


def weight_moments(xp, lw_e, lu, data, W):
    """Posterior-weighted accumulations of normalized weights for one block.

    ``lw_e`` ``[B, N, n_max]`` and ``lu`` ``[B, M]`` are log weights at ``B``
    hyperparameter vectors with posterior weights ``W`` ``[B]`` (0 for
    padding). Returns the weighted sums needed by the Monte-Carlo error
    formulas plus per-vector diagnostics (see
    :mod:`gwpop_search.analysis.edge_mc_error` for the definitions).
    """
    n_events = lw_e.shape[1]
    lse_e = _logsumexp(xp, lw_e, 2)  # [B, N]
    ok_e = xp.isfinite(lse_e)
    shift_e = xp.where(ok_e, lse_e, 0.0)
    omega_e = xp.where(ok_e[..., None], xp.exp(lw_e - shift_e[..., None]), 0.0)
    lse_s = _logsumexp(xp, lu, 1)  # [B]
    ok_s = xp.isfinite(lse_s)
    shift_s = xp.where(ok_s, lse_s, 0.0)
    omega_s = xp.where(ok_s[:, None], xp.exp(lu - shift_s[:, None]), 0.0)
    frac = omega_s @ data["campaign_onehot"]  # [B, K]
    s2_e = xp.sum(omega_e * omega_e, axis=2)  # [B, N] = 1/ESS_i
    s2_s = xp.sum(omega_s * omega_s, axis=1)  # [B] = 1/ESS_sel
    f2 = xp.sum(frac * frac / data["campaign_n_draw"], axis=1)  # [B]
    sigma_e2 = s2_e - 1.0 / data["pe_counts"]
    sigma_a2 = s2_s - f2
    variance = xp.sum(sigma_e2, axis=1) + float(n_events) ** 2 * sigma_a2
    supported = xp.all(ok_e, axis=1) & ok_s
    log_like = xp.where(
        supported,
        xp.sum(shift_e - xp.log(data["pe_counts"]), axis=1) - n_events * shift_s,
        -xp.inf,
    )
    W2 = W * W
    return {
        "omega_e_sum": xp.einsum("b,bij->ij", W, omega_e),
        "omega_s_sum": W @ omega_s,
        "frac_sum": W @ frac,
        "diag_e": W2 @ s2_e,
        "diag_s": xp.sum(W2 * s2_s),
        "diag_f": xp.sum(W2 * f2),
        "sigma_e2_sum": W @ sigma_e2,
        "sigma_a2_sum": xp.sum(W * sigma_a2),
        "variance_sum": xp.sum(W * variance),
        # per-vector diagnostics
        "variance": variance,
        "sigma_a2": sigma_a2,
        "selection_ess": 1.0 / s2_s,
        "min_event_ess": xp.min(1.0 / s2_e, axis=1),
        "log_likelihood": log_like,
        "supported": supported,
    }


def bootstrap_log_likelihoods(xp, lw_e, lu, counts_e, counts_s, pe_counts):
    """``log L`` under resampled PE samples/injections.

    ``counts_e`` ``[R, N, n_max]`` and ``counts_s`` ``[R, M]`` are resampling
    multiplicities (zero for padding); ``log ell_i = log(sum_j c_ij w_ij /
    n_i)`` and ``log A = log sum_m c_m u_m``. Returns ``[R, B]`` with the
    shape-likelihood ``-inf`` convention.
    """
    n_events = lw_e.shape[1]
    peak_e = xp.max(lw_e, axis=2)
    peak_e = xp.where(xp.isfinite(peak_e), peak_e, 0.0)  # [B, N]
    scaled_e = xp.exp(lw_e - peak_e[..., None])  # [B, N, n]
    sums_e = xp.einsum("rij,bij->rbi", counts_e, scaled_e)  # [R, B, N]
    with np.errstate(divide="ignore"):
        log_ell = xp.log(sums_e) + peak_e[None] - xp.log(pe_counts)[None, None, :]
    peak_s = xp.max(lu, axis=1)
    peak_s = xp.where(xp.isfinite(peak_s), peak_s, 0.0)  # [B]
    scaled_s = xp.exp(lu - peak_s[:, None])  # [B, M]
    sums_s = counts_s @ scaled_s.T  # [R, B]
    with np.errstate(divide="ignore"):
        log_a = xp.log(sums_s) + peak_s[None, :]
    valid = xp.all(xp.isfinite(log_ell), axis=2) & xp.isfinite(log_a)
    safe_ell = xp.where(xp.isfinite(log_ell), log_ell, 0.0)
    safe_a = xp.where(xp.isfinite(log_a), log_a, 0.0)
    return xp.where(valid, xp.sum(safe_ell, axis=2) - n_events * safe_a, -xp.inf)


_JIT_LOCK = threading.Lock()
_JIT_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_JIT_CACHE_SIZE = 32


def _cached_jit(kind: str, population_model, names: tuple[str, ...], builder):
    """One jitted function per (kind, model object, names); LRU of 32 entries.

    The model object is kept in the cache entry so that its ``id`` cannot be
    reused while the entry exists.
    """
    key = (kind, id(population_model), names)
    with _JIT_LOCK:
        if key in _JIT_CACHE:
            _JIT_CACHE.move_to_end(key)
            return _JIT_CACHE[key][1]
        fn = builder()
        _JIT_CACHE[key] = (population_model, fn)
        while len(_JIT_CACHE) > _JIT_CACHE_SIZE:
            _JIT_CACHE.popitem(last=False)
        return fn


class CatalogWeightEvaluator:
    """Jitted per-sample importance-weight computations for one catalog/model.

    ``backend="jax"`` (default) evaluates blocks of ``batch_size``
    hyperparameter vectors with ``jax.jit``; ``backend="numpy"`` evaluates the
    same formulas eagerly with NumPy (reference implementation for tests; the
    population model is called with NumPy arrays and its output converted).
    """

    def __init__(
        self,
        catalog: PaddedCatalog,
        population_model,
        names: Sequence[str],
        *,
        batch_size: int = 16,
        backend: str = "jax",
    ):
        self.catalog = catalog
        self.population_model = population_model
        self.names = tuple(str(name) for name in names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("hyperparameter names must be unique and non-empty")
        self.ndim = len(self.names)
        self.batch_size = as_int("batch_size", batch_size, minimum=1)
        if backend not in ("jax", "numpy"):
            raise ValueError("backend must be 'jax' or 'numpy'")
        self.backend = backend
        if backend == "jax":
            jax, jnp = _require_jax()
            self._jax, self._jnp = jax, jnp
            self._data = jax.tree_util.tree_map(jnp.asarray, catalog.device_arrays())
        else:
            self._data = catalog.device_arrays()

    # -- raw log weights --------------------------------------------------

    def _numpy_log_weights(self, X):
        lw, lu, bad = [], [], []
        model = self.population_model

        def wrapped(samples, hp):
            return np.asarray(model(samples, hp), dtype=np.float64)

        for x in X:
            a, b, c = _single_log_weights(np, wrapped, self.names, x, self._data)
            lw.append(a)
            lu.append(b)
            bad.append(bool(c))
        return np.stack(lw), np.stack(lu), np.asarray(bad)

    def _jitted(self, kind):
        jax, jnp = self._jax, self._jnp
        model, names = self.population_model, self.names

        def single(x, data):
            return _single_log_weights(jnp, model, names, x, data)

        if kind == "moments":

            def build():
                def fn(X, W, data):
                    lw_e, lu, invalid = jax.vmap(single, in_axes=(0, None))(X, data)
                    out = weight_moments(jnp, lw_e, lu, data, W)
                    out["invalid"] = invalid
                    return out

                return jax.jit(fn)

        elif kind == "log_weights":

            def build():
                def fn(X, data):
                    return jax.vmap(single, in_axes=(0, None))(X, data)

                return jax.jit(fn)

        else:  # pragma: no cover - internal
            raise ValueError(kind)
        return _cached_jit(kind, model, names, build)

    def _raise_invalid(self, X, invalid):
        invalid = np.asarray(invalid, dtype=bool)
        if invalid.any():
            rows = np.flatnonzero(invalid)
            raise PopulationDensityError(
                "population density returned NaN/+inf importance weights for "
                + _describe(self.names, X, rows)
            )

    # -- public -------------------------------------------------------------

    def moments(self, X, W) -> dict[str, np.ndarray]:
        """Sum :func:`weight_moments` over ``X`` ``[K, ndim]`` with weights ``W`` ``[K]``.

        Accumulated sums are returned as NumPy arrays; per-vector diagnostics
        are concatenated (length ``K``). A vector with positive weight but
        zero likelihood (an event or the selection without support) raises
        ``ValueError``: it cannot belong to the posterior.
        """
        X = _check_block(X, self.ndim, self.names)
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (X.shape[0],) or not np.all(np.isfinite(W)) or np.any(W < 0):
            raise ValueError("W must be finite, non-negative, one entry per row")
        sums: dict[str, np.ndarray] = {}
        per: dict[str, list] = {}
        per_keys = ("variance", "sigma_a2", "selection_ess", "min_event_ess", "log_likelihood", "supported")
        if self.backend == "numpy":
            blocks = [(X, W)]
        else:
            padded, n_blocks = _pad_rows(X, self.batch_size)
            Wp = np.zeros(padded.shape[0])
            Wp[: X.shape[0]] = W
            blocks = [
                (
                    padded[b * self.batch_size : (b + 1) * self.batch_size],
                    Wp[b * self.batch_size : (b + 1) * self.batch_size],
                )
                for b in range(n_blocks)
            ]
        for Xb, Wb in blocks:
            if self.backend == "numpy":
                lw_e, lu, invalid = self._numpy_log_weights(Xb)
                self._raise_invalid(Xb, invalid)
                out = weight_moments(np, lw_e, lu, self._data, Wb)
            else:
                fn = self._jitted("moments")
                out = self._jax.device_get(
                    fn(self._jnp.asarray(Xb), self._jnp.asarray(Wb), self._data)
                )
                self._raise_invalid(Xb, out.pop("invalid"))
            for key, value in out.items():
                value = np.asarray(value)
                if key in per_keys:
                    per.setdefault(key, []).append(value)
                else:
                    sums[key] = value.astype(np.float64) + sums.get(key, 0.0)
        result = dict(sums)
        for key, values in per.items():
            result[key] = np.concatenate(values)[: X.shape[0]]
        unsupported = (W > 0) & ~result["supported"].astype(bool)
        if unsupported.any():
            rows = np.flatnonzero(unsupported)
            raise ValueError(
                "posterior points with positive weight have zero likelihood (an event or the "
                f"selection without population support): {_describe(self.names, X, rows)}"
            )
        return result

    def bootstrap(self, X, counts_e, counts_s, *, replicate_chunk: int = 16) -> np.ndarray:
        """``log L`` ``[R, K]`` at ``X`` for each resampling replicate.

        The per-sample log weights of every block of ``X`` are evaluated once
        and kept on the device while the replicates are applied in chunks of
        ``replicate_chunk`` (the multiplicities stay on the device as int32).
        """
        X = _check_block(X, self.ndim, self.names)
        counts_e = np.asarray(counts_e)
        counts_s = np.asarray(counts_s)
        cat = self.catalog
        if counts_e.ndim != 3 or counts_e.shape[1:] != (cat.n_events, cat.n_max):
            raise ValueError(f"counts_e must have shape [R, {cat.n_events}, {cat.n_max}]")
        if counts_s.ndim != 2 or counts_s.shape != (counts_e.shape[0], cat.m_pad):
            raise ValueError(f"counts_s must have shape [R, {cat.m_pad}]")
        for name, counts in (("counts_e", counts_e), ("counts_s", counts_s)):
            if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.round(counts)):
                raise ValueError(f"{name} must be non-negative integer multiplicities")
        replicate_chunk = as_int("replicate_chunk", replicate_chunk, minimum=1)
        n_rep = counts_e.shape[0]
        if self.backend == "numpy":
            lw_e, lu, invalid = self._numpy_log_weights(X)
            self._raise_invalid(X, invalid)
            return bootstrap_log_likelihoods(
                np, lw_e, lu, counts_e.astype(np.float64), counts_s.astype(np.float64), self._data["pe_counts"]
            )
        jax, jnp = self._jax, self._jnp
        logw = self._jitted("log_weights")
        ce = jnp.asarray(counts_e.astype(np.int32))
        cs = jnp.asarray(counts_s.astype(np.int32))
        padded, n_blocks = _pad_rows(X, self.batch_size)
        out = np.empty((n_rep, padded.shape[0]), dtype=np.float64)
        for b in range(n_blocks):
            sl = slice(b * self.batch_size, (b + 1) * self.batch_size)
            Xb = padded[sl]
            lw_e, lu, invalid = logw(jnp.asarray(Xb), self._data)
            self._raise_invalid(Xb, jax.device_get(invalid))
            for r0 in range(0, n_rep, replicate_chunk):
                r1 = min(n_rep, r0 + replicate_chunk)
                values = _BOOTSTRAP_FROM_LOG_WEIGHTS(lw_e, lu, ce[r0:r1], cs[r0:r1], self._data["pe_counts"])
                out[r0:r1, sl] = np.asarray(jax.device_get(values), dtype=np.float64)
        return out[:, : X.shape[0]]


def _bootstrap_from_log_weights(lw_e, lu, counts_e, counts_s, pe_counts):
    import jax.numpy as jnp

    return bootstrap_log_likelihoods(
        jnp, lw_e, lu, counts_e.astype(jnp.float64), counts_s.astype(jnp.float64), pe_counts
    )


class _LazyJit:
    """``jax.jit`` of a model-independent function, built on first use."""

    def __init__(self, fn):
        self._fn = fn
        self._jitted = None
        self._lock = threading.Lock()

    def __call__(self, *args):
        if self._jitted is None:
            with self._lock:
                if self._jitted is None:
                    import jax

                    self._jitted = jax.jit(self._fn)
        return self._jitted(*args)


_BOOTSTRAP_FROM_LOG_WEIGHTS = _LazyJit(_bootstrap_from_log_weights)
