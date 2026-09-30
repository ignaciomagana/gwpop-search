"""Nested-sampling diagnostics for repeated, independent dynesty runs.

Everything here is plain NumPy/SciPy on finished runs; nothing re-integrates a
run. In particular no function uses ``dynesty.utils.jitter_run``,
``resample_run``, ``merge_runs`` or ``unravel_run``: those rebuild prior
volumes without the plateau bookkeeping of the main run and are biased by
~+0.6-0.7 nat on runs whose likelihood is ``-inf`` on part of the prior
(design note ``MODEL_COMPARISON_MATH.md`` Sec. 7.3). Uncertainties come from
independent repeats and pooled posteriors are equal-weight mixtures of
separately normalized runs.

Contracts
---------
* ``pooled_weighted_samples``: the pooled posterior is the equal mixture
  ``p(theta) = (1/R) sum_r p_r(theta)``; every run's importance weights are
  normalized by its own evidence (``w_rk = exp(logwt_rk - logz_r)``) before the
  mixture weights ``1/R`` are applied.
* ``rank_normalized_split_rhat``: Vehtari, Gelman, Simpson, Carpenter &
  Buerkner (2021), Bayesian Analysis 16, 667: ``max(bulk, folded)``
  rank-normalized split R-hat of ``chains`` shaped ``[n_chains, n_draws]``.
* ``cross_run_rhat``: each run is a "chain" of ``M = min(max_draws, floor(min
  Kish ESS))`` equal-weight draws obtained by systematic resampling followed by
  a random permutation (resampled draws are ordered by likelihood, which would
  inflate split R-hat).
* ``max_pairwise_z``: ``max_{r<s} |lnZ_r - lnZ_s| / sqrt(err_r^2 + err_s^2)``.
* ``insertion_index_ranks``: port of the Fowlie, Handley & Su (2020) check in
  ``design/validation/insertion_index.py``; it needs dynesty's ``samples_it``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.special import logsumexp, ndtri

# dynesty stores zero-likelihood (plateau) points with logl = -1e300; results
# of the gwpop-search backend report them as -inf.
_LOWL = -1.0e299


def normalized_weights(log_weights) -> np.ndarray:
    """Normalized importance weights ``exp(logwt - logsumexp(logwt))``.

    ``-inf`` entries (zero-likelihood plateau points) get weight zero; NaN or
    ``+inf`` log weights raise ``ValueError``.
    """
    log_weights = np.asarray(log_weights, dtype=float)
    if log_weights.ndim != 1 or log_weights.size == 0:
        raise ValueError("log_weights must be a non-empty 1-D array")
    if np.any(np.isnan(log_weights) | np.isposinf(log_weights)):
        raise ValueError("log_weights contain NaN or +inf")
    if not np.any(np.isfinite(log_weights)):
        raise ValueError("log_weights have no finite entry")
    return np.exp(log_weights - logsumexp(log_weights))


def kish_ess(log_weights) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` of importance weights."""
    weights = normalized_weights(log_weights)
    return float(1.0 / np.sum(weights**2))


def pooled_weighted_samples(results: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Equal mixture of separately normalized runs.

    ``results`` expose ``samples [n_r, d]`` and ``log_weights [n_r]`` (a
    :class:`~gwpop_search.inference.dynesty_backend.DynestyResult`). Returns
    ``(samples [sum n_r, d], weights [sum n_r])`` with weights summing to one
    and each run carrying total mass ``1/R``.
    """
    results = tuple(results)
    if not results:
        raise ValueError("at least one run is required")
    names = getattr(results[0], "names", None)
    blocks = []
    weights = []
    for result in results:
        if getattr(result, "names", None) != names:
            raise ValueError("runs have different parameter names")
        samples = np.asarray(result.samples, dtype=float)
        w = normalized_weights(result.log_weights)
        if samples.ndim != 2 or samples.shape[0] != w.size:
            raise ValueError("samples and log_weights have inconsistent shapes")
        blocks.append(samples)
        weights.append(w / len(results))
    return np.vstack(blocks), np.concatenate(weights)


def weighted_quantiles(values, weights, quantiles: Sequence[float]) -> np.ndarray:
    """Quantiles of a weighted 1-D sample (inverted weighted CDF).

    Returns, for each ``q``, the smallest value whose cumulative weight is at
    least ``q`` (the weighted analogue of ``numpy.quantile(method=
    "inverted_cdf")``).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.shape != weights.shape or values.size == 0:
        raise ValueError("values and weights must be non-empty 1-D arrays of equal length")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    q = np.asarray(quantiles, dtype=float)
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    index = np.searchsorted(cumulative, q, side="left")
    index = np.minimum(index, values.size - 1)
    return values[order][index]


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Rank-normalized z-scores ``Phi^-1((r - 3/8) / (S + 1/4))`` over all draws."""
    flat = values.reshape(-1)
    ranks = stats.rankdata(flat, method="average")
    z = ndtri((ranks - 0.375) / (flat.size + 0.25))
    return z.reshape(values.shape)


def _split_chains(chains: np.ndarray) -> np.ndarray:
    n = chains.shape[1] // 2
    return np.concatenate([chains[:, :n], chains[:, chains.shape[1] - n :]], axis=0)


def _plain_rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    within = chains.var(axis=1, ddof=1).mean()
    between = n * chain_means.var(ddof=1)
    if within <= 0.0:
        # Every chain is constant: identical constants agree, distinct ones do not.
        return 1.0 if between <= 0.0 else math.inf
    var_plus = (n - 1) / n * within + between / n
    return float(math.sqrt(var_plus / within))


def rank_normalized_split_rhat(chains) -> float:
    """``max(bulk, folded)`` rank-normalized split R-hat (Vehtari et al. 2021).

    ``chains`` is ``[n_chains, n_draws]`` with at least 2 chains and 4 draws
    per chain; each chain is split in half, the pooled draws are rank
    normalized, and R-hat is computed for the normalized draws (bulk) and for
    the normalized absolute deviations from the pooled median (folded).
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[0] < 2 or chains.shape[1] < 4:
        raise ValueError("R-hat needs at least 2 chains with at least 4 draws each")
    if not np.all(np.isfinite(chains)):
        raise ValueError("chains contain non-finite draws")
    split = _split_chains(chains)
    bulk = _plain_rhat(_rank_normalize(split))
    folded = _plain_rhat(_rank_normalize(np.abs(split - np.median(split))))
    return float(max(bulk, folded))


def cross_run_rhat(
    results: Sequence[object],
    *,
    max_draws: int = 2000,
    seed: int,
) -> dict[str, object]:
    """Cross-run rank-normalized split R-hat, one "chain" per independent run.

    Each run contributes ``M = min(max_draws, floor(min_r Kish_r))`` draws from
    its weighted samples (systematic resampling, then a random permutation,
    with the generator ``default_rng([seed, run_index])``). Requires at least
    two runs; ``M`` is at least 4.
    """
    from .dynesty_backend import equal_weight_resample

    results = tuple(results)
    if len(results) < 2:
        raise ValueError("cross-run R-hat needs at least two independent runs")
    if int(max_draws) < 4:
        raise ValueError("max_draws must be at least 4")
    names = tuple(results[0].names)
    kish = [kish_ess(result.log_weights) for result in results]
    n_draws = int(max(4, min(int(max_draws), math.floor(min(kish)))))
    chains = []
    for index, result in enumerate(results):
        if tuple(result.names) != names:
            raise ValueError("runs have different parameter names")
        rng = np.random.default_rng([int(seed), int(index)])
        weights = normalized_weights(result.log_weights)
        chains.append(
            equal_weight_resample(np.asarray(result.samples, dtype=float), weights, n_draws, rng)
        )
    stacked = np.stack(chains)  # [n_runs, n_draws, ndim]
    per_parameter = {
        name: rank_normalized_split_rhat(stacked[:, :, k]) for k, name in enumerate(names)
    }
    worst = max(per_parameter, key=per_parameter.get)
    return {
        "method": "rank_normalized_split_rhat_max_bulk_folded",
        "n_runs": len(results),
        "draws_per_run": n_draws,
        "per_parameter": per_parameter,
        "max": float(per_parameter[worst]),
        "worst_parameter": worst,
    }


def max_pairwise_z(estimates: Sequence[float], errors: Sequence[float]) -> float | None:
    """Largest pairwise repeat inconsistency ``|dlnZ| / sqrt(err_r^2 + err_s^2)``.

    ``None`` for fewer than two runs; ``inf`` for a difference with zero
    combined error.
    """
    estimates = np.asarray(estimates, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if estimates.shape != errors.shape or estimates.ndim != 1:
        raise ValueError("estimates and errors must be 1-D arrays of equal length")
    if estimates.size < 2:
        return None
    worst = 0.0
    for r in range(estimates.size):
        for s in range(r + 1, estimates.size):
            delta = abs(float(estimates[r] - estimates[s]))
            scale = math.hypot(float(errors[r]), float(errors[s]))
            if scale > 0.0:
                z = delta / scale
            else:
                z = 0.0 if delta == 0.0 else math.inf
            worst = max(worst, z)
    return float(worst)


def run_termination(result) -> str:
    """Why a run stopped: ``"dlogz"`` (converged), ``"budget"`` or ``"unknown"``.

    Uses the backend's ``diagnostics["converged"]`` (final remaining-evidence
    estimate below ``dlogz``); ``"budget"`` means maxiter/maxcall stopped the
    run first.
    """
    diagnostics = getattr(result, "diagnostics", {}) or {}
    converged = diagnostics.get("converged")
    if converged is True:
        return "dlogz"
    if converged is False:
        return "budget"
    return "unknown"


def prior_edge_mass(
    samples: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
    priors: Mapping[str, object],
    *,
    fraction: float = 0.01,
) -> dict[str, dict[str, float]]:
    """Posterior mass within ``fraction`` of each bounded prior edge (advisory).

    Uniform priors use the linear width, log-uniform priors the logarithmic
    width; unbounded (normal) priors are skipped.
    """
    out: dict[str, dict[str, float]] = {}
    for k, name in enumerate(names):
        spec = priors[name]
        family = getattr(spec, "family", None)
        if family not in {"uniform", "log_uniform"}:
            continue
        x = np.asarray(samples[:, k], dtype=float)
        low, high = float(spec.low), float(spec.high)
        if family == "log_uniform":
            x, low, high = np.log(x), math.log(low), math.log(high)
        width = fraction * (high - low)
        out[name] = {
            "lower": float(np.sum(weights[x <= low + width])),
            "upper": float(np.sum(weights[x >= high - width])),
        }
    return out


def insertion_index_ranks(
    log_likelihoods,
    samples_it,
    *,
    niter: int,
    nlive: int,
) -> tuple[np.ndarray, int]:
    """Insertion ranks of new live points among the current live points.

    Port of ``design/validation/insertion_index.py`` (dynesty 3.1.0 static
    runs): a point born at iteration ``b >= 1`` replaced the point that died
    at row ``b - 1``; the live set just before its insertion is every point
    born at or before ``b - 1`` that died at or after row ``b``. Insertions
    made while any live point sits on the zero-likelihood plateau are excluded
    and counted. Returns ``(ranks, n_excluded)``.
    """
    logl = np.asarray(log_likelihoods, dtype=float).copy()
    logl[~np.isfinite(logl)] = -1.0e300
    born = np.asarray(samples_it, dtype=np.int64)
    if logl.shape != born.shape or logl.ndim != 1:
        raise ValueError("log_likelihoods and samples_it must be 1-D of equal length")
    death = np.arange(logl.size)
    ranks: list[int] = []
    excluded = 0
    for j in np.argsort(born, kind="stable"):
        b = int(born[j])
        if b == 0:
            continue
        t = b - 1
        if t >= int(niter):
            continue
        live = (born <= b - 1) & (death > t)
        live[j] = False
        current = logl[live]
        if current.size not in (nlive - 1, nlive):
            continue
        if np.any(current <= _LOWL):
            excluded += 1
            continue
        ranks.append(int(np.sum(current < logl[j])))
    return np.asarray(ranks, dtype=np.int64), excluded


def insertion_index_test(ranks, nlive: int) -> dict[str, float | int]:
    """KS test of ``(rank + 1/2) / nlive`` against U(0, 1) (advisory; low power)."""
    ranks = np.asarray(ranks, dtype=float)
    if ranks.size == 0:
        return {"n_insertions": 0, "ks_statistic": math.nan, "p_value": math.nan}
    result = stats.kstest((ranks + 0.5) / float(nlive), "uniform")
    return {
        "n_insertions": int(ranks.size),
        "ks_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
