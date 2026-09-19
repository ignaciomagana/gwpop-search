"""PSIS leave-one-out influence on the exact factorization of the shape likelihood.

Identities (DYNESTY_INTEGRATION_PLAN.md Sec. 8.1). The rate-marginalized shape
likelihood factorizes exactly, because ``N log A = sum_i log A``::

    L(Lambda) = prod_i lambda_i(Lambda),       lambda_i = ell_i / A
    p(Lambda | D_-i)  ∝ p(Lambda | D) / lambda_i           (same 1/R marginalization, N-1 events)
    p(d_i | D_-i)     = 1 / E_{p(Lambda|D)}[1 / lambda_i] = exp(elpd_i)
    ln Z_-i - ln Z    = ln E_{p(Lambda|D)}[1 / lambda_i] = -elpd_i
    Delta_i ln BF_c/p = ln BF_c/p(D_-i) - ln BF_c/p(D) = -(elpd_i^c - elpd_i^p)

The dropped constants (``Gamma(N)``, PE evidences) are model-independent, so
``Delta_i ln BF`` is exact.

Estimator. Every repeat's dead and final live points are used with their
nested-sampling weights (runs pooled as the equal mixture of separately
normalized runs; ``jitter_run``/``resample_run``/``merge_runs`` are never
used). A nested-sampling run is an importance sample from the density ``g``
of its dead points with weights ``W_k ∝ p(Lambda_k | D) / g``, so the
leave-one-out target has importance weights ``W_k / lambda_i(Lambda_k)``
relative to the same ``g``. Pareto-smoothed importance sampling (Vehtari et
al. 2024, JMLR 25(72); generalized-Pareto fit of Zhang & Stephens 2009 with
the weakly-informative prior on ``k`` used by ``loo``/``arviz``) is applied to
these combined weights: the largest ``M = min(S/5, 3 sqrt(S / r_eff))``
weights (``S`` = number of points with positive weight) are replaced by the
expected order statistics of the fitted generalized Pareto distribution and
``k-hat`` diagnoses the tail. For equal base weights this reduces to
standard PSIS (tested against ``arviz``). Then::

    elpd_i = log sum_k w~_k lambda_i(Lambda_k) - log sum_k w~_k
    ESS_i  = (sum_k w~_k)^2 / sum_k w~_k^2
    se_i   = sqrt(sum_k w~_k^2 (lambda_k - mu)^2) / (mu sum_k w~_k),  mu = exp(elpd_i)

and ``elpd_i`` is also computed from each repeat separately (repeat scatter).

Support complement. The identities need ``supp p(Lambda | D_-i) ⊆ supp p(Lambda | D)``.
This fails for events that define a hard edge (e.g. the event that pins
``mmin``): without it, the leave-one-out posterior extends where
``lambda_i = 0`` under the full data, which full-data posterior points never
visit and ``k-hat`` need not flag. Using every dead point (including the
zero-likelihood plateau points drawn from the prior) and its prior-volume
element ``Delta X_k``::

    c_i = sum_{k: lambda_i = 0} Delta X_k L_-i(Lambda_k) / sum_k Delta X_k L_-i(Lambda_k),
    L_-i = prod_{j != i} lambda_j

``c_i`` is the fraction of the leave-one-out evidence integrand that the PSIS
estimate cannot see (``Z_-i`` is underestimated by the factor ``1 - c_i``, so
``elpd_i`` is overestimated by ``-ln(1 - c_i)``; the advisory
``elpd_complement_corrected = elpd_i + ln(1 - c_i)`` is reported, but flagged
events still need exact refits because ``c_i`` rests on the prior draws of
the plateau phase). Volumes follow dynesty 3.1 bookkeeping
(``X_0 = X_init nlive/(nlive+1)``).

Flags (exact refit required): ``k-hat > k_threshold`` (or undefined), LOO
ESS below ``min_loo_ess``, ``c_i > max_complement_mass``, repeat scatter
above ``max(min_scatter_flag, scatter_se_factor * se)``, or an event listed
as forced (e.g. an F0 zero-support event).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from ._common import (
    AnalysisInputError,
    as_float,
    common_likelihood_identity,
    hbi_config_from_identity,
    json_ready,
    require_identity_matches,
    require_same_data,
)
from .terms import BatchedCatalogTerms, shape_log_likelihood_from_terms

PSIS_LOO_FORMAT = "gwpop-search-psis-loo-influence-1.0"


# ---------------------------------------------------------------------------
# Pareto smoothing
# ---------------------------------------------------------------------------


def gpd_fit(exceedances) -> tuple[float, float]:
    """Generalized-Pareto ``(k, sigma)`` for sorted positive exceedances.

    Empirical-Bayes estimate of Zhang & Stephens (2009) with the weakly
    informative prior on ``k`` (Gaussian centred at 0.5, weight 10 samples)
    used by Vehtari et al. (2024), ``loo`` and ``arviz``. Returns
    ``(nan, nan)`` when the first quartile does not exceed the minimum (the
    sample is far from a generalized Pareto tail), as the references do.
    """
    x = np.asarray(exceedances, dtype=np.float64)
    n = x.size
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(n**0.5)
    xstar = x[int(n / 4 + 0.5) - 1]
    if xstar <= x[0]:
        return float("nan"), float("nan")
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=np.float64) - 0.5))
    b /= prior_bs * xstar
    b += 1.0 / x[-1]
    k_b = np.mean(np.log1p(-b[:, None] * x), axis=1)
    log_like = n * (np.log(-b / k_b) - k_b - 1.0)
    w = np.exp(log_like - logsumexp(log_like))
    keep = w >= 10.0 * np.finfo(float).eps
    w, b = w[keep], b[keep]
    w /= w.sum()
    b_post = float(np.sum(b * w))
    k = float(np.mean(np.log1p(-b_post * x)))
    sigma = -k / b_post
    k = (n * k + prior_k * 0.5) / (n + prior_k)
    if math.isnan(k):
        return float("inf"), float("nan")
    return float(k), float(sigma)


def gpd_quantile(p, k: float, sigma: float, mu: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if not sigma > 0:
        return np.full_like(p, np.nan)
    if k == 0.0:
        return mu - sigma * np.log1p(-p)
    return mu + sigma * np.expm1(-k * np.log1p(-p)) / k


def psis_tail_length(n_points: int, r_eff: float = 1.0) -> int:
    """``3 sqrt(S / r_eff)`` when ``S r_eff > 225``, else ``S / 5`` (Vehtari et al. 2024)."""
    if n_points * r_eff > 225:
        return int(np.floor(3.0 * math.sqrt(n_points / r_eff)))
    return int(np.floor(n_points / 5.0))


@dataclass(frozen=True)
class PSISSmoothing:
    """Pareto-smoothed log weights on the scale of the input (not normalized)."""

    log_weights: np.ndarray
    khat: float
    tail_length: int
    smoothed: bool


def psis_smooth(log_weights, *, r_eff: float = 1.0) -> PSISSmoothing:
    """Pareto-smooth the largest importance weights (right tail).

    ``log_weights`` are log importance weights of ``S`` draws (``-inf``
    entries are allowed and stay ``-inf``; they are excluded from ``S``). The
    output keeps the input scale, so ``logsumexp`` of the output estimates the
    same normalizing sum as ``logsumexp`` of the input. With fewer than 5 tail
    draws or an undefined ``k-hat`` no smoothing is applied and ``khat`` is
    NaN (callers must treat that as unreliable).
    """
    lw = np.asarray(log_weights, dtype=np.float64)
    if lw.ndim != 1:
        raise ValueError("log_weights must be 1-D")
    if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
        raise ValueError("log_weights must not contain NaN or +inf")
    finite = np.flatnonzero(np.isfinite(lw))
    out = lw.copy()
    S = finite.size
    tail = psis_tail_length(S, as_float("r_eff", r_eff, positive=True))
    if S == 0:
        raise ValueError("no finite log weights")
    if tail < 5 or tail >= S:
        return PSISSmoothing(out, float("nan"), int(tail), False)
    values = lw[finite]
    shift = float(np.max(values))
    values = values - shift
    order = np.argsort(values, kind="stable")
    tail_idx = order[S - tail :]
    log_tail = values[tail_idx]
    log_cutoff = values[order[S - tail - 1]]
    if abs(float(log_tail[-1] - log_tail[0])) < np.finfo(float).tiny:
        return PSISSmoothing(out, float("nan"), int(tail), False)
    cutoff = math.exp(log_cutoff)
    k, sigma = gpd_fit(np.exp(log_tail) - cutoff)
    if not math.isfinite(k):
        return PSISSmoothing(out, float(k), int(tail), False)
    p = (np.arange(tail, dtype=np.float64) + 0.5) / tail
    smoothed = np.log(gpd_quantile(p, k, sigma, cutoff))
    smoothed = np.minimum(smoothed, log_tail[-1])
    values[tail_idx] = smoothed
    out[finite] = values + shift
    return PSISSmoothing(out, float(k), int(tail), True)


# ---------------------------------------------------------------------------
# Terms at every nested-sampling point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RunLOOTerms:
    """One repeat: every dead/live point with its weight, volume and terms."""

    seed: int
    nlive: int
    log_weights: np.ndarray  # [K] normalized within the run (-inf = zero weight)
    log_volumes: np.ndarray  # [K]
    event_terms: np.ndarray  # [K, N] log ell_i
    log_exposure: np.ndarray  # [K]

    @property
    def n_points(self) -> int:
        return int(self.log_weights.size)

    def log_volume_elements(self) -> np.ndarray:
        """``log(X_{k-1} - X_k)`` with ``X_init = X_0 (nlive + 1) / nlive``."""
        lv = np.asarray(self.log_volumes, dtype=np.float64)
        log_init = lv[0] + math.log((self.nlive + 1) / self.nlive)
        prev = np.concatenate(([log_init], lv[:-1]))
        with np.errstate(divide="ignore"):
            return prev + np.log1p(-np.exp(lv - prev))


@dataclass(frozen=True, eq=False)
class ModelLOOTerms:
    label: str
    event_names: tuple[str, ...]
    runs: tuple[RunLOOTerms, ...]
    log_evidences: tuple[float, ...]
    likelihood_identity: Mapping[str, object] | None

    @property
    def n_events(self) -> int:
        return len(self.event_names)

    @property
    def log_evidence(self) -> float:
        return float(np.mean(self.log_evidences))


def evaluate_model_loo_terms(
    results: Sequence,
    posterior,
    selection,
    population_model,
    *,
    label: str | None = None,
    hbi_config=None,
    batch_size: int = 64,
    verify_identity: bool = True,
    parity_rtol: float = 1e-8,
    terms_fn: BatchedCatalogTerms | None = None,
) -> ModelLOOTerms:
    """Evaluate ``log ell_i`` and ``log A`` at every point of every repeat.

    The evaluated ``sum_i log ell_i - N log A`` must reproduce the stored
    ``log L`` of every point (finite values to ``parity_rtol``, ``-inf``
    exactly): a mismatch means the data or model differ from the sampled
    ones and raises :class:`AnalysisInputError`.
    """
    results = tuple(results)
    if not results:
        raise ValueError("at least one dynesty result is required")
    names = tuple(results[0].names)
    if any(tuple(item.names) != names for item in results):
        raise AnalysisInputError("repeats have different parameter names")
    identity = common_likelihood_identity(results)
    if hbi_config is None:
        hbi_config = hbi_config_from_identity(identity)
    if verify_identity:
        require_identity_matches(
            identity, posterior, selection, population_model, names, hbi_config,
            what=f"PSIS-LOO terms of {label or 'model'}",
        )
    fn = terms_fn or BatchedCatalogTerms(
        posterior, selection, population_model, names, hbi_config=hbi_config, batch_size=batch_size
    )
    runs = []
    for item in results:
        events, exposure = fn(item.samples)
        log_like = shape_log_likelihood_from_terms(events, exposure)
        stored = np.asarray(item.log_likelihoods, dtype=np.float64)
        both = np.isfinite(log_like) & np.isfinite(stored)
        pattern = np.isfinite(log_like) != np.isfinite(stored)
        tol = parity_rtol * np.maximum(1.0, np.abs(np.where(both, stored, 0.0)))
        diff = np.abs(np.where(both, log_like, 0.0) - np.where(both, stored, 0.0))
        bad = pattern | (both & (diff > tol))
        if bad.any():
            rows = np.flatnonzero(bad)[:5]
            raise AnalysisInputError(
                f"{label or 'model'} (seed {item.seed}): evaluated log L differs from the stored "
                f"log L at {int(bad.sum())} point(s), e.g. rows {rows.tolist()}: "
                f"{log_like[rows].tolist()} vs {stored[rows].tolist()}"
            )
        log_w = np.asarray(item.log_weights, dtype=np.float64)
        log_w = log_w - logsumexp(log_w)
        runs.append(
            RunLOOTerms(
                seed=int(item.seed),
                nlive=int(item.config.nlive),
                log_weights=log_w,
                log_volumes=np.asarray(item.log_volumes, dtype=np.float64),
                event_terms=events,
                log_exposure=exposure,
            )
        )
    return ModelLOOTerms(
        label=str(label) if label is not None else "model",
        event_names=tuple(posterior.event_names),
        runs=tuple(runs),
        log_evidences=tuple(float(item.log_evidence) for item in results),
        likelihood_identity=identity,
    )


# ---------------------------------------------------------------------------
# Per-model LOO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PSISLOOConfig:
    k_threshold: float = 0.7
    min_loo_ess: float = 100.0
    max_complement_mass: float = 0.01
    min_scatter_flag: float = 0.5
    scatter_se_factor: float = 3.0
    r_eff: float = 1.0

    def __post_init__(self) -> None:
        as_float("k_threshold", self.k_threshold)
        as_float("min_loo_ess", self.min_loo_ess, positive=True)
        as_float("max_complement_mass", self.max_complement_mass, nonnegative=True)
        as_float("min_scatter_flag", self.min_scatter_flag, nonnegative=True)
        as_float("scatter_se_factor", self.scatter_se_factor, positive=True)
        as_float("r_eff", self.r_eff, positive=True)

    def to_dict(self) -> dict[str, float]:
        return {
            "k_threshold": self.k_threshold,
            "min_loo_ess": self.min_loo_ess,
            "max_complement_mass": self.max_complement_mass,
            "min_scatter_flag": self.min_scatter_flag,
            "scatter_se_factor": self.scatter_se_factor,
            "r_eff": self.r_eff,
        }


@dataclass(frozen=True)
class LOOEstimate:
    """PSIS estimate for one event (see the module docstring)."""

    elpd: float
    se: float
    khat: float
    ess: float
    raw_log_mean_inverse: float


def psis_loo_estimate(log_base_weights, log_lambda, *, r_eff: float = 1.0) -> LOOEstimate:
    """PSIS ``elpd_i`` from posterior log weights and ``log lambda_i`` at the same points.

    ``log_base_weights`` are the (possibly unnormalized, ``-inf`` for zero
    weight) posterior log weights of the points; ``log_lambda`` is
    ``log ell_i - log A`` there. ``raw_log_mean_inverse`` is the unsmoothed
    ``log E_post[1/lambda_i]`` (``= ln Z_-i - ln Z``).
    """
    log_base = np.asarray(log_base_weights, dtype=np.float64)
    log_base = log_base - logsumexp(log_base)
    return _loo_event(log_base, np.asarray(log_lambda, dtype=np.float64), r_eff)


def _loo_event(log_base, log_lambda, r_eff) -> LOOEstimate:
    """PSIS estimate for one event from normalized base log weights and log lambda_i."""
    keep = np.isfinite(log_base)
    lb, ll = log_base[keep], log_lambda[keep]
    if not np.all(np.isfinite(ll)):
        raise AnalysisInputError(
            "a point with positive posterior weight has lambda_i = 0 (zero likelihood)"
        )
    combined = lb - ll
    raw = float(logsumexp(combined))
    sm = psis_smooth(combined, r_eff=r_eff)
    lw = sm.log_weights
    log_norm = logsumexp(lw)
    elpd = float(logsumexp(lw + ll) - log_norm)
    w = np.exp(lw - log_norm)
    ess = float(1.0 / np.sum(w * w))
    with np.errstate(over="ignore"):
        lam = np.exp(ll - elpd)  # lambda_k / mu with mu = exp(elpd)
    se = float(math.sqrt(float(np.sum(w * w * (lam - 1.0) ** 2))))
    return LOOEstimate(elpd=elpd, se=se, khat=sm.khat, ess=ess, raw_log_mean_inverse=raw)


def _complement_mass(run: RunLOOTerms, i: int, n_events: int) -> float:
    """``c_i`` of one run (see the module docstring)."""
    events = run.event_terms
    log_a = run.log_exposure
    others_finite = np.all(np.isfinite(np.delete(events, i, axis=1)), axis=1) & np.isfinite(log_a)
    if not others_finite.any():
        return float("nan")
    others = np.where(
        others_finite,
        np.sum(np.where(np.isfinite(events), events, 0.0), axis=1)
        - np.where(np.isfinite(events[:, i]), events[:, i], 0.0)
        - (n_events - 1) * np.where(np.isfinite(log_a), log_a, 0.0),
        -np.inf,
    )
    log_dx = run.log_volume_elements()
    terms = log_dx + others
    total = logsumexp(terms[others_finite])
    missing = others_finite & ~np.isfinite(events[:, i])
    if not missing.any():
        return 0.0
    return float(np.exp(logsumexp(terms[missing]) - total))


@dataclass(frozen=True, eq=False)
class ModelLOOResult:
    label: str
    event_names: tuple[str, ...]
    elpd: np.ndarray
    se: np.ndarray
    khat: np.ndarray
    loo_ess: np.ndarray
    raw_log_mean_inverse: np.ndarray
    per_run_elpd: np.ndarray  # [R, N]
    repeat_scatter: np.ndarray
    complement_mass: np.ndarray  # mean over runs
    complement_mass_max: np.ndarray
    flags: tuple[tuple[str, ...], ...]
    log_evidence: float
    log_evidences: tuple[float, ...]
    config: PSISLOOConfig

    @property
    def n_runs(self) -> int:
        return int(self.per_run_elpd.shape[0])

    @property
    def flagged(self) -> np.ndarray:
        return np.asarray([bool(f) for f in self.flags])

    @property
    def total_uncertainty(self) -> np.ndarray:
        """``sqrt(se^2 + scatter^2 / R)`` per event."""
        scatter = np.nan_to_num(self.repeat_scatter, nan=0.0)
        return np.sqrt(self.se**2 + scatter**2 / self.n_runs)

    def elpd_loo_total(self) -> dict[str, float]:
        n = self.elpd.size
        return {
            "elpd_loo": float(np.sum(self.elpd)),
            "se": float(math.sqrt(n * np.var(self.elpd, ddof=1))) if n > 1 else float("nan"),
        }

    def to_rows(self) -> list[dict[str, object]]:
        rows = []
        for i, name in enumerate(self.event_names):
            rows.append(
                {
                    "event": name,
                    "elpd": float(self.elpd[i]),
                    "delta_log_evidence": float(-self.elpd[i]),
                    "se": float(self.se[i]),
                    "khat": float(self.khat[i]),
                    "loo_ess": float(self.loo_ess[i]),
                    "raw_log_mean_inverse_lambda": float(self.raw_log_mean_inverse[i]),
                    "per_run_elpd": self.per_run_elpd[:, i].tolist(),
                    "repeat_scatter": float(self.repeat_scatter[i]),
                    "complement_mass": float(self.complement_mass[i]),
                    "complement_mass_max": float(self.complement_mass_max[i]),
                    "elpd_complement_corrected": _complement_corrected(
                        float(self.elpd[i]), float(self.complement_mass[i])
                    ),
                    "flags": list(self.flags[i]),
                    "flagged": bool(self.flags[i]),
                }
            )
        return rows


def _complement_corrected(elpd: float, complement: float) -> float | None:
    """``elpd + ln(1 - c)`` (advisory); ``None`` when ``c`` is undefined or 1."""
    if not math.isfinite(complement) or complement >= 1.0:
        return None
    return float(elpd + math.log1p(-complement))


def psis_loo_model(
    terms: ModelLOOTerms,
    *,
    config: PSISLOOConfig | None = None,
    forced_events: Iterable[str] = (),
) -> ModelLOOResult:
    """PSIS-LOO per event for one model from all its repeats (see module doc)."""
    config = PSISLOOConfig() if config is None else config
    forced = set(str(name) for name in forced_events)
    unknown = forced - set(terms.event_names)
    if unknown:
        raise AnalysisInputError(f"forced events {sorted(unknown)} are not in the catalog")
    n_runs = len(terms.runs)
    n = terms.n_events
    # pooled base weights (equal mixture of runs) and log lambda at every point
    log_base = np.concatenate([run.log_weights - math.log(n_runs) for run in terms.runs])
    log_lambda = np.concatenate(
        [run.event_terms - run.log_exposure[:, None] for run in terms.runs], axis=0
    )
    elpd = np.empty(n)
    se = np.empty(n)
    khat = np.empty(n)
    ess = np.empty(n)
    raw = np.empty(n)
    per_run = np.empty((n_runs, n))
    comp = np.empty((n_runs, n))
    for i in range(n):
        est = _loo_event(log_base, log_lambda[:, i], config.r_eff)
        elpd[i], se[i], khat[i], ess[i], raw[i] = (
            est.elpd, est.se, est.khat, est.ess, est.raw_log_mean_inverse,
        )
        for r, run in enumerate(terms.runs):
            per_run[r, i] = _loo_event(
                run.log_weights, run.event_terms[:, i] - run.log_exposure, config.r_eff
            ).elpd
            comp[r, i] = _complement_mass(run, i, n)
    scatter = np.std(per_run, axis=0, ddof=1) if n_runs > 1 else np.full(n, np.nan)
    comp_mean = np.nanmean(comp, axis=0) if np.isfinite(comp).any() else np.full(n, np.nan)
    comp_max = np.nanmax(np.where(np.isfinite(comp), comp, -np.inf), axis=0)
    flags = []
    for i, name in enumerate(terms.event_names):
        f = []
        if not math.isfinite(khat[i]):
            f.append("khat_undefined")
        elif khat[i] > config.k_threshold:
            f.append("khat")
        if ess[i] < config.min_loo_ess:
            f.append("loo_ess")
        if not math.isfinite(comp_max[i]) or comp_max[i] > config.max_complement_mass:
            f.append("support_complement")
        if n_runs > 1 and scatter[i] > max(config.min_scatter_flag, config.scatter_se_factor * se[i]):
            f.append("repeat_scatter")
        if name in forced:
            f.append("forced")
        flags.append(tuple(f))
    return ModelLOOResult(
        label=terms.label,
        event_names=terms.event_names,
        elpd=elpd,
        se=se,
        khat=khat,
        loo_ess=ess,
        raw_log_mean_inverse=raw,
        per_run_elpd=per_run,
        repeat_scatter=scatter,
        complement_mass=comp_mean,
        complement_mass_max=comp_max,
        flags=tuple(flags),
        log_evidence=terms.log_evidence,
        log_evidences=terms.log_evidences,
        config=config,
    )


# ---------------------------------------------------------------------------
# Edges, model probabilities and the report
# ---------------------------------------------------------------------------


def loo_edge_influence(
    parent: ModelLOOResult,
    child: ModelLOOResult,
    *,
    log_bayes_factor: float | None = None,
) -> list[dict[str, object]]:
    """Per-event ``Delta_i ln BF_{child/parent}`` with uncertainty and flags."""
    if parent.event_names != child.event_names:
        raise AnalysisInputError("parent and child LOO results cover different events")
    rows = []
    err_p, err_c = parent.total_uncertainty, child.total_uncertainty
    for i, name in enumerate(parent.event_names):
        delta = float(-(child.elpd[i] - parent.elpd[i]))
        row = {
            "event": name,
            "delta_log_bayes_factor": delta,
            "error": float(math.hypot(err_p[i], err_c[i])),
            "flagged": bool(parent.flags[i] or child.flags[i]),
            "parent_flags": list(parent.flags[i]),
            "child_flags": list(child.flags[i]),
        }
        if log_bayes_factor is not None:
            dropped = log_bayes_factor + delta
            row["log_bayes_factor_without_event"] = float(dropped)
            row["sign_reversal"] = bool(np.sign(dropped) != np.sign(log_bayes_factor))
        rows.append(row)
    return rows


def loo_model_probabilities(
    results: Mapping[str, ModelLOOResult],
    log_model_priors: Mapping[str, float],
) -> dict[str, object]:
    """``ln P(M | D_-i) = ln pi(M) + ln Z_M - elpd_i^M - logsumexp(...)`` per event.

    ``log_model_priors`` must cover exactly the keys of ``results`` (the
    deduplicated model set).
    """
    keys = sorted(results)
    if set(log_model_priors) != set(keys):
        raise AnalysisInputError("model priors must cover exactly the LOO model set")
    event_names = results[keys[0]].event_names
    lnz = np.asarray([results[k].log_evidence for k in keys])
    lp = np.asarray([float(log_model_priors[k]) for k in keys])
    full = lnz + lp
    full = np.exp(full - logsumexp(full))
    elpd = np.stack([results[k].elpd for k in keys])  # [K, N]
    loo = lnz[:, None] + lp[:, None] - elpd
    loo = np.exp(loo - logsumexp(loo, axis=0, keepdims=True))
    return {
        "models": keys,
        "full_data": dict(zip(keys, full.tolist())),
        "per_event": {
            name: dict(zip(keys, loo[:, i].tolist())) for i, name in enumerate(event_names)
        },
    }


def psis_loo_report(
    model_results: Mapping[str, ModelLOOResult],
    *,
    edges: Sequence[Mapping[str, str]] = (),
    log_model_priors: Mapping[str, float] | None = None,
    structure_membership: Mapping[str, Sequence[str]] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """JSON report (``gwpop-search-psis-loo-influence-1.0``).

    ``edges`` are mappings with ``parent_hash``, ``child_hash`` and
    ``mutation_id``; ``structure_membership`` maps a structural label to the
    model keys containing it (for the per-event change of structural mass).
    """
    if not model_results:
        raise ValueError("at least one model LOO result is required")
    configs = [r.config.to_dict() for r in model_results.values()]
    if any(config != configs[0] for config in configs):
        raise AnalysisInputError("model LOO results were computed with different configurations")
    models = {}
    for key, result in sorted(model_results.items()):
        models[key] = {
            "label": result.label,
            "log_evidence": result.log_evidence,
            "log_evidences": list(result.log_evidences),
            "n_runs": result.n_runs,
            "elpd_loo": result.elpd_loo_total(),
            "n_flagged_events": int(np.sum(result.flagged)),
            "flagged_events": [
                name for name, flags in zip(result.event_names, result.flags) if flags
            ],
            "events": result.to_rows(),
            "interpretation": "elpd_loo is a predictive score, not a Bayes factor",
        }
    edge_rows = []
    for edge in edges:
        p, c = str(edge["parent_hash"]), str(edge["child_hash"])
        if p not in model_results or c not in model_results:
            continue
        lnbf = model_results[c].log_evidence - model_results[p].log_evidence
        rows = loo_edge_influence(model_results[p], model_results[c], log_bayes_factor=lnbf)
        worst = max(rows, key=lambda row: abs(row["delta_log_bayes_factor"]))
        edge_rows.append(
            {
                "parent_hash": p,
                "child_hash": c,
                "mutation_id": str(edge.get("mutation_id", "")),
                "log_bayes_factor": lnbf,
                "max_abs_delta_log_bayes_factor": abs(worst["delta_log_bayes_factor"]),
                "max_abs_delta_event": worst["event"],
                "n_flagged_events": int(sum(row["flagged"] for row in rows)),
                "any_sign_reversal": bool(any(row.get("sign_reversal") for row in rows)),
                "events": rows,
            }
        )
    payload: dict[str, object] = {
        "format_version": PSIS_LOO_FORMAT,
        "config": next(iter(model_results.values())).config.to_dict(),
        "models": models,
        "edges": edge_rows,
        "exact_refit_required": sorted(
            {
                (key, name)
                for key, result in model_results.items()
                for name, flags in zip(result.event_names, result.flags)
                if flags
            }
        ),
    }
    if log_model_priors is not None:
        probs = loo_model_probabilities(model_results, log_model_priors)
        payload["model_probabilities"] = probs
        if structure_membership:
            structural = {}
            for label, members in structure_membership.items():
                members = [m for m in members if m in model_results]
                full = sum(probs["full_data"][m] for m in members)
                per_event = {
                    name: sum(values[m] for m in members) - full
                    for name, values in probs["per_event"].items()
                }
                worst = max(per_event.items(), key=lambda item: abs(item[1]))
                structural[label] = {
                    "posterior_mass": full,
                    "delta_mass_per_event": per_event,
                    "max_abs_delta_mass": abs(worst[1]),
                    "max_abs_delta_event": worst[0],
                }
            payload["structural_mass"] = structural
    if extra:
        payload.update(dict(extra))
    payload["exact_refit_required"] = [list(item) for item in payload["exact_refit_required"]]
    return json_ready(payload)


def flagged_events(report: Mapping[str, object]) -> list[str]:
    """Events that need an exact drop-one refit for at least one model."""
    require = report.get("exact_refit_required") or []
    return sorted({str(item[1]) for item in require})


def check_same_catalog(results_by_model: Mapping[str, Sequence]) -> None:
    """All models of one LOO analysis must share the data (condition C1)."""
    items = list(results_by_model.items())
    first_key, first = items[0]
    ref = common_likelihood_identity(first)
    for key, results in items[1:]:
        require_same_data(ref, common_likelihood_identity(results), what=f"{first_key[:12]} and {key[:12]}")


__all__ = [
    "LOOEstimate",
    "ModelLOOResult",
    "ModelLOOTerms",
    "PSISLOOConfig",
    "PSISSmoothing",
    "PSIS_LOO_FORMAT",
    "RunLOOTerms",
    "check_same_catalog",
    "evaluate_model_loo_terms",
    "flagged_events",
    "gpd_fit",
    "gpd_quantile",
    "loo_edge_influence",
    "loo_model_probabilities",
    "psis_loo_estimate",
    "psis_loo_model",
    "psis_loo_report",
    "psis_smooth",
    "psis_tail_length",
]

