"""Posterior-quality and importance gates for repeated dynesty runs.

Nested-sampling diagnostics replace the MCMC ones (orchestrator decision D3):

* cross-run rank-normalized split R-hat (Vehtari et al. 2021, Bayesian Anal.
  16, 667): every repeat is a "chain" of its first ``min(max_draws, Kish ESS)``
  equal-weight draws (the backend's draws are systematically resampled and then
  randomly permuted, so a prefix is an unordered subsample); per parameter the
  maximum of the bulk (rank-normalized) and tail (folded) split R-hat, and the
  maximum over parameters is gated;
* Kish ESS of every run, ``(sum w)^2 / sum w^2`` over dead and live points;
* termination by ``dlogz`` (not by ``maxiter``/``maxcall``);
* importance diagnostics (min event ESS, selection ESS, max event weight, max
  selection weight, ``Var(log L)``) at the posterior-median point AND at the
  median over ``n_draws`` pooled posterior draws AND at the tail (the 10%
  quantile for ESS-type metrics, the 90% quantile for weights/variance).

The pooled posterior is the equal-weight mixture of separately normalized
runs. Evidence-precision gates are not part of this module (holdout refits
use posteriors only).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from ._common import AnalysisInputError, as_float, as_int, common_likelihood_identity, pool_dynesty_results

POSTERIOR_GATES_FORMAT = "gwpop-search-posterior-gates-1.0"


def _z_scale(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average").reshape(values.shape)
    return ndtri((ranks - 3.0 / 8.0) / (values.size - 2.0 * 3.0 / 8.0 + 1.0))


def _split(chains: np.ndarray) -> np.ndarray:
    half = chains.shape[1] // 2
    return np.vstack((chains[:, :half], chains[:, -half:]))


def _classic_rhat(chains: np.ndarray) -> float:
    n = chains.shape[1]
    means = chains.mean(axis=1)
    within = chains.var(axis=1, ddof=1).mean()
    between = n * means.var(ddof=1)
    return float(math.sqrt((between / within + n - 1.0) / n))


def rank_normalized_split_rhat(chains) -> float:
    """``max(bulk, tail)`` rank-normalized split R-hat of ``chains [n_chains, n_draws]``."""
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2 or chains.shape[0] < 2 or chains.shape[1] < 4:
        raise ValueError("R-hat needs at least 2 chains of at least 4 draws")
    split = _split(chains)
    bulk = _classic_rhat(_z_scale(split))
    folded = np.abs(split - np.median(split))
    tail = _classic_rhat(_z_scale(folded))
    return float(max(bulk, tail))


def cross_run_rhat(results: Sequence, *, max_draws: int = 2000) -> dict[str, object]:
    """Per-parameter cross-run R-hat of repeated runs (see module doc)."""
    results = tuple(results)
    if len(results) < 2:
        raise ValueError("cross-run R-hat needs at least two runs")
    n = min(
        as_int("max_draws", max_draws, minimum=4),
        *(int(math.floor(r.kish_ess)) for r in results),
        *(r.posterior_samples.shape[0] for r in results),
    )
    if n < 4:
        raise ValueError("too few effective samples for R-hat")
    names = tuple(results[0].names)
    per = {}
    for k, name in enumerate(names):
        chains = np.stack([np.asarray(r.posterior_samples[:n, k]) for r in results])
        per[name] = rank_normalized_split_rhat(chains)
    worst = max(per, key=per.get)
    return {"n_draws_per_run": int(n), "per_parameter": per, "max": per[worst], "worst_parameter": worst}


@dataclass(frozen=True)
class PosteriorGateCriteria:
    """Thresholds; defaults are the F4 values of decision D3."""

    max_cross_run_rhat: float = 1.01
    min_kish_ess_per_run: float = 2000.0
    require_dlogz_termination: bool = True
    min_event_ess: float = 20.0
    min_selection_ess: float = 200.0
    max_event_weight_fraction: float = 0.25
    max_selection_weight_fraction: float = 0.10
    max_shape_log_likelihood_variance: float = 1.0
    n_draws: int = 512
    tail_low_quantile: float = 0.1
    tail_high_quantile: float = 0.9
    rhat_max_draws: int = 2000

    def __post_init__(self) -> None:
        if not self.max_cross_run_rhat > 1.0:
            raise ValueError("max_cross_run_rhat must exceed 1")
        for name in ("min_kish_ess_per_run", "min_event_ess", "min_selection_ess", "max_shape_log_likelihood_variance"):
            as_float(name, getattr(self, name), positive=True)
        for name in ("max_event_weight_fraction", "max_selection_weight_fraction"):
            value = as_float(name, getattr(self, name), positive=True)
            if value > 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        as_int("n_draws", self.n_draws, minimum=1)
        as_int("rhat_max_draws", self.rhat_max_draws, minimum=4)
        if not 0.0 < self.tail_low_quantile < 0.5 < self.tail_high_quantile < 1.0:
            raise ValueError("tail quantiles must satisfy 0 < low < 0.5 < high < 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def f3(cls) -> "PosteriorGateCriteria":
        return cls(
            max_cross_run_rhat=1.05,
            min_kish_ess_per_run=500.0,
            min_event_ess=10.0,
            min_selection_ess=100.0,
            max_event_weight_fraction=0.35,
            max_selection_weight_fraction=0.15,
            max_shape_log_likelihood_variance=2.0,
        )

    @classmethod
    def f4(cls) -> "PosteriorGateCriteria":
        return cls()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PosteriorGateCriteria":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown posterior gate field(s) {unknown}")
        return cls(**dict(payload))


# (metric key in the diagnostics batch summary, criteria attribute, direction)
_IMPORTANCE = (
    ("min_event_ess", "min_event_ess", ">="),
    ("selection_ess", "min_selection_ess", ">="),
    ("max_event_max_weight", "max_event_weight_fraction", "<="),
    ("selection_max_weight", "max_selection_weight_fraction", "<="),
    ("shape_log_likelihood_variance", "max_shape_log_likelihood_variance", "<="),
)


def _check(name, stage, value, op, limit):
    value = float(value)
    passed = math.isfinite(value) and (value >= limit if op == ">=" else value <= limit)
    return {"name": name, "stage": stage, "value": value, "op": op, "limit": float(limit), "passed": bool(passed)}


def evaluate_posterior_gates(
    results: Sequence,
    posterior,
    selection,
    population_model,
    *,
    criteria: PosteriorGateCriteria | None = None,
    hbi_config=None,
    seed: int = 0,
    batch_size: int = 16,
) -> dict[str, object]:
    """Gate a set of repeated dynesty runs of one model (see module doc)."""
    from gwpop_search.inference.dynesty_backend import build_importance_diagnostics

    from ._common import hbi_config_from_identity, require_identity_matches

    criteria = PosteriorGateCriteria() if criteria is None else criteria
    results = tuple(results)
    if not results:
        raise ValueError("at least one dynesty result is required")
    identity = common_likelihood_identity(results)
    if hbi_config is None:
        hbi_config = hbi_config_from_identity(identity)
    names = tuple(results[0].names)
    require_identity_matches(identity, posterior, selection, population_model, names, hbi_config,
                             what="posterior gates")
    checks = []
    runs = []
    for r in results:
        converged = r.diagnostics.get("converged")
        runs.append({"seed": int(r.seed), "kish_ess": float(r.kish_ess), "converged": converged,
                     "log_evidence": float(r.log_evidence), "log_evidence_error": float(r.log_evidence_error),
                     "sample": r.config.sample, "nlive": int(r.config.nlive)})
        checks.append(_check(f"kish_ess_seed_{r.seed}", "gate", r.kish_ess, ">=", criteria.min_kish_ess_per_run))
        if criteria.require_dlogz_termination:
            checks.append({"name": f"dlogz_termination_seed_{r.seed}", "stage": "gate",
                           "value": converged, "passed": bool(converged)})
    rhat = None
    if len(results) >= 2:
        try:
            rhat = cross_run_rhat(results, max_draws=criteria.rhat_max_draws)
            checks.append(_check("cross_run_rhat", "gate", rhat["max"], "<=", criteria.max_cross_run_rhat))
        except ValueError as exc:
            checks.append({"name": "cross_run_rhat", "stage": "gate", "value": None, "passed": False,
                           "reason": str(exc)})
    else:
        checks.append({"name": "cross_run_rhat", "stage": "gate", "value": None, "passed": False,
                       "reason": "cross-run R-hat needs at least two repeats"})

    pooled = pool_dynesty_results(results)
    median = pooled.weighted_median()
    draws = pooled.equal_weight_draws(criteria.n_draws, seed)
    fn = build_importance_diagnostics(posterior, selection, population_model, names,
                                      hbi_config=hbi_config, batch_size=batch_size)
    batch = fn(np.vstack([median[None, :], draws]))
    stats = batch.summary_statistics()
    importance = {"at_posterior_median": {}, "over_posterior": {}}
    for key, attr, op in _IMPORTANCE:
        limit = getattr(criteria, attr)
        at_point = float(stats[key][0])
        values = np.asarray(stats[key][1:], dtype=np.float64)
        med = float(np.quantile(values, 0.5, method="inverted_cdf"))
        q = criteria.tail_low_quantile if op == ">=" else criteria.tail_high_quantile
        tail = float(np.quantile(values, q, method="inverted_cdf"))
        importance["at_posterior_median"][key] = at_point
        importance["over_posterior"][key] = {
            "median": med, f"q{q:g}": tail, "min": float(np.min(values)), "max": float(np.max(values)),
            "fraction_failing": float(np.mean(values < limit) if op == ">=" else np.mean(values > limit)),
        }
        checks.append(_check(f"{key}_at_posterior_median", "gate", at_point, op, limit))
        checks.append(_check(f"{key}_median_over_posterior", "gate", med, op, limit))
        checks.append(_check(f"{key}_tail_q{q:g}", "gate", tail, op, limit))
    passed = all(check["passed"] for check in checks if check["stage"] == "gate")
    return {
        "format_version": POSTERIOR_GATES_FORMAT,
        "passed": bool(passed),
        "criteria": criteria.to_dict(),
        "checks": checks,
        "runs": runs,
        "cross_run_rhat": rhat,
        "posterior_median": {name: float(median[k]) for k, name in enumerate(names)},
        "importance": {**importance, "n_draws": int(criteria.n_draws), "seed": int(seed),
                       "worst_event_at_median": batch.event_names[int(np.argmin(batch.event_ess[0]))]},
        "pooled_kish_ess": pooled.kish_ess,
    }


def require_repeats(results: Sequence, n: int) -> None:
    if len(results) < n:
        raise AnalysisInputError(f"{len(results)} repeats available; {n} required")
