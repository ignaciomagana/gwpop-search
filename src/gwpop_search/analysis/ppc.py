"""Posterior predictive checks with selection effects (v2 criterion D6).

For each of ``S`` hyperparameter draws ``Lambda_s`` from a model's posterior:

* **Predicted detected catalog.** ``N`` found injections are drawn with
  replacement with probability proportional to the effective exposure weight
  ``u_m(Lambda_s) = p_pop(theta_m | Lambda_s) / p_draw(theta_m) * (T_k / N_k)``
  (the weights of the HBI selection estimator, :mod:`.terms`). The drawn
  ``theta_m`` are samples of the *detected* population
  ``p_det(theta) p_pop(theta | Lambda_s) / A(Lambda_s)``. ``N`` is the number
  of observed events (259 for the v2 primary catalog).
* **Observed catalog.** For every event one PE sample is drawn with
  probability proportional to ``w_ij = p_pop(theta_ij | Lambda_s) / pi_ij``:
  a draw of that event's source parameters under the population ``Lambda_s``
  (the "posterior-reweighted event samples").

Both catalogs are compared through the **pre-declared statistics** (plan
2026-09-30, D6), each a scalar ``T(catalog; Lambda_s)``:

* ``ks_<x>`` for the marginals ``x in (m1, q, chi_eff, z)``: the
  Kolmogorov-Smirnov distance between the catalog's values of ``x`` and the
  predicted *detected* distribution of ``x`` under ``Lambda_s`` (the
  ``u``-weighted empirical CDF of all found injections, evaluated as a
  mid-CDF so that atoms of the discrete reference are handled symmetrically).
  This is a Gelman-Meng-Stern discrepancy: it depends on ``Lambda_s``, so the
  observed and predicted catalogs are compared against the same reference.
* ``spearman_chi_eff_q`` and ``spearman_chi_eff_z``: Spearman rank
  correlations of the catalog (average ranks for ties).

The two-sided posterior predictive p-value of a statistic is::

    p_hi = P(T_pred >= T_obs),  p_lo = P(T_pred <= T_obs)   (ties counted 1/2)
    p    = min(1, 2 min(p_hi, p_lo))

estimated over the ``S`` draws. **Claim criterion (D6):** the model fails the
check if ``p < alpha = 0.01`` for any pre-declared statistic. The Monte-Carlo
standard error of each p is reported, and a p within two standard errors of
``alpha`` is flagged ``borderline`` (the label is still decided by the point
estimate, as pre-declared).

Diagnostics: the Kish ESS of the injection weights at every draw
(``1 / sum omega_m^2``) must be at least ``4 N`` for the predicted catalog to
be a faithful draw (Farr 2019's criterion for the same weights); a check with
more than ``max_low_ess_fraction`` of draws below that is ``unreliable``.

Known approximation (documented, conservative): the observed catalog reuses the
data that produced the posterior (no leave-one-out), which pulls the
reweighted event samples towards the population and makes a failure *less*
likely. The check can therefore miss mild misfits; it does not manufacture
failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from ._common import (
    AnalysisInputError,
    WeightedPosterior,
    as_float,
    as_int,
    hbi_config_from_identity,
    json_ready,
    model_hash_of,
    require_identity_matches,
)
from .terms import CatalogWeightEvaluator, pad_catalog

PPC_FORMAT = "gwpop-search-ppc-1.0"
#: D6 level: a pre-declared statistic with p < PPC_ALPHA is a failure.
PPC_ALPHA = 0.01
#: observable name -> column in the PE and selection catalogs.
DEFAULT_COLUMNS: Mapping[str, str] = {
    "m1": "m1_source",
    "q": "q",
    "chi_eff": "chi_eff",
    "z": "z",
}
MARGINAL_VARIABLES = ("m1", "q", "chi_eff", "z")
CORRELATION_PAIRS = (("chi_eff", "q"), ("chi_eff", "z"))
PREDECLARED_STATISTICS = tuple(f"ks_{x}" for x in MARGINAL_VARIABLES) + tuple(
    f"spearman_{a}_{b}" for a, b in CORRELATION_PAIRS
)
DEFAULT_BAND_LEVELS = tuple(np.round(np.linspace(0.05, 0.95, 19), 4).tolist())


@dataclass(frozen=True)
class PPCConfig:
    """Settings of one posterior predictive check.

    ``n_draws`` hyperparameter draws (systematic resampling of the weighted
    posterior); ``n_catalog`` defaults to the number of observed events.
    ``min_selection_ess_per_event`` is the ``4`` of the ``ESS >= 4 N`` rule;
    ``max_low_ess_fraction`` is the fraction of draws allowed to violate it
    before the check is ``unreliable``.
    """

    n_draws: int = 2000
    seed: int = 0
    alpha: float = PPC_ALPHA
    columns: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_COLUMNS))
    n_catalog: int | None = None
    batch_size: int = 16
    min_selection_ess_per_event: float = 4.0
    max_low_ess_fraction: float = 0.01
    band_levels: tuple[float, ...] = DEFAULT_BAND_LEVELS

    def __post_init__(self) -> None:
        as_int("n_draws", self.n_draws, minimum=2)
        as_int("seed", self.seed, minimum=0)
        alpha = as_float("alpha", self.alpha, positive=True)
        if not alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.n_catalog is not None:
            as_int("n_catalog", self.n_catalog, minimum=2)
        as_int("batch_size", self.batch_size, minimum=1)
        as_float("min_selection_ess_per_event", self.min_selection_ess_per_event, nonnegative=True)
        frac = as_float("max_low_ess_fraction", self.max_low_ess_fraction, nonnegative=True)
        if frac > 1.0:
            raise ValueError("max_low_ess_fraction must lie in [0, 1]")
        columns = {str(k): str(v) for k, v in dict(self.columns).items()}
        missing = [x for x in MARGINAL_VARIABLES if x not in columns]
        if missing:
            raise ValueError(f"columns must map every pre-declared observable; missing {missing}")
        object.__setattr__(self, "columns", columns)
        levels = tuple(float(x) for x in self.band_levels)
        if not levels or any(not 0.0 <= x <= 1.0 for x in levels):
            raise ValueError("band_levels must be quantile levels in [0, 1]")
        object.__setattr__(self, "band_levels", levels)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_draws": int(self.n_draws),
            "seed": int(self.seed),
            "alpha": float(self.alpha),
            "columns": dict(self.columns),
            "n_catalog": self.n_catalog,
            "batch_size": int(self.batch_size),
            "min_selection_ess_per_event": float(self.min_selection_ess_per_event),
            "max_low_ess_fraction": float(self.max_low_ess_fraction),
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def ks_uniform_distance(u) -> float:
    """Kolmogorov-Smirnov distance of values ``u`` in [0, 1] from U(0, 1)."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = u.size
    if n == 0:
        raise ValueError("ks_uniform_distance needs at least one value")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(max(np.max(i / n - u), np.max(u - (i - 1.0) / n)))


def spearman_rho(x, y) -> float:
    """Spearman rank correlation (average ranks for ties; 0 for a constant input)."""
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    if denom == 0.0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def two_sided_ppp(t_obs, t_pred) -> dict[str, float]:
    """Two-sided posterior predictive p-value from paired draws (ties count 1/2)."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    t_pred = np.asarray(t_pred, dtype=np.float64)
    if t_obs.shape != t_pred.shape or t_obs.ndim != 1 or t_obs.size == 0:
        raise ValueError("t_obs and t_pred must be 1-D arrays of equal, non-zero length")
    ties = 0.5 * np.mean(t_pred == t_obs)
    p_hi = float(np.mean(t_pred > t_obs) + ties)
    p_lo = float(np.mean(t_pred < t_obs) + ties)
    p = min(1.0, 2.0 * min(p_hi, p_lo))
    s = t_obs.size
    # the MC standard error of 2 min(p_hi, p_lo) as a binomial proportion
    p_tail = min(p_hi, p_lo)
    se = 2.0 * math.sqrt(max(p_tail * (1.0 - p_tail), 0.0) / s)
    return {"p_upper": p_hi, "p_lower": p_lo, "p_value": p, "mc_standard_error": se}


class _WeightedReference:
    """Mid-CDF of a fixed set of values under draw-dependent weights."""

    def __init__(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64)
        self.order = np.argsort(values, kind="stable")
        self.sorted = values[self.order]

    def mid_cdf(self, weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        cum = np.cumsum(weights[self.order])
        cum = cum / cum[-1]
        left = np.searchsorted(self.sorted, x, side="left")
        right = np.searchsorted(self.sorted, x, side="right")
        below = np.where(left > 0, cum[np.maximum(left - 1, 0)], 0.0)
        upto = np.where(right > 0, cum[np.maximum(right - 1, 0)], 0.0)
        return 0.5 * (below + upto)


def _quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, [0.05, 0.5, 0.95])
    return {"q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2]), "mean": float(values.mean())}


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------


#: source-frame columns the check can derive from detector-frame ones with the
#: population model's own cosmology (``z = z(d_L)``, ``m1_source = m1_det / (1 + z)``)
DERIVABLE_COLUMNS = ("m1_source", "z")


def _derived_source_frame(samples: Mapping[str, np.ndarray], cosmology) -> dict[str, np.ndarray]:
    d_l = np.asarray(samples["luminosity_distance"], dtype=np.float64)
    z = np.asarray(cosmology.z_of_dL(d_l), dtype=np.float64)
    return {"z": z, "m1_source": np.asarray(samples["m1_detector"], dtype=np.float64) / (1.0 + z)}


def observable_samples(data, columns: Mapping[str, str], population_model=None, *, what: str):
    """The observable columns of a PE or selection catalog (derived where needed).

    A missing ``m1_source``/``z`` column is derived from ``m1_detector`` and
    ``luminosity_distance`` with ``population_model.cosmology`` (the
    conversion the model itself applies), and the derivation is reported.
    """
    samples = data.samples
    needed = sorted(set(columns.values()))
    missing = [c for c in needed if c not in samples]
    derived: dict[str, np.ndarray] = {}
    cosmology = getattr(population_model, "cosmology", None)
    derivable = (
        cosmology is not None
        and hasattr(cosmology, "z_of_dL")
        and "m1_detector" in samples
        and "luminosity_distance" in samples
    )
    if missing and derivable and set(missing) <= set(DERIVABLE_COLUMNS):
        derived = _derived_source_frame(samples, cosmology)
        missing = []
    if missing:
        raise AnalysisInputError(
            f"posterior predictive check: observable column(s) {missing} missing from the {what} "
            f"catalog; pass columns= to map the observables {sorted(columns)} to available columns"
        )
    out = {c: np.asarray(samples[c], dtype=np.float64) if c in samples else derived[c] for c in needed}
    return out, sorted(c for c in needed if c not in samples)


def _padded_event_values(posterior, values: np.ndarray, n_max: int) -> np.ndarray:
    """``[N, n_max]`` values in the layout of :func:`.terms.pad_catalog`."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty((posterior.n_events, n_max), dtype=np.float64)
    for i in range(posterior.n_events):
        v = values[posterior.event_slice(i)]
        out[i, : v.size] = v
        out[i, v.size :] = v[0]
    return out


def _softmax_rows(log_w: np.ndarray) -> np.ndarray:
    peak = np.max(log_w, axis=-1, keepdims=True)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    w = np.exp(log_w - safe)
    return w


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


@dataclass
class PPCResult:
    """Per-draw statistics and the derived p-values (``to_dict`` for JSON)."""

    label: str
    model_hash: str | None
    config: PPCConfig
    n_events: int
    n_catalog: int
    t_obs: dict[str, np.ndarray]
    t_pred: dict[str, np.ndarray]
    selection_ess: np.ndarray
    min_event_ess: np.ndarray
    bands: dict[str, dict[str, object]]
    identity_verified: bool
    derived_columns: Mapping[str, list] = field(default_factory=dict)
    #: ``None`` for an untapered likelihood, else the recorded treatment
    #: (always ``"weights_only"``: the check uses the population weights only)
    taper_treatment: str | None = None

    def p_values(self) -> dict[str, dict[str, float]]:
        return {name: two_sided_ppp(self.t_obs[name], self.t_pred[name]) for name in PREDECLARED_STATISTICS}

    def summary(self) -> dict[str, object]:
        alpha = float(self.config.alpha)
        stats = {}
        failed = []
        borderline = []
        for name, p in self.p_values().items():
            is_fail = p["p_value"] < alpha
            near = abs(p["p_value"] - alpha) <= 2.0 * p["mc_standard_error"]
            if is_fail:
                failed.append(name)
            if near:
                borderline.append(name)
            stats[name] = {
                "kind": "ks_marginal" if name.startswith("ks_") else "spearman",
                "observed": _quantiles(self.t_obs[name]),
                "predicted": _quantiles(self.t_pred[name]),
                **p,
                "failed": bool(is_fail),
                "borderline": bool(near),
            }
        floor = self.config.min_selection_ess_per_event * self.n_catalog
        low = float(np.mean(self.selection_ess < floor))
        reliable = low <= self.config.max_low_ess_fraction
        resolvable = self.config.n_draws * alpha >= 5.0
        if failed:
            status = "fail"
        elif not reliable:
            status = "unreliable"
        elif not resolvable:
            status = "insufficient_draws"
        else:
            status = "pass"
        return {
            "status": status,
            "failed_statistics": failed,
            "borderline_statistics": borderline,
            "statistics": stats,
            "diagnostics": {
                "selection_ess": _quantiles(self.selection_ess),
                "selection_ess_min": float(np.min(self.selection_ess)),
                "selection_ess_floor": float(floor),
                "fraction_draws_below_selection_ess_floor": low,
                "reliable": bool(reliable),
                "min_event_ess": _quantiles(self.min_event_ess),
                "alpha_resolvable": bool(resolvable),
            },
        }

    def to_dict(self, *, include_draws: bool = True) -> dict[str, object]:
        payload = {
            "format_version": PPC_FORMAT,
            "label": self.label,
            "model_hash": self.model_hash,
            "identity_verified": bool(self.identity_verified),
            "derived_columns": dict(self.derived_columns),
            "taper_treatment": self.taper_treatment,
            "config": self.config.to_dict(),
            "n_events": int(self.n_events),
            "n_catalog": int(self.n_catalog),
            "predeclared_statistics": list(PREDECLARED_STATISTICS),
            "criterion": (
                f"D6: fail if any pre-declared statistic has a two-sided posterior predictive "
                f"p < {self.config.alpha:g}"
            ),
            "method": (
                "predicted catalog: N found injections resampled with p_pop/p_draw*(T_k/N_k) "
                "weights; observed catalog: one population-reweighted PE sample per event; "
                "marginals: KS distance to the predicted detected distribution under the same "
                "draw (mid-CDF of the weighted injections); correlations: Spearman rho; "
                "p = min(1, 2 min(P(T_pred>=T_obs), P(T_pred<=T_obs))), ties 1/2; "
                "no leave-one-out (conservative)"
            ),
            **self.summary(),
            "bands": self.bands,
        }
        if include_draws:
            payload["draws"] = {
                "t_obs": {k: v for k, v in self.t_obs.items()},
                "t_pred": {k: v for k, v in self.t_pred.items()},
            }
        return json_ready(payload)


def posterior_predictive_check(
    sample: WeightedPosterior,
    posterior,
    selection,
    population_model,
    *,
    config: PPCConfig | None = None,
    hbi_config=None,
    label: str | None = None,
    verify_identity: bool = True,
    backend: str = "jax",
) -> PPCResult:
    """Run the pre-declared D6 posterior predictive check of one model.

    ``sample`` is the model's weighted posterior (pooled dynesty runs);
    ``posterior``/``selection`` the catalogs it was sampled on (the identity is
    verified unless ``verify_identity=False``, which only mock tests use);
    ``population_model`` the compiled model. Nothing about the likelihood is
    changed: the weights are those of :mod:`.terms`.
    """
    config = PPCConfig() if config is None else config
    identity = sample.likelihood_identity
    if hbi_config is None:
        from gwpop_search.hbi import HBIConfig

        hbi_config = (
            hbi_config_from_identity(identity) if identity is not None else HBIConfig(selection_chunk_size=None)
        )
    if verify_identity:
        require_identity_matches(
            identity, posterior, selection, population_model, sample.names, hbi_config,
            what=f"posterior predictive check of {label or 'model'}",
        )
    columns = dict(config.columns)
    pe_columns, pe_derived = observable_samples(posterior, columns, population_model, what="PE")
    sel_columns, sel_derived = observable_samples(selection, columns, population_model, what="selection")
    # The predicted and observed catalogs use the population weights only, which
    # do not depend on a variance taper; the posterior draws already carry it.
    taper_treatment = "weights_only" if getattr(hbi_config, "variance_taper", None) is not None else None
    catalog = pad_catalog(
        posterior, selection, population_model, hbi_config=hbi_config, taper_treatment=taper_treatment
    )
    n_events = catalog.n_events
    n_catalog = n_events if config.n_catalog is None else int(config.n_catalog)
    n_sel = int(catalog.n_selected)
    variables = tuple(columns)
    pe_values = {x: _padded_event_values(posterior, pe_columns[columns[x]], catalog.n_max) for x in variables}
    sel_values = {x: sel_columns[columns[x]] for x in variables}
    references = {x: _WeightedReference(sel_values[x]) for x in MARGINAL_VARIABLES}

    X = sample.equal_weight_draws(config.n_draws, config.seed)
    rng = np.random.default_rng(np.random.SeedSequence([int(config.seed), 0x9C9C]))
    evaluator = CatalogWeightEvaluator(
        catalog, population_model, sample.names, batch_size=config.batch_size, backend=backend
    )
    S = X.shape[0]
    t_obs = {name: np.empty(S) for name in PREDECLARED_STATISTICS}
    t_pred = {name: np.empty(S) for name in PREDECLARED_STATISTICS}
    sel_ess = np.empty(S)
    min_event_ess = np.empty(S)
    levels = np.asarray(config.band_levels)
    q_obs = {x: np.empty((S, levels.size)) for x in MARGINAL_VARIABLES}
    q_pred = {x: np.empty((S, levels.size)) for x in MARGINAL_VARIABLES}
    rows = np.arange(n_events)

    for start in range(0, S, config.batch_size):
        block = X[start : start + config.batch_size]
        lw_e, lu = evaluator.log_weights(block)
        for b in range(block.shape[0]):
            s = start + b
            w_e = _softmax_rows(lw_e[b])  # [N, n_max]
            tot_e = w_e.sum(axis=1)
            if np.any(~(tot_e > 0.0)):
                bad = [catalog.event_names[i] for i in np.flatnonzero(~(tot_e > 0.0))[:5]]
                raise ValueError(
                    f"posterior draw {s} gives zero population support to event(s) {bad}; "
                    "it cannot belong to the posterior"
                )
            omega_e = w_e / tot_e[:, None]
            min_event_ess[s] = float(np.min(1.0 / np.sum(omega_e * omega_e, axis=1)))
            cum_e = np.cumsum(omega_e, axis=1)
            u = rng.random(n_events)
            j = np.minimum((cum_e < u[:, None]).sum(axis=1), catalog.pe_counts - 1)
            w_s = _softmax_rows(lu[b][:n_sel])
            tot_s = w_s.sum()
            if not tot_s > 0.0:
                raise ValueError(f"posterior draw {s} gives zero weight to every found injection")
            omega_s = w_s / tot_s
            sel_ess[s] = float(1.0 / np.sum(omega_s * omega_s))
            cum_s = np.cumsum(omega_s)
            idx = np.searchsorted(cum_s, rng.random(n_catalog) * cum_s[-1], side="right")
            idx = np.minimum(idx, n_sel - 1)
            obs = {x: pe_values[x][rows, j] for x in variables}
            pred = {x: sel_values[x][idx] for x in variables}
            for x in MARGINAL_VARIABLES:
                ref = references[x]
                t_obs[f"ks_{x}"][s] = ks_uniform_distance(ref.mid_cdf(omega_s, obs[x]))
                t_pred[f"ks_{x}"][s] = ks_uniform_distance(ref.mid_cdf(omega_s, pred[x]))
                q_obs[x][s] = np.quantile(obs[x], levels)
                q_pred[x][s] = np.quantile(pred[x], levels)
            for a, c in CORRELATION_PAIRS:
                name = f"spearman_{a}_{c}"
                t_obs[name][s] = spearman_rho(obs[a], obs[c])
                t_pred[name][s] = spearman_rho(pred[a], pred[c])

    bands = {}
    for x in MARGINAL_VARIABLES:
        bands[x] = {
            "column": columns[x],
            "levels": levels.tolist(),
            "observed": {k: np.quantile(q_obs[x], p, axis=0).tolist() for k, p in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))},
            "predicted": {k: np.quantile(q_pred[x], p, axis=0).tolist() for k, p in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))},
        }
    return PPCResult(
        label=str(label) if label is not None else "model",
        model_hash=model_hash_of(identity),
        config=config,
        n_events=n_events,
        n_catalog=n_catalog,
        t_obs=t_obs,
        t_pred=t_pred,
        selection_ess=sel_ess,
        min_event_ess=min_event_ess,
        bands=bands,
        identity_verified=bool(verify_identity),
        derived_columns={"pe": pe_derived, "selection": sel_derived},
        taper_treatment=taper_treatment,
    )


def ppc_criterion(payload: Mapping[str, object] | None, *, alpha: float = PPC_ALPHA) -> dict[str, object]:
    """D6 status from a :meth:`PPCResult.to_dict` payload.

    ``pass`` only if every pre-declared statistic was evaluated with
    ``p >= alpha`` on a reliable, resolvable check; ``fail`` if any has
    ``p < alpha``; otherwise ``missing``/``incomplete``.
    """
    if payload is None:
        return {"status": "missing", "reason": "no posterior predictive check supplied"}
    if payload.get("format_version") != PPC_FORMAT:
        return {"status": "incomplete", "reason": f"unsupported PPC format {payload.get('format_version')!r}"}
    stats = payload.get("statistics") or {}
    absent = [name for name in PREDECLARED_STATISTICS if name not in stats]
    if absent:
        return {"status": "incomplete", "reason": f"pre-declared statistics missing: {absent}"}
    used_alpha = float((payload.get("config") or {}).get("alpha", alpha))
    if abs(used_alpha - alpha) > 1e-12:
        return {"status": "incomplete", "reason": f"PPC evaluated at alpha={used_alpha}, D6 requires {alpha}"}
    p_values = {name: float(stats[name]["p_value"]) for name in PREDECLARED_STATISTICS}
    failed = sorted(name for name, p in p_values.items() if p < alpha)
    detail = {
        "p_values": p_values,
        "failed_statistics": failed,
        "borderline_statistics": list(payload.get("borderline_statistics") or []),
        "n_draws": (payload.get("config") or {}).get("n_draws"),
        "identity_verified": payload.get("identity_verified"),
    }
    if failed:
        return {"status": "fail", **detail}
    diagnostics = payload.get("diagnostics") or {}
    if not diagnostics.get("reliable", False):
        return {"status": "incomplete", "reason": "selection ESS below 4N at too many draws", **detail}
    if not diagnostics.get("alpha_resolvable", False):
        return {"status": "incomplete", "reason": "too few draws to resolve alpha", **detail}
    if not payload.get("identity_verified", False):
        return {"status": "incomplete", "reason": "PPC ran without likelihood-identity verification", **detail}
    return {"status": "pass", **detail}


def ppc_report(results: Sequence[PPCResult], *, include_draws: bool = True) -> dict[str, object]:
    """Several models' checks keyed by model hash (or label)."""
    models = {}
    for item in results:
        key = item.model_hash or item.label
        if key in models:
            raise AnalysisInputError(f"duplicate PPC result for {key}")
        models[key] = item.to_dict(include_draws=include_draws)
    return {"format_version": PPC_FORMAT + "-report", "models": models}
