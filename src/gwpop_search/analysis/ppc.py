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
* **Width-sensitive statistics** (operator decision 2026-10-02, pilot (b):
  the six statistics above passed a root R0 with a constant chi_eff width on
  a mock whose true width depends on q, so they do not test how the chi_eff
  *spread* varies across the catalog). For each conditioning observable
  ``x in (q, z, m1)``:

  - ``iqr_chi_eff_<x>_t<k>`` (``k = 1, 2, 3``): the interquartile range of
    the catalog's chi_eff values among the events in the ``k``-th tercile of
    ``x`` (catalog sorted by ``x``, split into three equal-count groups,
    ``t1`` = lowest ``x``; linear-interpolation quantiles);
  - ``spearman_absdev_chi_eff_<x>``: the Spearman correlation of ``x`` with
    ``|chi_eff - median(chi_eff)|`` (the catalog's own median), a
    rank-based width-trend statistic.

  These are computed identically on the observed catalog (the
  population-reweighted PE draws) and on the predicted detected catalog.
  They are statistics of *source* values, not of PE point estimates: a
  point estimate carries the event's measurement scatter, which the
  predicted catalog of found injections does not, so comparing point
  estimates with injections would fail every model; the reweighted event
  draws are the latent-variable discrepancy of Gelman, Meng & Stern (1996).

The posterior predictive p-value of a statistic, estimated over the ``S``
draws (ties counted 1/2), is

* one-sided for the KS distances: ``p = P(T_pred >= T_obs)``. A distance is
  a discrepancy: only an observed catalog *farther* from the reference than
  the predicted ones indicates misfit. A small ``P(T_pred <= T_obs)`` means
  the observed catalog fits *better* than predicted, which is what the reuse
  of the data (below) produces for a correct model, so it is not a failure;
* two-sided for the Spearman correlations and for the width statistics (a
  misfit can have either sign: a spread too large or too small):
  ``p = min(1, 2 min(P(T_pred >= T_obs), P(T_pred <= T_obs)))``.

**Claim criterion (D6):** the model fails the check if ``p < alpha = 0.01``
for any of the six **binding** statistics (:data:`BINDING_STATISTICS`: the
four KS distances and the two Spearman correlations). The twelve width
statistics (:data:`REPORTED_STATISTICS`) are computed, reported with their
p-values and flagged when below ``alpha``, but are **not binding** (operator
decision 2026-10-02: on catalogs measured like GWTC-5 they cannot reject a
constant-width root, see the measured limitation below, while they add to
the family-wise false-fail rate). The Monte-Carlo standard error of each p
is reported, and a binding p within two standard errors of ``alpha`` is
flagged ``borderline`` (the label is still decided by the point estimate, as
pre-declared).

**Multiplicity (explicit).** ``alpha`` stays 0.01 *per statistic* with no
Bonferroni-style correction, over the six binding statistics. Reason: a D6
failure can only *remove* a SUPPORTED label (D6 is required for SUPPORTED,
never for DISFAVOURED), so a family-wise false fail costs power, never a
false claim. The family-wise false-fail rate of a correct model is disclosed
two ways with every result:

* the independence bound ``1 - (1 - alpha)^6 = 5.9%`` over the binding
  statistics (16.5% if the twelve reported width statistics were binding;
  the statistics are positively correlated, so the true rates are lower);
* an empirical estimate from the predicted replicates themselves: each
  predicted catalog ``r`` is scored as if it were the observed one against
  the other ``S - 1`` predicted catalogs (same sidedness, ties 1/2), and the
  rate is the fraction of replicates with any binding statistic below
  ``alpha`` (per family also for the reported width family). This keeps the
  full correlation structure of the
  statistics; it ignores the data reuse below, which makes the real
  posterior predictive p-values more conservative, so it is an upper-side
  estimate for a correct model.
The check is resolvable only if the rarest tail probability it tests
(``alpha / 2`` for the two-sided statistics) has at least 5 expected counts:
``n_draws * alpha / 2 >= 5``, i.e. at least 1000 draws at ``alpha = 0.01``.

Diagnostics: the Kish ESS of the injection weights at every draw
(``1 / sum omega_m^2``) must be at least ``4 N`` for the predicted catalog to
be a faithful draw (Farr 2019's criterion for the same weights); a check with
more than ``max_low_ess_fraction`` of draws below that is ``unreliable``.

Source-frame observables (``m1_source``, ``z``) are always derived from the
detector-frame columns (``m1_detector``, ``luminosity_distance``) with the
population model's cosmology, the conversion the likelihood itself applies;
stored source-frame columns (which may be at another cosmology) are ignored
and their largest relative difference from the derived values is reported.

Known approximation (documented): the observed catalog reuses the data that
produced the posterior (no leave-one-out), which pulls the reweighted event
samples towards the population. For the one-sided KS discrepancies this makes
a failure *less* likely (the check can miss mild misfits but does not
manufacture failures); for the two-sided correlations the pull is towards the
predicted correlation, again away from either tail.

**Measured limitation of the width statistics (pilot (b) closure mock,
2026-10-02; staging/v2/b3_investigation/ppc_revalidation).** When the
per-event chi_eff measurement width (median PE std 0.13 on the mock) is
comparable to or larger than the population width (0.06 at q ~ 1 to 0.15 at
q ~ 0.5), a population-reweighted event draw is dominated by the model's own
population: the root R0 (constant width) fitted to the width(q) closure mock
retains only 9-28% of the true-catalog deviation of the q-tercile spreads
and 14% of the Spearman(q, |chi_eff - median|) trend (-0.064 observed vs
-0.004 predicted; the true source values give -0.44), and passes every width
statistic (smallest p = 0.49 at 2000 draws), although the same statistics of
the true source values reject it (two-sided p < 0.001: none of the
2000 predicted catalogs is as extreme). The width statistics detect a
width misfit only when the events resolve the width (well-measured toys,
``tests/test_analysis_ppc.py``); on catalogs measured like GWTC-5, D6 cannot
be relied on to reject a constant-width root. A data-space check (point
estimates against predicted catalogs with simulated measurement noise) is
the candidate replacement; it needs a measurement model for the injections
and has not been pre-declared.
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

#: 1.1: one-sided KS p, derived source frame; 1.2: width statistics (binding);
#: 1.3: width statistics reported only, D6 decided by the six binding statistics
PPC_FORMAT = "gwpop-search-ppc-1.3"
#: payload formats :func:`ppc_criterion` reads (a 1.2 payload carries every
#: statistic and its p-value; D6 is re-decided on the binding six)
PPC_ACCEPTED_FORMATS = ("gwpop-search-ppc-1.2", PPC_FORMAT)
#: D6 level: a pre-declared statistic with p < PPC_ALPHA is a failure.
PPC_ALPHA = 0.01
#: expected tail counts needed to resolve the smallest tested tail (alpha / 2)
PPC_MIN_TAIL_COUNTS = 5.0
#: observable name -> column in the PE and selection catalogs.
DEFAULT_COLUMNS: Mapping[str, str] = {
    "m1": "m1_source",
    "q": "q",
    "chi_eff": "chi_eff",
    "z": "z",
}
MARGINAL_VARIABLES = ("m1", "q", "chi_eff", "z")
CORRELATION_PAIRS = (("chi_eff", "q"), ("chi_eff", "z"))
#: width statistics: chi_eff spread conditioned on each of these observables
WIDTH_TARGET = "chi_eff"
WIDTH_CONDITIONING = ("q", "z", "m1")
#: equal-count bins of the conditioning observable (terciles)
WIDTH_N_BINS = 3
MARGINAL_STATISTICS = tuple(f"ks_{x}" for x in MARGINAL_VARIABLES)
CORRELATION_STATISTICS = tuple(f"spearman_{a}_{b}" for a, b in CORRELATION_PAIRS)
WIDTH_IQR_STATISTICS = tuple(
    f"iqr_{WIDTH_TARGET}_{x}_t{k + 1}" for x in WIDTH_CONDITIONING for k in range(WIDTH_N_BINS)
)
WIDTH_TREND_STATISTICS = tuple(f"spearman_absdev_{WIDTH_TARGET}_{x}" for x in WIDTH_CONDITIONING)
WIDTH_STATISTICS = WIDTH_IQR_STATISTICS + WIDTH_TREND_STATISTICS
#: D6 is decided by these six (operator decision 2026-10-02)
BINDING_STATISTICS = MARGINAL_STATISTICS + CORRELATION_STATISTICS
#: computed and reported with every check, never binding (operator decision 2026-10-02)
REPORTED_STATISTICS = WIDTH_STATISTICS
#: every pre-declared statistic (binding and reported)
PREDECLARED_STATISTICS = BINDING_STATISTICS + REPORTED_STATISTICS
#: statistic families (multiplicity is disclosed overall and per family)
STATISTIC_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "marginal": MARGINAL_STATISTICS,
    "correlation": CORRELATION_STATISTICS,
    "width": WIDTH_STATISTICS,
}
#: statistics tested one-sided (distances: only a larger observed distance is a misfit)
ONE_SIDED_STATISTICS = MARGINAL_STATISTICS


def statistic_kind(name: str) -> str:
    """``ks_marginal`` / ``spearman`` / ``width_iqr`` / ``width_spearman_absdev``."""
    if name in MARGINAL_STATISTICS:
        return "ks_marginal"
    if name in WIDTH_IQR_STATISTICS:
        return "width_iqr"
    if name in WIDTH_TREND_STATISTICS:
        return "width_spearman_absdev"
    return "spearman"


def family_wise_false_fail_bound(alpha: float = PPC_ALPHA, n_statistics: int = len(BINDING_STATISTICS)) -> float:
    """``1 - (1 - alpha)^n``: the family-wise false-fail rate for independent statistics.

    For positively correlated statistics (the pre-declared ones) the true rate
    is lower; :func:`replicate_family_wise_rate` estimates it with the
    correlations kept.
    """
    return 1.0 - (1.0 - float(alpha)) ** int(n_statistics)


def ppc_draws_resolve_alpha(n_draws: int, alpha: float = PPC_ALPHA) -> bool:
    """``n_draws * alpha / 2 >= 5``: the two-sided tail has at least 5 expected counts."""
    return float(n_draws) * float(alpha) / 2.0 >= PPC_MIN_TAIL_COUNTS
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


def tercile_spreads(x, y, n_bins: int = WIDTH_N_BINS) -> np.ndarray:
    """Interquartile range of ``y`` in each equal-count bin of ``x`` (lowest ``x`` first).

    The catalog is sorted by ``x`` (stable) and split with
    ``np.array_split`` (bin sizes differ by at most one).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if x.size < 2 * n_bins:
        raise ValueError(f"need at least {2 * n_bins} catalog entries for {n_bins} bins")
    order = np.argsort(x, kind="stable")
    out = np.empty(n_bins)
    for k, idx in enumerate(np.array_split(order, n_bins)):
        lo, hi = np.quantile(y[idx], [0.25, 0.75])
        out[k] = hi - lo
    return out


def absdev_trend(x, y) -> float:
    """Spearman correlation of ``x`` with ``|y - median(y)|`` (a rank width trend)."""
    y = np.asarray(y, dtype=np.float64)
    return spearman_rho(x, np.abs(y - np.median(y)))


def width_statistics(catalog: Mapping[str, np.ndarray]) -> dict[str, float]:
    """The twelve width statistics of one catalog (``{observable: values}``)."""
    out: dict[str, float] = {}
    target = catalog[WIDTH_TARGET]
    for x in WIDTH_CONDITIONING:
        spreads = tercile_spreads(catalog[x], target)
        for k in range(WIDTH_N_BINS):
            out[f"iqr_{WIDTH_TARGET}_{x}_t{k + 1}"] = float(spreads[k])
        out[f"spearman_absdev_{WIDTH_TARGET}_{x}"] = absdev_trend(catalog[x], target)
    return out


def replicate_p_values(name: str, t_pred) -> np.ndarray:
    """p-value of every predicted replicate scored against the other ``S - 1`` (ties 1/2).

    Same sidedness as :func:`statistic_ppp`. Used to estimate the family-wise
    false-fail rate with the statistics' correlations kept.
    """
    t = np.asarray(t_pred, dtype=np.float64)
    s = t.size
    if s < 2:
        raise ValueError("need at least two replicates")
    srt = np.sort(t)
    below = np.searchsorted(srt, t, side="left")              # others strictly below
    equal = np.searchsorted(srt, t, side="right") - below - 1  # others tied (minus itself)
    above = s - 1 - below - equal
    p_hi = (above + 0.5 * equal) / (s - 1)
    p_lo = (below + 0.5 * equal) / (s - 1)
    if name in ONE_SIDED_STATISTICS:
        return p_hi
    return np.minimum(1.0, 2.0 * np.minimum(p_hi, p_lo))


def replicate_family_wise_rate(t_pred: Mapping[str, np.ndarray], alpha: float = PPC_ALPHA,
                               names: Sequence[str] = BINDING_STATISTICS) -> dict[str, object]:
    """Empirical family-wise false-fail rate from the predicted replicates.

    Each replicate is treated as the observed catalog of a correct model; the
    rate is the fraction of replicates with any statistic at ``p < alpha``,
    overall and per :data:`STATISTIC_FAMILIES` family, with the per-statistic
    rates and the binomial standard error of the overall rate.
    """
    names = [n for n in names if n in t_pred]
    fails = {n: replicate_p_values(n, t_pred[n]) < float(alpha) for n in names}
    s = len(next(iter(t_pred.values())))
    any_fail = np.zeros(s, dtype=bool)
    for n in names:
        any_fail |= fails[n]
    rate = float(any_fail.mean())
    per_family = {}
    for fam, members in STATISTIC_FAMILIES.items():
        members = [n for n in members if n in fails]
        if members:
            f = np.zeros(s, dtype=bool)
            for n in members:
                f |= fails[n]
            per_family[fam] = float(f.mean())
    return {
        "rate": rate,
        "standard_error": math.sqrt(max(rate * (1.0 - rate), 0.0) / s),
        "n_replicates": int(s),
        "per_family": per_family,
        "per_statistic": {n: float(fails[n].mean()) for n in names},
        "method": "each predicted replicate scored against the other S-1 (same sidedness, ties 1/2); "
                  "keeps the statistics' correlations; ignores the data reuse (upper-side for a correct model)",
    }


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
    return {"p_upper": p_hi, "p_lower": p_lo, "p_value": p, "mc_standard_error": se,
            "sidedness": "two_sided"}


def upper_tail_ppp(t_obs, t_pred) -> dict[str, float]:
    """One-sided posterior predictive p-value ``P(T_pred >= T_obs)`` (ties count 1/2).

    For a discrepancy (a distance): only an observed value above the predicted
    ones indicates misfit.
    """
    two = two_sided_ppp(t_obs, t_pred)
    p = two["p_upper"]
    se = math.sqrt(max(p * (1.0 - p), 0.0) / np.asarray(t_obs).size)
    return {"p_upper": two["p_upper"], "p_lower": two["p_lower"], "p_value": p,
            "mc_standard_error": se, "sidedness": "upper"}


def statistic_ppp(name: str, t_obs, t_pred) -> dict[str, float]:
    """The pre-declared p-value of one statistic (one-sided for KS, two-sided otherwise)."""
    if name in ONE_SIDED_STATISTICS:
        return upper_tail_ppp(t_obs, t_pred)
    return two_sided_ppp(t_obs, t_pred)


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
    """The observable columns of a PE or selection catalog.

    ``m1_source``/``z`` are derived from ``m1_detector`` and
    ``luminosity_distance`` with ``population_model.cosmology`` (the
    conversion the likelihood itself applies) whenever that is possible,
    *even if the catalog stores them*: stored source-frame columns can be at
    another cosmology (the v2r2 products store LAL Planck15). Returns
    ``(values, derived column names, {stored column: max relative difference
    from the derived values}``); a stored column is used only when the
    derivation is impossible.
    """
    samples = data.samples
    needed = sorted(set(columns.values()))
    cosmology = getattr(population_model, "cosmology", None)
    derivable = (
        cosmology is not None
        and hasattr(cosmology, "z_of_dL")
        and "m1_detector" in samples
        and "luminosity_distance" in samples
    )
    derived: dict[str, np.ndarray] = {}
    if derivable and set(needed) & set(DERIVABLE_COLUMNS):
        derived = {c: v for c, v in _derived_source_frame(samples, cosmology).items() if c in needed}
    missing = [c for c in needed if c not in samples and c not in derived]
    if missing:
        raise AnalysisInputError(
            f"posterior predictive check: observable column(s) {missing} missing from the {what} "
            f"catalog; pass columns= to map the observables {sorted(columns)} to available columns"
        )
    differences: dict[str, float] = {}
    for c, values in derived.items():
        if c in samples:
            stored = np.asarray(samples[c], dtype=np.float64)
            scale = np.maximum(np.abs(values), 1e-300)
            differences[c] = float(np.max(np.abs(stored - values) / scale)) if values.size else 0.0
    out = {c: derived[c] if c in derived else np.asarray(samples[c], dtype=np.float64) for c in needed}
    return out, sorted(derived), differences


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
    #: catalog -> {column: max |stored - derived| / |derived|} for stored
    #: source-frame columns that were replaced by the derived ones
    stored_column_difference: Mapping[str, dict] = field(default_factory=dict)

    def p_values(self) -> dict[str, dict[str, float]]:
        return {name: statistic_ppp(name, self.t_obs[name], self.t_pred[name]) for name in PREDECLARED_STATISTICS}

    def summary(self) -> dict[str, object]:
        alpha = float(self.config.alpha)
        stats = {}
        failed = []
        borderline = []
        reported_below = []
        for name, p in self.p_values().items():
            binding = name in BINDING_STATISTICS
            below = p["p_value"] < alpha
            near = abs(p["p_value"] - alpha) <= 2.0 * p["mc_standard_error"]
            if binding and below:
                failed.append(name)
            if binding and near:
                borderline.append(name)
            if not binding and below:
                reported_below.append(name)
            stats[name] = {
                "kind": statistic_kind(name),
                "binding": bool(binding),
                "observed": _quantiles(self.t_obs[name]),
                "predicted": _quantiles(self.t_pred[name]),
                **p,
                "below_alpha": bool(below),
                "failed": bool(binding and below),
                "borderline": bool(binding and near),
            }
        floor = self.config.min_selection_ess_per_event * self.n_catalog
        low = float(np.mean(self.selection_ess < floor))
        reliable = low <= self.config.max_low_ess_fraction
        resolvable = ppc_draws_resolve_alpha(self.config.n_draws, alpha)
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
            "reported_statistics_below_alpha": reported_below,
            "binding_statistics": list(BINDING_STATISTICS),
            "reported_statistics": list(REPORTED_STATISTICS),
            "statistics": stats,
            "diagnostics": {
                "selection_ess": _quantiles(self.selection_ess),
                "selection_ess_min": float(np.min(self.selection_ess)),
                "selection_ess_floor": float(floor),
                "fraction_draws_below_selection_ess_floor": low,
                "reliable": bool(reliable),
                "min_event_ess": _quantiles(self.min_event_ess),
                "alpha_resolvable": bool(resolvable),
                "stored_source_frame_max_relative_difference": dict(self.stored_column_difference),
            },
            "multiplicity": {
                "correction": "none: alpha per statistic (a D6 false fail can only remove SUPPORTED)",
                "n_statistics": len(BINDING_STATISTICS),
                "n_statistics_reported_only": len(REPORTED_STATISTICS),
                "families": {k: list(v) for k, v in STATISTIC_FAMILIES.items()},
                "binding_families": ["marginal", "correlation"],
                "family_wise_false_fail_upper_bound": family_wise_false_fail_bound(
                    alpha, len(BINDING_STATISTICS)),
                "family_wise_false_fail_upper_bound_if_width_binding": family_wise_false_fail_bound(
                    alpha, len(PREDECLARED_STATISTICS)),
                "family_wise_false_fail_independence_bound_per_family": {
                    k: family_wise_false_fail_bound(alpha, len(v)) for k, v in STATISTIC_FAMILIES.items()},
                "family_wise_false_fail_empirical": replicate_family_wise_rate(self.t_pred, alpha),
                "family_wise_false_fail_empirical_all_statistics": replicate_family_wise_rate(
                    self.t_pred, alpha, names=PREDECLARED_STATISTICS),
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
                f"D6: fail if any of the six binding statistics (four KS distances, two Spearman "
                f"correlations) has a posterior predictive p < {self.config.alpha:g} (one-sided "
                "P(T_pred >= T_obs) for the KS distances, two-sided for the Spearman correlations; "
                "alpha per statistic, family-wise false-fail rate disclosed); the twelve width "
                "statistics (two-sided) are reported only, not binding (operator decision 2026-10-02)"
            ),
            "method": (
                "predicted catalog: N found injections resampled with p_pop/p_draw*(T_k/N_k) "
                "weights; observed catalog: one population-reweighted PE sample per event; "
                "marginals: KS distance to the predicted detected distribution under the same "
                "draw (mid-CDF of the weighted injections); correlations: Spearman rho; width: "
                "IQR of chi_eff in equal-count terciles of q, z, m1 and Spearman(x, |chi_eff - "
                "median chi_eff|); "
                "KS: p = P(T_pred>=T_obs); Spearman: p = min(1, 2 min(P(T_pred>=T_obs), "
                "P(T_pred<=T_obs))); ties 1/2; source frame derived from (m1_detector, d_L) at the "
                "population cosmology; no leave-one-out"
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
    pe_columns, pe_derived, pe_diff = observable_samples(posterior, columns, population_model, what="PE")
    sel_columns, sel_derived, sel_diff = observable_samples(
        selection, columns, population_model, what="selection"
    )
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
            for name, value in width_statistics(obs).items():
                t_obs[name][s] = value
            for name, value in width_statistics(pred).items():
                t_pred[name][s] = value

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
        stored_column_difference={"pe": pe_diff, "selection": sel_diff},
    )


def ppc_criterion(payload: Mapping[str, object] | None, *, alpha: float = PPC_ALPHA) -> dict[str, object]:
    """D6 status from a :meth:`PPCResult.to_dict` payload.

    ``pass`` only if every binding statistic (:data:`BINDING_STATISTICS`) was
    evaluated with ``p >= alpha`` on a reliable, resolvable check; ``fail`` if
    any has ``p < alpha``; otherwise ``missing``/``incomplete``. The width
    statistics (:data:`REPORTED_STATISTICS`) are reported (their p-values and
    those below ``alpha``) and never change the status (operator decision
    2026-10-02); a payload in format 1.2 (width statistics then binding) is
    re-decided on the binding six.
    """
    if payload is None:
        return {"status": "missing", "reason": "no posterior predictive check supplied"}
    if payload.get("format_version") not in PPC_ACCEPTED_FORMATS:
        return {"status": "incomplete", "reason": f"unsupported PPC format {payload.get('format_version')!r}"}
    stats = payload.get("statistics") or {}
    absent = [name for name in BINDING_STATISTICS if name not in stats]
    if absent:
        return {"status": "incomplete", "reason": f"binding statistics missing: {absent}"}
    used_alpha = float((payload.get("config") or {}).get("alpha", alpha))
    if abs(used_alpha - alpha) > 1e-12:
        return {"status": "incomplete", "reason": f"PPC evaluated at alpha={used_alpha}, D6 requires {alpha}"}
    p_values = {name: float(stats[name]["p_value"]) for name in BINDING_STATISTICS}
    failed = sorted(name for name, p in p_values.items() if p < alpha)
    reported = {name: float(stats[name]["p_value"]) for name in REPORTED_STATISTICS if name in stats}
    binding_borderline = [name for name in (payload.get("borderline_statistics") or [])
                          if name in BINDING_STATISTICS]
    empirical = (payload.get("multiplicity") or {}).get("family_wise_false_fail_empirical") or {}
    if payload.get("format_version") != PPC_FORMAT:  # 1.2: the empirical rate was over all 18
        empirical = {}
    detail = {
        "p_values": p_values,
        "failed_statistics": failed,
        "borderline_statistics": binding_borderline,
        "reported_p_values": reported,
        "reported_statistics_below_alpha": sorted(name for name, p in reported.items() if p < alpha),
        "reported_statistics_missing": [name for name in REPORTED_STATISTICS if name not in stats],
        "binding": "the six statistics of BINDING_STATISTICS; the width statistics are reported only "
                   "(operator decision 2026-10-02)",
        "n_draws": (payload.get("config") or {}).get("n_draws"),
        "identity_verified": payload.get("identity_verified"),
        "sidedness": {name: stats[name].get("sidedness") for name in BINDING_STATISTICS},
        "family_wise_false_fail_upper_bound": family_wise_false_fail_bound(alpha),
        "family_wise_false_fail_empirical": empirical.get("rate"),
    }
    if failed:
        return {"status": "fail", **detail}
    diagnostics = payload.get("diagnostics") or {}
    if not diagnostics.get("reliable", False):
        return {"status": "incomplete", "reason": "selection ESS below 4N at too many draws", **detail}
    n_draws = (payload.get("config") or {}).get("n_draws")
    if not diagnostics.get("alpha_resolvable", False) or n_draws is None \
            or not ppc_draws_resolve_alpha(int(n_draws), alpha):
        return {"status": "incomplete",
                "reason": "too few draws to resolve alpha / 2 (need n_draws * alpha / 2 >= 5)", **detail}
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
