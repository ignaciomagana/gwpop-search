"""Savage–Dickey cross-checks of nested-sampling Bayes factors.

For nested models ``M_S ⊂ M_L`` with ``M_L(omega = omega_0, psi) = M_S(psi)``
(MODEL_COMPARISON_MATH.md Sec. 4)::

    Z_S / Z_L = [ p_L(omega_0 | D) / pi_L(omega_0) ] * VW,
    VW = E_{psi ~ p_L(psi | omega_0, D)} [ pi_S(psi) / pi_L(psi | omega_0) ]
         (needs supp pi_S ⊆ supp pi_L(. | omega_0)),

so ``ln BF_{L/S} = ln pi_L(omega_0) - ln p_L(omega_0 | D) - ln VW`` with
``VW = 1`` when the priors of the shared parameters are identical (the plain
SDDR is then exact). Parameters of ``M_L`` that are unidentified at the null
(the logistic transition and width, a removed peak's location and width) are
integrated over; they do not break the identity. The SDDR is an alternative
estimator of the same ``Z_S/Z_L`` from the *larger* model's posterior and uses
the same Monte-Carlo likelihood: it cross-checks the nested-sampling
integration only.

Density at the null (weighted posterior samples, Kish ``n_eff``):

* interior null: Gaussian kernel density with Silverman's bandwidth
  ``0.9 min(sd, IQR/1.34) n_eff^{-1/5}``, reflected at the prior bounds;
* boundary null (``peak_fraction = 0``, ``chi_fraction in {0, 1}``): the
  one-sided density at the bound. A naive kernel estimate there is biased low
  by about a factor 2 (``+ln 2`` in favour of the extra component, Sec. 4.2).
  The primary estimator is the local-linear boundary kernel of Jones (1993,
  Stat. Comput. 3, 135): with ``u = |x - a| / h`` and
  ``a_l = int_0^inf u^l phi(u) du`` (``a_0 = a_2 = 1/2``, ``a_1 = phi(0)``),
  ``f(a) = sum_k W_k phi(u_k) (a_2 - a_1 u_k) / (h (a_0 a_2 - a_1^2))``, which
  has ``O(h^2)`` bias at the boundary. The reflection estimator
  ``2 sum_k W_k phi(u_k) / h`` (``O(h)`` bias proportional to ``f'(a)``) and
  the histogram ratios ``P(x in [a, a+eps]) / eps`` are reported as
  cross-checks.

Uncertainty: bootstrap over posterior draws (``n_eff`` draws with replacement
from the weighted sample per replicate, bandwidth re-selected). When fewer
than ``min_local_samples`` effective samples lie within one bandwidth of the
null the density is not estimable (the null is deep in the posterior tail,
where the evidence route is the only usable one); a one-sided bound from the
95% Poisson upper limit on the local count is reported instead.

Edge classification (:func:`classify_edge`) is derived from the grammar: a
registry of the embeddings of each mutation atom (null parameter/value,
parameter map, unidentified parameters), the parent's option state (an option
*replacement* is not nested) and a comparison of the shared priors (identical
priors give an exact SDDR, a support-compatible mismatch needs VW). Nesting is
verified numerically by :func:`verify_nesting` on the compiled models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import chi2

from gwpop_search.grammar import ModelSpec

from ._common import AnalysisInputError, WeightedPosterior, as_float, as_int, json_ready, weighted_quantile

SDDR_FORMAT = "gwpop-search-sddr-crosscheck-1.0"
EDGE_CLASSIFICATION_FORMAT = "gwpop-search-edge-nesting-1.0"
_PHI0 = 1.0 / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Prior densities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniformDifferencePrior:
    """Prior of ``d = x - y`` for independent ``x ~ U(a1, b1)``, ``y ~ U(a2, b2)``.

    The density is the overlap length ``|[a1, b1] cap [a2 + d, b2 + d]| / (w1 w2)``
    (a trapezoid) on ``[a1 - b2, b1 - a2]``. It is the prior of the tested
    coordinate of a difference null (``alpha_2 = alpha_1``, ``beta_high = beta_low``).
    """

    a1: float
    b1: float
    a2: float
    b2: float
    family: str = "uniform_difference"

    @property
    def parameters(self) -> dict[str, float]:
        return {"a1": self.a1, "b1": self.b1, "a2": self.a2, "b2": self.b2}


def _difference_density(params, x):
    x = np.asarray(x, dtype=np.float64)
    a1, b1, a2, b2 = params["a1"], params["b1"], params["a2"], params["b2"]
    overlap = np.minimum(b1, b2 + x) - np.maximum(a1, a2 + x)
    return np.maximum(overlap, 0.0) / ((b1 - a1) * (b2 - a2))


def prior_log_density(prior, value: float) -> float:
    """Log density of a ``PriorSpec`` or grammar ``PriorConfig`` at ``value``.

    Densities at a support bound are the one-sided limits (e.g. ``1/(b - a)``
    for a uniform prior at ``a``); outside the support ``-inf``.
    """
    family, params = _prior_parts(prior)
    x = float(value)
    if family == "uniform_difference":
        d = float(_difference_density(params, x))
        return math.log(d) if d > 0 else -math.inf
    if family == "uniform":
        lo, hi = params["low"], params["high"]
        return -math.log(hi - lo) if lo <= x <= hi else -math.inf
    if family == "log_uniform":
        lo, hi = params["low"], params["high"]
        return -math.log(x) - math.log(math.log(hi / lo)) if lo <= x <= hi else -math.inf
    if family == "normal":
        loc, scale = params["loc"], params["scale"]
        return -0.5 * ((x - loc) / scale) ** 2 - math.log(scale) - 0.5 * math.log(2 * math.pi)
    raise ValueError(f"unsupported prior family {family!r}")


def prior_log_density_array(prior, values) -> np.ndarray:
    family, params = _prior_parts(prior)
    x = np.asarray(values, dtype=np.float64)
    if family == "uniform_difference":
        d = _difference_density(params, x)
        with np.errstate(divide="ignore"):
            return np.where(d > 0, np.log(np.where(d > 0, d, 1.0)), -np.inf)
    if family == "uniform":
        lo, hi = params["low"], params["high"]
        return np.where((x >= lo) & (x <= hi), -math.log(hi - lo), -np.inf)
    if family == "log_uniform":
        lo, hi = params["low"], params["high"]
        inside = (x >= lo) & (x <= hi)
        safe = np.where(inside, x, 1.0)
        return np.where(inside, -np.log(safe) - math.log(math.log(hi / lo)), -np.inf)
    if family == "normal":
        loc, scale = params["loc"], params["scale"]
        return -0.5 * ((x - loc) / scale) ** 2 - math.log(scale) - 0.5 * math.log(2 * math.pi)
    raise ValueError(f"unsupported prior family {family!r}")


def prior_support(prior) -> tuple[float, float]:
    family, params = _prior_parts(prior)
    if family == "uniform_difference":
        return float(params["a1"] - params["b2"]), float(params["b1"] - params["a2"])
    if family in {"uniform", "log_uniform"}:
        return float(params["low"]), float(params["high"])
    return -math.inf, math.inf


def _prior_parts(prior) -> tuple[str, dict[str, float]]:
    family = str(prior.family)
    if hasattr(prior, "parameters"):
        params = {k: float(v) for k, v in dict(prior.parameters).items()}
    else:
        params = {
            k: float(getattr(prior, k))
            for k in ("low", "high", "loc", "scale")
            if getattr(prior, k, None) is not None
        }
    return family, params


def _prior_key(prior) -> tuple:
    family, params = _prior_parts(prior)
    return (family, tuple(sorted(params.items())))


# ---------------------------------------------------------------------------
# Density estimation at the null
# ---------------------------------------------------------------------------


def silverman_bandwidth(values, weights) -> float:
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w * w)
    mean = float(np.sum(w * values))
    sd = math.sqrt(float(np.sum(w * (values - mean) ** 2)))
    q25, q75 = weighted_quantile(values, w, [0.25, 0.75])
    spread = sd if not (q75 - q25) > 0 else min(sd, (q75 - q25) / 1.34)
    if not spread > 0:
        raise ValueError("posterior samples of the tested parameter have zero spread")
    return 0.9 * spread * n_eff ** (-0.2)


def kde_density(values, weights, x0: float, h: float, *, support=(-math.inf, math.inf)) -> float:
    """Gaussian KDE at an interior point, reflected at finite support bounds."""
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    lo, hi = support
    total = np.sum(w * np.exp(-0.5 * ((x0 - values) / h) ** 2))
    if math.isfinite(lo):
        total += np.sum(w * np.exp(-0.5 * ((x0 - (2 * lo - values)) / h) ** 2))
    if math.isfinite(hi):
        total += np.sum(w * np.exp(-0.5 * ((x0 - (2 * hi - values)) / h) ** 2))
    return float(total * _PHI0 / h)


def boundary_density(values, weights, bound: float, h: float, *, side: str, method: str) -> float:
    """One-sided density at a support bound (``side`` ``lower``/``upper``).

    ``method``: ``linear`` (Jones 1993 local-linear boundary kernel, primary),
    ``reflection`` or ``naive`` (the biased plain kernel, for comparison).
    """
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if side == "lower":
        u = (values - bound) / h
    elif side == "upper":
        u = (bound - values) / h
    else:
        raise ValueError("side must be 'lower' or 'upper'")
    if np.any(u < -1e-12):
        raise ValueError("samples lie outside the bounded support")
    k = _PHI0 * np.exp(-0.5 * u * u)
    if method == "naive":
        return float(np.sum(w * k) / h)
    if method == "reflection":
        return float(2.0 * np.sum(w * k) / h)
    if method == "linear":
        a0 = a2 = 0.5
        a1 = _PHI0
        return float(np.sum(w * k * (a2 - a1 * u)) / (h * (a0 * a2 - a1 * a1)))
    raise ValueError(f"unknown boundary density method {method!r}")


def histogram_boundary_density(values, weights, bound: float, eps: float, *, side: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if side == "lower":
        inside = (values >= bound) & (values <= bound + eps)
    else:
        inside = (values <= bound) & (values >= bound - eps)
    return float(np.sum(w[inside]) / eps)


@dataclass(frozen=True)
class DensityAtNull:
    """``ln p(omega_0 | D)`` with its bootstrap uncertainty."""

    parameter: str
    null_value: float
    location: str  # interior | lower | upper
    method: str
    log_density: float | None
    sigma: float | None
    bandwidth: float
    n_eff: float
    local_effective_samples: float
    estimable: bool
    log_density_upper_limit: float
    alternatives: Mapping[str, float] = field(default_factory=dict)
    n_bootstrap: int = 0

    def to_dict(self) -> dict[str, object]:
        return json_ready(
            {
                "parameter": self.parameter,
                "null_value": self.null_value,
                "location": self.location,
                "method": self.method,
                "log_density": self.log_density,
                "sigma": self.sigma,
                "bandwidth": self.bandwidth,
                "n_eff": self.n_eff,
                "local_effective_samples": self.local_effective_samples,
                "estimable": self.estimable,
                "log_density_upper_limit_95": self.log_density_upper_limit,
                "alternatives": dict(self.alternatives),
                "n_bootstrap": self.n_bootstrap,
            }
        )


def _density(values, weights, x0, location, method, support, h):
    if location == "interior":
        return kde_density(values, weights, x0, h, support=support)
    return boundary_density(values, weights, x0, h, side=location, method=method)


def density_at_null(
    values,
    weights,
    null_value: float,
    *,
    parameter: str = "omega",
    location: str = "interior",
    support=(-math.inf, math.inf),
    method: str | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    min_local_samples: float = 20.0,
    bandwidth: float | None = None,
    histogram_eps: Sequence[float] = (),
) -> DensityAtNull:
    """Posterior density of one parameter at its null value (see module doc)."""
    values = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or w.shape != values.shape or values.size < 5:
        raise ValueError("need at least 5 weighted samples of one parameter")
    w = w / w.sum()
    x0 = float(null_value)
    if location not in {"interior", "lower", "upper"}:
        raise ValueError("location must be interior, lower or upper")
    if method is None:
        method = "kde_reflected" if location == "interior" else "linear"
    if (location == "interior") != (method == "kde_reflected"):
        raise ValueError("interior nulls use 'kde_reflected'; boundary nulls 'linear'/'reflection'/'naive'")
    n_eff = 1.0 / float(np.sum(w * w))
    h = silverman_bandwidth(values, w) if bandwidth is None else as_float("bandwidth", bandwidth, positive=True)
    # effective samples within one bandwidth of the null (a window of width 2h in the
    # interior, h at a bound) and the 95% Poisson upper limit on the density there
    local = n_eff * float(np.sum(w[np.abs(values - x0) <= h]))
    window = 2.0 * h if location == "interior" else h
    upper = chi2.ppf(0.95, 2.0 * (local + 1.0)) / 2.0 / (n_eff * window)
    alternatives: dict[str, float] = {}
    if location != "interior":
        for alt in ("linear", "reflection", "naive"):
            value = boundary_density(values, w, x0, h, side=location, method=alt)
            alternatives[f"log_density_{alt}"] = math.log(value) if value > 0 else -math.inf
        for eps in histogram_eps:
            value = histogram_boundary_density(values, w, x0, float(eps), side=location)
            alternatives[f"log_density_histogram_eps_{eps:g}"] = math.log(value) if value > 0 else -math.inf
    for scale in (0.5, 2.0):
        value = _density(values, w, x0, location, method, support, h * scale)
        alternatives[f"log_density_bandwidth_x{scale:g}"] = math.log(value) if value > 0 else -math.inf
    estimate = _density(values, w, x0, location, method, support, h)
    estimable = bool(local >= min_local_samples and estimate > 0)
    sigma = None
    log_density = math.log(estimate) if estimate > 0 else None
    n_boot = as_int("n_bootstrap", n_bootstrap, minimum=0)
    if estimable and n_boot > 1:
        rng = np.random.default_rng(as_int("seed", seed, minimum=0))
        m = max(5, int(round(n_eff)))
        logs = []
        for _ in range(n_boot):
            idx = rng.choice(values.size, size=m, replace=True, p=w)
            sample = values[idx]
            ones = np.full(m, 1.0 / m)
            try:
                hb = silverman_bandwidth(sample, ones) if bandwidth is None else h
            except ValueError:
                continue
            value = _density(sample, ones, x0, location, method, support, hb)
            if value > 0:
                logs.append(math.log(value))
        if len(logs) >= 2:
            sigma = float(np.std(logs, ddof=1))
        if len(logs) < 0.9 * n_boot:
            estimable = False  # the density is too often non-positive in resamples
    return DensityAtNull(
        parameter=str(parameter),
        null_value=x0,
        location=location,
        method=method,
        log_density=log_density,
        sigma=sigma,
        bandwidth=float(h),
        n_eff=float(n_eff),
        local_effective_samples=float(local),
        estimable=estimable,
        log_density_upper_limit=float(math.log(upper)),
        alternatives=alternatives,
        n_bootstrap=n_boot,
    )


# ---------------------------------------------------------------------------
# Edge classification from the grammar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullEmbedding:
    """``M_L`` equals ``M_S`` at ``tested_parameter = null_value``.

    ``parameter_map`` maps smaller-model parameters to their larger-model
    counterparts (identity for names not listed); ``unidentified`` are
    larger-model parameters that drop out at the null.

    ``difference_of = (a, b)`` makes the tested coordinate the derived
    difference ``a - b`` of two larger-model parameters (``tested_parameter``
    is then only its label). With independent identical uniform priors on
    ``a`` and ``b`` the null ``a - b = 0`` is exactly nested: in the
    coordinates ``(b, d = a - b)`` the conditional prior of ``b`` at ``d = 0`` is
    its own uniform prior (the smaller model's prior of the parameter mapped
    to ``b``), and the prior density of ``d`` at 0 is ``1 / (high - low)``.
    """

    tested_parameter: str
    null_value: float
    location: str  # interior | lower | upper
    parameter_map: Mapping[str, str] = field(default_factory=dict)
    unidentified: tuple[str, ...] = ()
    difference_of: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "tested_parameter": self.tested_parameter,
            "null_value": self.null_value,
            "location": self.location,
            "parameter_map": dict(self.parameter_map),
            "unidentified": list(self.unidentified),
        }
        if self.difference_of is not None:
            payload["difference_of"] = list(self.difference_of)
        return payload

    def tested_values(self, sample) -> np.ndarray:
        if self.difference_of is None:
            return sample.column(self.tested_parameter)
        a, b = self.difference_of
        return sample.column(a) - sample.column(b)

    def tested_prior(self, larger_priors: Mapping[str, object]):
        if self.difference_of is None:
            return larger_priors[self.tested_parameter]
        a, b = self.difference_of
        fa, pa = _prior_parts(larger_priors[a])
        fb, pb = _prior_parts(larger_priors[b])
        if fa != "uniform" or fb != "uniform":
            raise AnalysisInputError("difference nulls need uniform priors on both parameters")
        return UniformDifferencePrior(pa["low"], pa["high"], pb["low"], pb["high"])

    @property
    def tested_names(self) -> tuple[str, ...]:
        return tuple(self.difference_of) if self.difference_of is not None else (self.tested_parameter,)


@dataclass(frozen=True)
class EdgeNesting:
    parent_hash: str
    child_hash: str
    mutation_id: str
    classification: str  # exact | approximate | evidence_only
    larger_model: str | None  # child | parent
    embeddings: tuple[NullEmbedding, ...] = ()
    symmetric_embeddings: bool = False
    vw_required: bool = False
    vw_first_form_valid: bool = False
    prior_mismatches: tuple[str, ...] = ()
    reason: str = ""

    @property
    def sddr_eligible(self) -> bool:
        return self.classification in {"exact", "approximate"} and bool(self.embeddings)

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_hash": self.parent_hash,
            "child_hash": self.child_hash,
            "mutation_id": self.mutation_id,
            "classification": self.classification,
            "larger_model": self.larger_model,
            "embeddings": [item.to_dict() for item in self.embeddings],
            "symmetric_embeddings": self.symmetric_embeddings,
            "vw_required": self.vw_required,
            "vw_first_form_valid": self.vw_first_form_valid,
            "prior_mismatches": list(self.prior_mismatches),
            "reason": self.reason,
        }


# (block, option, value) -> (tested parameter, unidentified parameters). The
# declarative compiler adds ``slope * (x - pivot)`` or
# ``delta / (1 + exp(-(m1 - transition) / width))``, so the tested parameter at
# 0 reproduces the ``constant`` dependence exactly.
_OPTION_NULLS: dict[tuple[str, str, str], tuple[str, tuple[str, ...]]] = {
    ("pairing", "beta_dependence", "linear_m1"): ("beta_q_m1_slope", ()),
    ("pairing", "beta_dependence", "logistic_m1"): (
        "delta_beta_q",
        ("beta_q_m1_transition", "beta_q_m1_width"),
    ),
    ("chieff", "mean_dependence", "linear_m1"): ("chi_mu_m1_slope", ()),
    ("chieff", "mean_dependence", "linear_q"): ("chi_mu_q_slope", ()),
    ("chieff", "mean_dependence", "linear_z"): ("chi_mu_z_slope", ()),
    ("chieff", "width_dependence", "linear_m1"): ("log_chi_sigma_m1_slope", ()),
    ("chieff", "width_dependence", "linear_q"): ("log_chi_sigma_q_slope", ()),
    ("chieff", "width_dependence", "linear_z"): ("log_chi_sigma_z_slope", ()),
}
_NULL_OPTION_VALUE = "constant"

# v2 (GWTC-5 atom search) option nulls: the LVK-linear chi_eff correlations
# (slope 0 = constant), kappa(m1) (slope 0), and the logistic pairing step, whose
# null beta_high = beta_low is a difference null (the step location and width
# are unidentified there; beta_low takes the root beta).
_OPTION_NULLS.update({
    ("chieff", "mean_q", "linear"): ("chi_mu_q_slope", ()),
    ("chieff", "mean_z", "linear"): ("chi_mu_z_slope", ()),
    ("chieff", "mean_log_m1", "linear"): ("chi_mu_log_m1_slope", ()),
    ("chieff", "log_sigma_q", "linear"): ("chi_log_sigma_q_slope", ()),
    ("chieff", "log_sigma_z", "linear"): ("chi_log_sigma_z_slope", ()),
    ("chieff", "log_sigma_log_m1", "linear"): ("chi_log_sigma_log_m1_slope", ()),
    ("redshift", "kappa_dependence", "linear_log_m1"): ("kappa_log_m1_slope", ()),
})
_OPTION_EMBEDDINGS: dict[tuple[str, str, str], tuple[NullEmbedding, ...]] = {
    ("pairing", "beta_dependence", "logistic_log_m1"): (
        NullEmbedding(
            "beta_high-beta_low", 0.0, "interior", {"beta": "beta_low"},
            ("beta_m_t", "beta_width"), difference_of=("beta_high", "beta_low"),
        ),
    ),
}
#: v2 edges that are deliberately evidence-only, with the reason.
_NOT_NESTED_REASONS: dict[tuple[str, str, str], str] = {
    ("pairing", "beta_dependence", "per_mass_component"):
        "beta per mass component: the null is a two-dimensional equality (beta_pl = beta_p10 = beta_p35)",
}

# (block, smaller family, larger family) -> embeddings of the smaller family
# in the larger one; ``exact`` marks an embedding that reproduces the smaller
# density exactly in the compiled components.
_FAMILY_NULLS: dict[tuple[str, str, str], dict[str, object]] = {
    ("mass", "powerlaw", "pl_peak"): {
        "exact": True,
        "symmetric": False,
        "embeddings": (
            NullEmbedding("peak_fraction", 0.0, "lower", {}, ("peak_mu", "peak_sigma")),
        ),
    },
    ("mass", "pl_peak", "pl_two_peak"): {
        # the compiled two-peak mixture clips fractions at 1e-12 (not exact)
        "exact": False,
        "symmetric": False,
        "embeddings": (
            NullEmbedding(
                "peak2_fraction",
                0.0,
                "lower",
                {"peak_fraction": "peak1_fraction", "peak_mu": "peak1_mu", "peak_sigma": "peak1_sigma"},
                ("peak2_mu", "peak2_sigma"),
            ),
            NullEmbedding(
                "peak1_fraction",
                0.0,
                "lower",
                {"peak_fraction": "peak2_fraction", "peak_mu": "peak2_mu", "peak_sigma": "peak2_sigma"},
                ("peak1_mu", "peak1_sigma"),
            ),
        ),
    },
    ("chieff", "truncated_gaussian", "gaussian_mixture"): {
        "exact": True,
        "symmetric": True,  # exchangeable components with identical priors
        "embeddings": (
            NullEmbedding(
                "chi_fraction", 0.0, "lower",
                {"chi_mu": "chi_mu_1", "chi_sigma": "chi_sigma_1"}, ("chi_mu_2", "chi_sigma_2"),
            ),
            NullEmbedding(
                "chi_fraction", 1.0, "upper",
                {"chi_mu": "chi_mu_2", "chi_sigma": "chi_sigma_2"}, ("chi_mu_1", "chi_sigma_1"),
            ),
        ),
        # a chi_eff mean/width slope in the smaller model has no counterpart
        "requires_constant_options": ("mean_dependence", "width_dependence"),
    },
}


_V2_CHIEFF_CORRELATION_OPTIONS = (
    "mean_q", "mean_z", "mean_log_m1", "log_sigma_q", "log_sigma_z", "log_sigma_log_m1",
    "q_pivot", "z_pivot", "m1_pivot",
)
_FAMILY_NULLS.update({
    # Dirichlet unit coordinates: dropping component c is lam_u_c = 0 with every
    # other prior unchanged (grammar.v2_structure), so these are exact.
    ("mass", "bp1p_low", "bp2p"): {
        "exact": True, "symmetric": False,
        "embeddings": (NullEmbedding("lam_u_p35", 0.0, "lower", {}, ("mu_p35", "sigma_p35")),),
    },
    ("mass", "bp1p_high", "bp2p"): {
        "exact": True, "symmetric": False,
        "embeddings": (NullEmbedding("lam_u_p10", 0.0, "lower", {}, ("mu_p10", "sigma_p10")),),
    },
    ("mass", "bp2p", "bp3p"): {
        "exact": True, "symmetric": False,
        "embeddings": (NullEmbedding("lam_u_p3", 0.0, "lower", {}, ("mu_p3", "sigma_p3")),),
    },
    # no break: alpha_2 = alpha_1 (m_break unidentified), a difference null.
    ("mass", "pl2p", "bp2p"): {
        "exact": True, "symmetric": False,
        "embeddings": (
            NullEmbedding("alpha_2-alpha_1", 0.0, "interior", {"alpha": "alpha_1"}, ("m_break",),
                          difference_of=("alpha_2", "alpha_1")),
        ),
    },
    # constant-fraction mixture: component 1 is the Gaussian (same names, same
    # correlations); the fraction of the ordered component 2 at 0.
    ("chieff", "linear_gaussian", "linear_gaussian_mixture"): {
        "exact": True, "symmetric": False,
        "embeddings": (
            NullEmbedding("chi_fraction", 0.0, "lower", {}, ("chi_mu_2_frac", "chi_log_sigma_2")),
        ),
        "requires_larger_options": {"fraction_dependence": "constant"},
        "requires_equal_options": _V2_CHIEFF_CORRELATION_OPTIONS,
    },
    ("chieff", "linear_gaussian", "linear_skew_normal"): {
        "exact": True, "symmetric": False,
        "embeddings": (NullEmbedding("chi_eps", 0.0, "interior", {}, ()),),
        "requires_equal_options": _V2_CHIEFF_CORRELATION_OPTIONS,
    },
    # Madau-Dickinson with kappa_MD = 0 is (1+z)^gamma exactly (z_peak unidentified).
    ("redshift", "powerlaw_1pz", "madau_dickinson_psi"): {
        "exact": True, "symmetric": False,
        "embeddings": (NullEmbedding("md_kappa", 0.0, "lower", {"kappa": "md_gamma"}, ("md_z_peak",)),),
        "requires_constant_options": ("kappa_dependence",),
    },
})


_V2_FAMILY_REASONS = {
    ("chieff", "linear_gaussian", "linear_student_t"):
        "Student-t: the Gaussian is the nu -> infinity limit, outside the nu prior",
    ("chieff", "linear_gaussian", "linear_gaussian_mixture"):
        "mass-dependent mixture fraction: the null chi_fraction_low = chi_fraction_high = 0 "
        "is two-dimensional",
}


def _family_rule(smaller: ModelSpec, larger: ModelSpec, block: str):
    key = (block, getattr(smaller, block).family, getattr(larger, block).family)
    rule = _FAMILY_NULLS.get(key)
    if rule is None:
        return None
    larger_options = getattr(larger, block).options
    for option, value in dict(rule.get("requires_larger_options", {})).items():
        if larger_options.get(option) != value:
            return None
    return rule


def _support_contains(outer, inner) -> bool:
    lo_o, hi_o = prior_support(outer)
    lo_i, hi_i = prior_support(inner)
    if _prior_parts(outer)[0] == "normal" or _prior_parts(inner)[0] == "normal":
        return _prior_parts(outer)[0] == "normal"
    return lo_o <= lo_i and hi_i <= hi_o


def classify_edge(parent: ModelSpec, child: ModelSpec, mutation_id: str) -> EdgeNesting:
    """Nesting class of a graph edge (exact SDDR / approximate / evidence-only)."""
    from gwpop_search.grammar import structural_diff_axes

    axes = structural_diff_axes(parent, child)
    base = {"parent_hash": parent.model_hash, "child_hash": child.model_hash, "mutation_id": mutation_id}
    if len(axes) != 1:
        return EdgeNesting(**base, classification="evidence_only", larger_model=None,
                           reason=f"edge changes {len(axes)} structural axes {list(axes)}")
    axis = axes[0]
    block = axis.split(".")[0]
    pb, cb = getattr(parent, block), getattr(child, block)
    embeddings: tuple[NullEmbedding, ...] = ()
    symmetric = False
    exact_embedding = True
    larger = None
    if ".options." in axis:
        option = axis.split(".options.")[1]
        key = (block, option, str(cb.options.get(option)))
        rule = _OPTION_NULLS.get(key)
        explicit = _OPTION_EMBEDDINGS.get(key)
        if key in _NOT_NESTED_REASONS:
            return EdgeNesting(**base, classification="evidence_only", larger_model=None,
                               reason=_NOT_NESTED_REASONS[key])
        if rule is None and explicit is None:
            return EdgeNesting(**base, classification="evidence_only", larger_model=None,
                               reason=f"no nesting embedding registered for {axis}={cb.options.get(option)!r}")
        if pb.options.get(option) != _NULL_OPTION_VALUE:
            return EdgeNesting(
                **base, classification="evidence_only", larger_model=None,
                reason=(f"option replacement {pb.options.get(option)!r} -> {cb.options.get(option)!r}: "
                        "non-nested siblings sharing the constant sub-model"),
            )
        if explicit is not None:
            embeddings = tuple(explicit)
        else:
            tested, unidentified = rule
            embeddings = (NullEmbedding(tested, 0.0, "interior", {}, tuple(unidentified)),)
        larger, smaller_spec, larger_spec = "child", parent, child
    else:
        rule = _family_rule(child, parent, block)
        if rule is not None:
            larger, smaller_spec, larger_spec = "parent", child, parent
        else:
            rule = _family_rule(parent, child, block)
            if rule is None:
                reason = _V2_FAMILY_REASONS.get(
                    (block, pb.family, cb.family),
                    _V2_FAMILY_REASONS.get((block, cb.family, pb.family)),
                ) or f"family change {pb.family} -> {cb.family} is not nested within the priors"
                return EdgeNesting(**base, classification="evidence_only", larger_model=None, reason=reason)
            larger, smaller_spec, larger_spec = "child", parent, child
        for option in rule.get("requires_constant_options", ()):
            if getattr(smaller_spec, block).options.get(option, _NULL_OPTION_VALUE) != _NULL_OPTION_VALUE:
                return EdgeNesting(
                    **base, classification="evidence_only", larger_model=None,
                    reason=f"the smaller model has {block}.{option}="
                    f"{getattr(smaller_spec, block).options.get(option)!r}, which the larger family lacks",
                )
        for option in rule.get("requires_equal_options", ()):
            if getattr(smaller_spec, block).options.get(option) != getattr(larger_spec, block).options.get(option):
                return EdgeNesting(
                    **base, classification="evidence_only", larger_model=None,
                    reason=f"{block}.{option} differs between the two families' blocks",
                )
        embeddings = tuple(rule["embeddings"])
        symmetric = bool(rule["symmetric"])
        exact_embedding = bool(rule["exact"])

    # prior comparison for every embedding
    mismatches: list[str] = []
    first_form_valid = True
    for emb in embeddings:
        mapped = dict(emb.parameter_map)
        for name, prior in smaller_spec.priors.items():
            target = mapped.get(name, name)
            if target not in larger_spec.priors:
                raise AnalysisInputError(
                    f"{mutation_id}: smaller-model parameter {name!r} maps to {target!r}, "
                    "which the larger model lacks"
                )
            other = larger_spec.priors[target]
            if _prior_key(prior) != _prior_key(other):
                mismatches.append(f"{name}->{target}")
                if not _support_contains(other, prior):
                    first_form_valid = False
        if emb.difference_of is not None:
            a, b = emb.difference_of
            if _prior_key(larger_spec.priors[a]) != _prior_key(larger_spec.priors[b]) or \
                    _prior_parts(larger_spec.priors[a])[0] != "uniform":
                return EdgeNesting(
                    **base, classification="evidence_only", larger_model=None,
                    reason=f"difference null {a} - {b} needs identical uniform priors on both",
                )
        expected = set(smaller_spec.priors) - set(mapped) | set(mapped.values())
        extra = set(larger_spec.priors) - expected - set(emb.tested_names) - set(emb.unidentified)
        if extra:
            raise AnalysisInputError(
                f"{mutation_id}: larger-model parameters {sorted(extra)} are neither shared, tested "
                "nor declared unidentified"
            )
    mismatches = sorted(set(mismatches))
    if not exact_embedding:
        classification, reason = "approximate", "embedding not exact in the compiled model"
    elif mismatches and first_form_valid:
        classification, reason = "approximate", "shared priors differ: SDDR x Verdinelli-Wasserman factor"
    elif mismatches:
        classification, reason = "approximate", "shared priors differ and supports only partially overlap"
    else:
        classification = "exact"
        reason = "nested with identical shared priors (VW = 1)"
    if classification == "approximate" and not exact_embedding and mismatches and not first_form_valid:
        reason = "embedding not exact and shared-prior supports only partially overlap"
    return EdgeNesting(
        **base,
        classification=classification,
        larger_model=larger,
        embeddings=embeddings,
        symmetric_embeddings=symmetric,
        vw_required=bool(mismatches),
        vw_first_form_valid=bool(mismatches) and first_form_valid,
        prior_mismatches=tuple(mismatches),
        reason=reason,
    )


def classify_graph_edges(graph) -> list[EdgeNesting]:
    by_hash = graph.by_hash
    return [
        classify_edge(by_hash[edge.parent_hash], by_hash[edge.child_hash], edge.mutation_id)
        for edge in graph.edges
    ]


def edge_classification_report(graph) -> dict[str, object]:
    rows = classify_graph_edges(graph)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    return {
        "format_version": EDGE_CLASSIFICATION_FORMAT,
        "graph_root_hash": graph.root_hash,
        "n_edges": len(rows),
        "counts": counts,
        "edges": [row.to_dict() for row in rows],
    }


def _synthetic_gwcat_samples(rng, n: int) -> dict[str, np.ndarray]:
    return {
        "m1_detector": rng.uniform(4.0, 250.0, n),
        "q": rng.uniform(0.06, 1.0, n),
        "luminosity_distance": rng.uniform(50.0, 12000.0, n),
        "ra": rng.uniform(0.0, 2.0 * math.pi, n),
        "dec": rng.uniform(-0.5 * math.pi, 0.5 * math.pi, n),
        "chi_eff": rng.uniform(-0.95, 0.95, n),
    }


def _draw_prior(rng, prior) -> float:
    family, params = _prior_parts(prior)
    if family == "uniform":
        return float(rng.uniform(params["low"], params["high"]))
    if family == "log_uniform":
        return float(math.exp(rng.uniform(math.log(params["low"]), math.log(params["high"]))))
    return float(rng.normal(params["loc"], params["scale"]))


def verify_nesting(
    nesting: EdgeNesting,
    parent: ModelSpec,
    child: ModelSpec,
    *,
    n_checks: int = 8,
    n_samples: int = 512,
    seed: int = 0,
    rtol: float = 1e-10,
) -> dict[str, object]:
    """Numerically check that the larger model reproduces the smaller at the null.

    Draws smaller-model hyperparameters from their priors, maps them to the
    larger model at the null (random values for unidentified parameters) and
    compares both compiled densities on synthetic gwcat-basis samples.
    """
    from gwpop_search.models import compile_model_spec

    if not nesting.embeddings:
        raise ValueError("edge has no nesting embedding")
    larger_spec = child if nesting.larger_model == "child" else parent
    smaller_spec = parent if nesting.larger_model == "child" else child
    larger_model, smaller_model = compile_model_spec(larger_spec), compile_model_spec(smaller_spec)
    rng = np.random.default_rng(as_int("seed", seed, minimum=0))
    samples = _synthetic_gwcat_samples(rng, n_samples)
    rows = []
    for emb in nesting.embeddings:
        max_diff, pattern_mismatch = 0.0, 0
        for _ in range(as_int("n_checks", n_checks, minimum=1)):
            hp_s = {name: _draw_prior(rng, prior) for name, prior in smaller_spec.priors.items()}
            hp_l = {}
            for name, value in hp_s.items():
                hp_l[emb.parameter_map.get(name, name)] = value
            if emb.difference_of is None:
                hp_l[emb.tested_parameter] = emb.null_value
            else:
                a, b = emb.difference_of
                hp_l[a] = hp_l[b] + emb.null_value
            for name in emb.unidentified:
                hp_l[name] = _draw_prior(rng, larger_spec.priors[name])
            a = np.asarray(smaller_model(samples, hp_s), dtype=np.float64)
            b = np.asarray(larger_model(samples, hp_l), dtype=np.float64)
            pattern_mismatch += int(np.sum(np.isfinite(a) != np.isfinite(b)))
            both = np.isfinite(a) & np.isfinite(b)
            if both.any():
                diff = np.abs(a[both] - b[both]) / np.maximum(1.0, np.abs(a[both]))
                max_diff = max(max_diff, float(np.max(diff)))
        rows.append(
            {
                "embedding": emb.to_dict(),
                "max_relative_difference": max_diff,
                "support_pattern_mismatches": pattern_mismatch,
                "nested": bool(pattern_mismatch == 0 and max_diff <= rtol),
            }
        )
    return {"edge": nesting.mutation_id, "embeddings": rows, "all_nested": all(r["nested"] for r in rows)}


# ---------------------------------------------------------------------------
# SDDR and VW for one edge
# ---------------------------------------------------------------------------


def vw_factor(
    sample: WeightedPosterior,
    embedding: NullEmbedding,
    smaller_priors: Mapping[str, object],
    larger_priors: Mapping[str, object],
    *,
    eps: float,
) -> dict[str, object]:
    """``VW = E[pi_S(psi) / pi_L(psi | omega_0)]`` over larger-model samples near the null.

    The conditional posterior at the null is approximated by the samples with
    ``|omega - omega_0| <= eps`` (``eps`` in the units of the tested
    parameter; approximate). Requires ``supp pi_S ⊆ supp pi_L`` (first VW form).
    """
    omega = embedding.tested_values(sample)
    near = np.abs(omega - embedding.null_value) <= eps
    w = sample.weights[near]
    if w.size == 0:
        return {"eps": eps, "n_near": 0, "n_eff_near": 0.0, "log_vw": None}
    log_ratio = np.zeros(w.size)
    for name, prior in smaller_priors.items():
        target = embedding.parameter_map.get(name, name)
        values = sample.column(target)[near]
        log_large = prior_log_density_array(larger_priors[target], values)
        if not np.all(np.isfinite(log_large)):
            raise AnalysisInputError(
                f"posterior samples of {target!r} lie outside the larger model's prior support"
            )
        log_ratio += prior_log_density_array(prior, values) - log_large
    wn = w / w.sum()
    with np.errstate(over="ignore"):
        ratio = np.exp(log_ratio)
    vw = float(np.sum(wn * ratio))
    return {
        "eps": float(eps),
        "n_near": int(w.size),
        "n_eff_near": float(1.0 / np.sum(wn * wn)),
        "log_vw": math.log(vw) if vw > 0 else -math.inf,
        "fraction_inside_smaller_support": float(np.sum(wn[np.isfinite(log_ratio)])),
    }


#: Method-level systematic (nats) that MAY be added to the SDDR/NS agreement
#: tolerance, by the geometry of the null. **Opt-in, never a default.**
#:
#: The SDDR and the nested-sampling evidence are two different estimators of the
#: same ``ln BF``, so their difference plausibly has a floor that neither error
#: bar describes. These numbers are the measured maxima, rounded up, of
#: ``ln BF_NS - ln BF_SDDR`` over the dynesty toys of
#: validation/analysis_estimators/sddr_validation.json (static rslice,
#: **nlive 500**, slices = 2 (3 + ndim), 2 repeats):
#:
#:   interior null (Gaussian mean)      n=9  +0.035 +- 0.014, max |diff| 0.087
#:   boundary null (mixture fraction)   n=6  -0.160 +- 0.021, max |diff| 0.243
#:   boundary null + VW (shared prior)  n=3  -0.285 +- 0.061, max |diff| 0.377
#:
#: They are therefore calibrated on the very disagreements the agreement test
#: exists to detect, on n = 9/6/3 cases, and at nlive 500 -- below the F3
#: production setting (1000) and well below F4 (2000). The stated hypothesis for
#: the residual (nested-sampling ln Z bias on the peak-mixture geometry, whose
#: peak location is unidentified as the mixture fraction goes to zero; the
#: boundary-density estimator is accurate to < 0.02 nats on such pile-ups, with
#: the opposite sign, per boundary_density_validation.json) predicts the offset
#: shrinks at production nlive, so the floors are wider than production warrants.
#:
#: A tolerance fitted post hoc to the failures it must catch is not a validated
#: systematic, so :func:`sddr_edge_check` uses the bare formula (8) unless a
#: caller asks for these values by name (``method_systematic="measured"``), and
#: the 0.40-nat VW band is in any case too wide to confirm an edge against the
#: 3-nat claim threshold. Until they are re-measured on an independent set of
#: cases at F3/F4 settings, the ``mass.family.powerlaw``, ``mass.family.pl_two_peak``
#: and ``chieff.family.gaussian_mixture`` cross-checks are evidence-only support,
#: not confirmation. The signed ``difference`` is always reported either way.
SDDR_METHOD_SYSTEMATIC: dict[str, float] = {
    "interior": 0.10,
    "boundary": 0.25,
    "boundary_vw": 0.40,
}


def method_systematic_for(nesting: "EdgeNesting") -> float:
    """The measured (post-hoc, nlive 500) method systematic for this null geometry.

    See :data:`SDDR_METHOD_SYSTEMATIC` for why this is opt-in.
    """
    if not nesting.embeddings:
        return 0.0
    at_bound = any(emb.location != "interior" for emb in nesting.embeddings)
    if not at_bound:
        return SDDR_METHOD_SYSTEMATIC["interior"]
    return SDDR_METHOD_SYSTEMATIC["boundary_vw" if nesting.vw_required else "boundary"]


def _resolve_method_systematic(value, nesting: "EdgeNesting") -> tuple[float, str]:
    """``(nats, source)`` for ``method_systematic``; ``None``/``0`` = formula (8)."""
    if value is None:
        return 0.0, "formula8"
    if isinstance(value, str):
        if value != "measured":
            raise ValueError(
                f"method_systematic must be a number or 'measured'; got {value!r}"
            )
        return float(method_systematic_for(nesting)), "measured"
    return as_float("method_systematic", value, nonnegative=True), "supplied"


@dataclass(frozen=True)
class SDDRCheck:
    mutation_id: str
    classification: str
    status: str  # agree | disagree | bound_consistent | bound_violated | not_estimable
    log_bf_child_over_parent_sddr: float | None
    sigma_sddr: float | None
    log_bf_child_over_parent_ns: float | None
    sigma_ns: float | None
    tolerance: float | None
    details: Mapping[str, object]
    method_systematic: float = 0.0
    method_systematic_source: str = "formula8"  # formula8 | measured | supplied

    @property
    def difference(self) -> float | None:
        """``ln BF_NS - ln BF_SDDR`` (nats): the quantity the tolerance bounds."""
        if self.log_bf_child_over_parent_ns is None or self.log_bf_child_over_parent_sddr is None:
            return None
        return self.log_bf_child_over_parent_ns - self.log_bf_child_over_parent_sddr

    def to_dict(self) -> dict[str, object]:
        return json_ready(
            {
                "mutation_id": self.mutation_id,
                "classification": self.classification,
                "status": self.status,
                "log_bf_child_over_parent_sddr": self.log_bf_child_over_parent_sddr,
                "sigma_sddr": self.sigma_sddr,
                "log_bf_child_over_parent_ns": self.log_bf_child_over_parent_ns,
                "sigma_ns": self.sigma_ns,
                "difference": self.difference,
                "method_systematic": self.method_systematic,
                "method_systematic_source": self.method_systematic_source,
                "tolerance": self.tolerance,
                "details": dict(self.details),
            }
        )


def sddr_edge_check(
    nesting: EdgeNesting,
    larger_sample: WeightedPosterior,
    larger_priors: Mapping[str, object],
    smaller_priors: Mapping[str, object],
    *,
    log_bf_ns: float | None = None,
    sigma_ns: float | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    min_local_samples: float = 20.0,
    vw_eps: Sequence[float] = (0.02, 0.05, 0.1),
    vw_primary_eps: float = 0.05,
    min_vw_near_ess: float = 20.0,
    agreement_sigmas: float = 2.0,
    method_systematic: float | str | None = None,
) -> SDDRCheck:
    """SDDR (x VW) estimate of ``ln BF_{child/parent}`` and its agreement with NS.

    ``sigma_ns`` is the nested-sampling error of ``ln BF_NS``
    (``sqrt(sigma_NS,a^2/R_a + sigma_NS,b^2/R_b)``); agreement requires::

        |ln BF_NS - ln BF_SDDR| <= agreement_sigmas * sqrt(sigma_ns^2 + sigma_SDDR^2)
                                   + method_systematic

    which is formula (8) of the note, optionally widened by a floor for the
    difference between the two estimators themselves. ``method_systematic``
    defaults to the bare formula (8) (``None`` or ``0.0``); pass a number for an
    externally justified floor, or the string ``"measured"`` for the per-geometry
    values of :data:`SDDR_METHOD_SYSTEMATIC` -- which were fitted post hoc to the
    disagreements this test exists to detect, on n = 9/6/3 toy cases at nlive
    500, and are NOT a validated systematic. The value used and where it came
    from are reported as ``method_systematic``/``method_systematic_source``, and
    the signed ``difference`` is always reported, so widening the band never
    hides a disagreement.
    """
    method_systematic, ms_source = _resolve_method_systematic(method_systematic, nesting)
    if not nesting.sddr_eligible:
        return SDDRCheck(nesting.mutation_id, nesting.classification, "not_applicable", None, None,
                         log_bf_ns, sigma_ns, None, {"reason": nesting.reason}, 0.0, "formula8")
    sign = 1.0 if nesting.larger_model == "child" else -1.0
    embeddings = nesting.embeddings if nesting.symmetric_embeddings else nesting.embeddings[:1]
    densities = []
    for k, emb in enumerate(embeddings):
        prior = emb.tested_prior(larger_priors)
        densities.append(
            density_at_null(
                emb.tested_values(larger_sample),
                larger_sample.weights,
                emb.null_value,
                parameter=emb.tested_parameter,
                location=emb.location,
                support=prior_support(prior),
                n_bootstrap=n_bootstrap,
                seed=seed + k,
                min_local_samples=min_local_samples / len(embeddings),
                histogram_eps=(0.005, 0.01, 0.02) if emb.location != "interior" else (),
            )
        )
    emb0 = embeddings[0]
    log_prior = prior_log_density(emb0.tested_prior(larger_priors), emb0.null_value)
    details: dict[str, object] = {
        "larger_model": nesting.larger_model,
        "densities": [d.to_dict() for d in densities],
        "log_prior_at_null": log_prior,
        "symmetrized": bool(nesting.symmetric_embeddings),
    }
    # symmetrized density: average of the embeddings' densities (exact symmetry)
    estimable = all(d.estimable and d.log_density is not None for d in densities)
    if estimable:
        dens = np.asarray([math.exp(d.log_density) for d in densities])
        log_density = float(math.log(np.mean(dens)))
        sig = [d.sigma for d in densities]
        if any(s is None for s in sig):
            sigma_density = None
        else:
            var = np.sum((dens * np.asarray(sig)) ** 2) / len(dens) ** 2
            sigma_density = float(math.sqrt(var) / np.mean(dens))
    else:
        log_density, sigma_density = None, None
    log_vw = 0.0
    if nesting.vw_required:
        if not nesting.vw_first_form_valid:
            return SDDRCheck(nesting.mutation_id, nesting.classification, "not_estimable", None, None,
                             log_bf_ns, sigma_ns, None,
                             {**details, "reason": "VW factor needs supp pi_S within supp pi_L"},
                             method_systematic, ms_source)
        vw_rows = []
        for eps in vw_eps:
            parts = [vw_factor(larger_sample, emb, smaller_priors, larger_priors, eps=float(eps)) for emb in embeddings]
            num = sum(math.exp(p["log_vw"]) * p["n_eff_near"] for p in parts if p["log_vw"] is not None)
            den = sum(p["n_eff_near"] for p in parts if p["log_vw"] is not None)
            vw_rows.append({"eps": float(eps), "parts": parts,
                            "log_vw": math.log(num / den) if den > 0 and num > 0 else None,
                            "n_eff_near": den})
        details["vw"] = vw_rows
        primary = next((row for row in vw_rows if abs(row["eps"] - vw_primary_eps) < 1e-12), vw_rows[0])
        if primary["log_vw"] is None or primary["n_eff_near"] < min_vw_near_ess:
            return SDDRCheck(nesting.mutation_id, nesting.classification, "not_estimable", None, None,
                             log_bf_ns, sigma_ns, None,
                             {**details, "reason": "too few posterior samples near the null for VW"},
                             method_systematic, ms_source)
        log_vw = float(primary["log_vw"])
        details["log_vw_primary"] = log_vw
    if log_density is None:
        # one-sided bound from the 95% Poisson upper limit on the local count
        upper = float(np.log(np.mean([math.exp(d.log_density_upper_limit) for d in densities])))
        bound_larger = log_prior - upper - log_vw  # ln BF_{L/S} >= bound
        details["log_bf_larger_over_smaller_lower_bound_95"] = bound_larger
        status = "not_estimable"
        if log_bf_ns is not None:
            ns_larger = sign * log_bf_ns
            margin = agreement_sigmas * (sigma_ns or 0.0)
            status = "bound_consistent" if ns_larger + margin >= bound_larger else "bound_violated"
        return SDDRCheck(nesting.mutation_id, nesting.classification, status, None, None,
                         log_bf_ns, sigma_ns, None, details, method_systematic, ms_source)
    log_bf_larger = log_prior - log_density - log_vw
    log_bf_child = sign * log_bf_larger
    if log_bf_ns is None:
        return SDDRCheck(nesting.mutation_id, nesting.classification, "computed", log_bf_child,
                         sigma_density, None, None, None, details, method_systematic, ms_source)
    sig_ns = as_float("sigma_ns", sigma_ns if sigma_ns is not None else 0.0, nonnegative=True)
    tolerance = (
        agreement_sigmas * math.sqrt(sig_ns**2 + (sigma_density or 0.0) ** 2) + method_systematic
    )
    status = "agree" if abs(log_bf_ns - log_bf_child) <= tolerance else "disagree"
    return SDDRCheck(nesting.mutation_id, nesting.classification, status, log_bf_child,
                     sigma_density, log_bf_ns, sig_ns, tolerance, details, method_systematic, ms_source)
