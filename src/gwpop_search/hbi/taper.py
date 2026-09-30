"""Variance taper on the Monte-Carlo log-likelihood estimator (inside the likelihood).

The hierarchical likelihood is a Monte-Carlo estimate: every event integral
``I_i`` and the selection exposure ``xi`` are importance-sampling averages. The
variance of the rate-marginalised (shape) log-likelihood estimator is, to
first order (Farr 2019, RNAAS 3, 66; Essick & Farr 2022, arXiv:2204.00461;
Talbot & Golomb 2023, MNRAS 526, 3495)::

    sigma^2(Lambda) = sum_i Var[ln I_i] + N^2 Var[xi] / xi^2

with ``Var[ln I_i] = 1/ESS_i - 1/n_i`` and ``Var[xi]/xi^2`` combined over the
injection campaigns. :mod:`gwpop_search.hbi.jax_backend` computes it with the
same estimator as the importance diagnostics
(:func:`gwpop_search.inference.dynesty_backend.build_importance_diagnostics`).

Where ``sigma^2`` is large the estimate is unreliable. GWTC-5.0 (LVK 2026,
arXiv:2605.27226, Sec. III) "require[s] a maximum variance of 1 on the
population likelihood estimator ... implemented as a sharp or
smoothly-tapered cutoff [Callister & Farr 2024, PRX 14, 021005]". The two
forms implemented here:

``sharp`` (gwpopulation, ``HyperparameterLikelihood.log_likelihood_ratio``,
ColmTalbot/gwpopulation at b3a34f9)::

    ln L_tapered = ln L - inf * [sigma^2 > threshold]

``smooth`` (the Callister & Farr 2024 taper; their public code,
tcallister/autoregressive-bbh-inference at 53573e9,
``code/autoregressive_mass_models.py``, applies
``numpyro.factor("Neff_inj_penalty", log(1 / (1 + (N_eff / (4 N_obs))**-30)))``
to the injection effective sample size). Applied to the variance, with
``x = sigma^2 / threshold`` and steepness ``p`` (30 in Callister & Farr)::

    ln T(sigma^2) = -ln(1 + x^p) = -softplus(p ln x)

Properties (pinned by ``tests/test_variance_taper.py``):

* ``T -> 1`` (``ln T -> 0``) for ``sigma^2 << threshold``; ``T = 1/2`` exactly
  at the threshold; ``ln T ~ -p ln x`` (power-law suppression ``x^-p``)
  above it. At ``p = 30`` the taper is below 1 % suppression for
  ``sigma^2 < 0.858 threshold`` and ``T(2 threshold) = 9.3e-10``.
* ``C^inf`` in ``sigma^2 > 0`` and continuous at ``sigma^2 = 0`` (``ln T = 0``)
  and at ``sigma^2 = inf`` (``ln T = -inf``).
* ``p -> inf`` recovers the sharp cut (except at ``sigma^2 = threshold``
  exactly, where the smooth taper keeps ``T = 1/2``).

The GWTC-5 paper does not print the smooth functional form; the LVK
pipeline configuration (gwpopulation_pipe on git.ligo.org) was not
accessible when this was written. The Callister & Farr form with ``p = 30``
applied to ``sigma^2`` is our reading of "[68, e.g.,]"; the steepness is a
configurable, recorded part of the likelihood identity.

Evidence semantics: the tapered likelihood ``L T`` is the likelihood that is
sampled and integrated, so ``Z = int L T pi dLambda`` counts only the region
where the estimate is reliable (weighted by ``T``). A tapered evidence is
comparable only with evidences under the same taper.

The *taper region* (for the posterior taper-mass diagnostic) is where the
taper suppresses the likelihood by more than ``region_suppression`` (1 %),
i.e. ``T < 1 - region_suppression``; for the sharp cut it is
``sigma^2 > threshold``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Mapping

import numpy as np

TAPER_KINDS = ("smooth", "sharp")
# Callister & Farr (2024) steepness of the logistic-in-log taper.
CALLISTER_FARR_EXPONENT = 30.0
DEFAULT_REGION_SUPPRESSION = 0.01


def _real(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value}")
    return value


@dataclass(frozen=True)
class VarianceTaper:
    """Taper of the shape log-likelihood on its Monte-Carlo variance.

    ``threshold`` is the variance ``sigma^2_lnL`` at which the taper acts
    (GWTC-5 default 1; sensitivity reruns at 2). ``exponent`` is the smooth
    taper's steepness ``p`` (ignored by ``sharp``). ``region_suppression``
    defines the taper region reported by the taper-mass diagnostic; it does
    not change the likelihood.
    """

    kind: str = "smooth"
    threshold: float = 1.0
    exponent: float = CALLISTER_FARR_EXPONENT
    region_suppression: float = DEFAULT_REGION_SUPPRESSION

    def __post_init__(self) -> None:
        if self.kind not in TAPER_KINDS:
            raise ValueError(f"variance taper kind must be one of {TAPER_KINDS}; got {self.kind!r}")
        threshold = _real("variance taper threshold", self.threshold)
        if not threshold > 0.0:
            raise ValueError("variance taper threshold must be positive")
        exponent = _real("variance taper exponent", self.exponent)
        if not exponent >= 1.0:
            raise ValueError("variance taper exponent must be >= 1")
        suppression = _real("variance taper region_suppression", self.region_suppression)
        if not 0.0 < suppression < 1.0:
            raise ValueError("variance taper region_suppression must lie in (0, 1)")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "region_suppression", suppression)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "threshold": float(self.threshold),
            "exponent": float(self.exponent),
            "region_suppression": float(self.region_suppression),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VarianceTaper":
        payload = dict(payload)
        known = {"kind", "threshold", "exponent", "region_suppression"}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown variance taper field(s): {unknown}")
        return cls(**payload)

    @classmethod
    def coerce(cls, value) -> "VarianceTaper | None":
        if value is None or isinstance(value, VarianceTaper):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(f"variance_taper must be a VarianceTaper, a mapping or None; got {value!r}")

    # -- taper region ----------------------------------------------------------

    @property
    def region_onset(self) -> float:
        """Smallest ``sigma^2`` inside the taper region (``T < 1 - region_suppression``)."""
        if self.kind == "sharp":
            return float(self.threshold)
        s = self.region_suppression
        # -ln(1 + x^p) < ln(1 - s)  <=>  x > (1 / (1 - s) - 1)^(1/p) = (s / (1 - s))^(1/p)
        return float(self.threshold * (s / (1.0 - s)) ** (1.0 / self.exponent))

    def in_region(self, variance) -> np.ndarray:
        """Boolean mask: ``sigma^2`` in the taper region (NaN counts as inside)."""
        v = np.asarray(variance, dtype=float)
        if self.kind == "sharp":
            return ~(v <= self.threshold)
        return ~(v <= self.region_onset)

    # -- the taper ---------------------------------------------------------

    def log_taper(self, variance) -> np.ndarray:
        """NumPy ``ln T(sigma^2)``; ``sigma^2 = 0 -> 0``, ``inf``/NaN ``-> -inf``."""
        return log_taper_numpy(variance, self)

    def log_taper_jax(self, variance):
        """JAX ``ln T(sigma^2)`` (jit/vmap/grad safe)."""
        return log_taper_jax(variance, self)


def log_taper_numpy(variance, taper: VarianceTaper) -> np.ndarray:
    v = np.asarray(variance, dtype=float)
    out = np.full(v.shape, -np.inf)
    ok = np.isfinite(v) & (v >= 0.0)
    if taper.kind == "sharp":
        out[ok] = np.where(v[ok] > taper.threshold, -np.inf, 0.0)
        return out
    positive = ok & (v > 0.0)
    out[ok & (v == 0.0)] = 0.0
    z = taper.exponent * (np.log(v[positive]) - math.log(taper.threshold))
    out[positive] = -np.logaddexp(0.0, z)
    return out


def log_taper_jax(variance, taper: VarianceTaper):
    import jax.numpy as jnp

    v = jnp.asarray(variance)
    ok = jnp.isfinite(v) & (v >= 0.0)
    if taper.kind == "sharp":
        return jnp.where(ok & (v <= taper.threshold), 0.0, -jnp.inf)
    positive = ok & (v > 0.0)
    # Safe log argument so neither the value nor the gradient is NaN at v = 0.
    safe = jnp.where(positive, v, 1.0)
    z = taper.exponent * (jnp.log(safe) - math.log(taper.threshold))
    smooth = -jnp.logaddexp(0.0, z)
    return jnp.where(positive, smooth, jnp.where(ok, 0.0, -jnp.inf))


def taper_region_summary(
    variance,
    weights,
    taper: VarianceTaper,
) -> dict[str, object]:
    """Posterior taper-mass diagnostic from per-sample ``sigma^2`` and weights.

    ``weights`` are (unnormalised, non-negative) posterior weights of the
    samples, e.g. dynesty importance weights. Returns the posterior fraction
    inside the taper region, the fraction above the threshold, the posterior
    mean of ``T`` and quantiles of ``sigma^2``.
    """
    v = np.asarray(variance, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.shape != w.shape:
        raise ValueError("variance and weights must have the same length")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    w = w / w.sum()
    inside = taper.in_region(v)
    above = ~(v <= taper.threshold)
    log_t = log_taper_numpy(v, taper)
    finite_v = np.where(np.isfinite(v), v, np.inf)
    order = np.argsort(finite_v, kind="stable")
    cdf = np.cumsum(w[order])

    def wq(q: float) -> float:
        index = int(np.searchsorted(cdf, q, side="left"))
        return float(finite_v[order][min(index, v.size - 1)])

    kish = float(1.0 / np.sum(w**2))
    return {
        "taper": taper.to_dict(),
        "region_onset_variance": taper.region_onset,
        "posterior_mass_in_taper_region": float(np.sum(w[inside])),
        "posterior_mass_above_threshold": float(np.sum(w[above])),
        "posterior_mean_taper": float(np.sum(w * np.exp(log_t))),
        "variance_quantiles": {
            "q0.5": wq(0.5),
            "q0.9": wq(0.9),
            "q0.99": wq(0.99),
            "max": float(np.max(finite_v)),
        },
        "n_samples": int(v.size),
        "kish_ess": kish,
    }
