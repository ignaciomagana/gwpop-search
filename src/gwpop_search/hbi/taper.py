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
smoothly-tapered cutoff [Callister & Farr 2024, e.g.]". The two forms
implemented here (sources and the release check are in
``staging/v2/TAPER_FORM.md``):

``sharp`` (the default; the exact LVK implementation for the BP2P Default fit
that the v2 root ports): gwpopulation ``hyperpe.py`` L185-189
(ColmTalbot/gwpopulation at b3a34f9), called by gwpopulation_pipe
``data_analysis.create_likelihood`` with ``--maximum-uncertainty``::

    ln L_tapered = ln L - inf * [sigma^2 > threshold]      (sigma^2 = threshold kept)

gwpopulation compares ``maximum_uncertainty`` directly with the *variance*
(its docstring calls it a standard deviation; the squared value it stores is
unused), so its cut is ``sigma^2 <= maximum_uncertainty``; GWInferno's
``max_variance_cut`` is the same (``less_equal(variance, 1)``). The GWTC-5
Default release posteriors end exactly at sigma^2 = 1 (and at 4 for the
relaxed run) with no sample beyond. Difference kept deliberately: a NaN
variance is cut here (-inf); gwpopulation does not cut it.

``smooth`` (diagnostic alternative, not the LVK Default form): the
Callister & Farr (2024) functional form ``S(x) = 1/(1 + x^-30)`` -- which they
apply to the injection effective sample size ``N_eff / (4 N_obs)``
(tcallister/autoregressive-bbh-inference at 9f78be3,
``code/autoregressive_mass_models.py`` L206), not to the variance -- applied
here to the variance, with ``x = sigma^2 / threshold`` and steepness ``p``::

    ln T(sigma^2) = -ln(1 + x^p) = -softplus(p ln x)

Properties of the smooth form (pinned by ``tests/test_variance_taper.py``):

* ``T -> 1`` (``ln T -> 0``) for ``sigma^2 << threshold``; ``T = 1/2`` exactly
  at the threshold; ``ln T ~ -p ln x`` (power-law suppression ``x^-p``)
  above it. At ``p = 30`` the taper is below 1 % suppression for
  ``sigma^2 < 0.858 threshold`` and ``T(2 threshold) = 9.3e-10``.
* ``C^inf`` in ``sigma^2 > 0`` and continuous at ``sigma^2 = 0`` (``ln T = 0``)
  and at ``sigma^2 = inf`` (``ln T = -inf``).
* ``p -> inf`` recovers the sharp cut (except at ``sigma^2 = threshold``
  exactly, where the smooth taper keeps ``T = 1/2``).

Evidence semantics: the tapered likelihood ``L T`` is the likelihood that is
sampled and integrated, so ``Z = int L T pi dLambda`` counts only the region
where the estimate is reliable (weighted by ``T``). A tapered evidence is
comparable only with evidences under the same taper.

The *taper region* (for the posterior taper-mass diagnostic) is where the
Monte-Carlo fluctuation of the taper itself is not negligible:

* ``smooth``: where the taper suppresses the likelihood by more than
  ``region_suppression`` (1 %), i.e. ``T < 1 - region_suppression``;
* ``sharp``: the band below the wall, ``sigma^2 > (1 - sharp_region_band)
  threshold`` (``sharp_region_band`` 0.05, DRAFT: the assumed relative
  Monte-Carlo error of ``sigma^2_hat``). Under a sharp cut no posterior mass
  lies above the threshold; mass within the band can flip in or out of the
  cut under a different Monte-Carlo realisation, which the first-order
  (cut-free) error of ``ln Z`` does not include. The band mass is reported,
  not gating: the v2 claim criteria instead measure the effect of the cut
  directly, from ``Z(c') = Z(c) P_post(sigma^2 <= c' | cut c)`` for a sharp
  cut (:func:`posterior_mass_below`; D2 at ``c = 1`` and ``c' = 0.9``).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Mapping

import numpy as np

TAPER_KINDS = ("smooth", "sharp")
#: The LVK GWTC-5 form (gwpopulation ``maximum_uncertainty``): see the module doc.
DEFAULT_TAPER_KIND = "sharp"
# Callister & Farr (2024) steepness of the logistic-in-log taper.
CALLISTER_FARR_EXPONENT = 30.0
DEFAULT_REGION_SUPPRESSION = 0.01
#: Diagnostic band below a sharp cut (assumed relative MC error of sigma^2_hat; DRAFT).
DEFAULT_SHARP_REGION_BAND = 0.05


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

    ``kind`` is ``"sharp"`` (default; the LVK GWTC-5 cut) or ``"smooth"``.
    ``threshold`` is the variance ``sigma^2_lnL`` at which the taper acts
    (GWTC-5 default 1; sensitivity reruns at 2). ``exponent`` is the smooth
    taper's steepness ``p`` (ignored by ``sharp``). ``region_suppression``
    (smooth) and ``sharp_region_band`` (sharp) define the taper region
    reported by the taper-mass diagnostic; neither changes the likelihood.
    """

    kind: str = DEFAULT_TAPER_KIND
    threshold: float = 1.0
    exponent: float = CALLISTER_FARR_EXPONENT
    region_suppression: float = DEFAULT_REGION_SUPPRESSION
    sharp_region_band: float = DEFAULT_SHARP_REGION_BAND

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
        band = _real("variance taper sharp_region_band", self.sharp_region_band)
        if not 0.0 < band < 1.0:
            raise ValueError("variance taper sharp_region_band must lie in (0, 1)")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(self, "region_suppression", suppression)
        # The band is a sharp-cut diagnostic only: a smooth taper ignores it, so it is
        # normalised to the default there (and left out of the smooth serialization,
        # which keeps smooth identities written before the field existed unchanged).
        object.__setattr__(
            self, "sharp_region_band", band if self.kind == "sharp" else DEFAULT_SHARP_REGION_BAND
        )

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        payload = {
            "kind": self.kind,
            "threshold": float(self.threshold),
            "exponent": float(self.exponent),
            "region_suppression": float(self.region_suppression),
        }
        if self.kind == "sharp":
            payload["sharp_region_band"] = float(self.sharp_region_band)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VarianceTaper":
        payload = dict(payload)
        known = {"kind", "threshold", "exponent", "region_suppression", "sharp_region_band"}
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
        """Smallest ``sigma^2`` inside the taper region.

        Smooth: ``T < 1 - region_suppression``. Sharp: the band
        ``sigma^2 > (1 - sharp_region_band) threshold`` below the cut.
        """
        if self.kind == "sharp":
            return float(self.threshold * (1.0 - self.sharp_region_band))
        s = self.region_suppression
        # -ln(1 + x^p) < ln(1 - s)  <=>  x > (1 / (1 - s) - 1)^(1/p) = (s / (1 - s))^(1/p)
        return float(self.threshold * (s / (1.0 - s)) ** (1.0 / self.exponent))

    def in_region(self, variance) -> np.ndarray:
        """Boolean mask: ``sigma^2`` in the taper region (NaN counts as inside)."""
        v = np.asarray(variance, dtype=float)
        return ~(v <= self.region_onset)

    def region_definition(self) -> str:
        """Human-readable definition of the taper region (for reports)."""
        if self.kind == "sharp":
            return (
                f"sigma^2 > {self.region_onset:.6g} = (1 - {self.sharp_region_band:g}) x threshold "
                f"{self.threshold:g} (band below the sharp cut)"
            )
        return (
            f"T(sigma^2) < {1.0 - self.region_suppression:g}, i.e. sigma^2 > {self.region_onset:.6g}"
        )

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


def cut_key(cut: float) -> str:
    """JSON key of a variance cut in ``posterior_mass_below`` (``repr`` of the float: ``"0.9"``, ``"1.0"``)."""
    return repr(float(cut))


def normalize_cuts(cuts) -> tuple[float, ...]:
    """Sorted, de-duplicated, finite positive variance cuts."""
    out = []
    for cut in cuts or ():
        value = _real("variance cut", cut)
        if not value > 0.0:
            raise ValueError(f"variance cuts must be positive; got {value}")
        out.append(value)
    return tuple(sorted(set(out)))


def posterior_mass_below(
    variance,
    weights,
    cuts,
    *,
    reference_kish_ess: float | None = None,
) -> dict[str, dict[str, float]]:
    """Weighted posterior fraction ``P(sigma^2 <= c)`` for each cut ``c``, with its error.

    Under a sharp cut at ``c`` (likelihood ``x 1[sigma^2 <= c]``) the evidence
    at a tighter cut ``c' < c`` is exactly ``Z(c') = Z(c) P_post(sigma^2 <= c'
    | cut c)``, so these fractions give the evidence at the tighter cuts
    without a rerun (claims_v2 D2). ``sigma^2 = NaN`` counts as above every
    cut (it is cut by the likelihood).

    The Monte-Carlo (binomial) error is ``sqrt(p (1 - p) / n_eff)`` with
    ``n_eff`` the Kish ESS of ``weights``; when the points are a resampled
    subset of a weighted sample whose Kish ESS is ``reference_kish_ess``, the
    two sampling stages add, ``1/n_eff = 1/kish(subset) + 1/reference``.
    Returns ``{cut_key(c): {"cut", "fraction", "error", "n_eff"}}``.
    """
    v = np.asarray(variance, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.shape != w.shape:
        raise ValueError("variance and weights must have the same length")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    w = w / w.sum()
    n_eff = float(1.0 / np.sum(w**2))
    if reference_kish_ess is not None:
        reference = _real("reference_kish_ess", reference_kish_ess)
        if not reference > 0.0:
            raise ValueError("reference_kish_ess must be positive")
        n_eff = float(1.0 / (1.0 / n_eff + 1.0 / reference))
    out = {}
    for cut in normalize_cuts(cuts):
        p = float(min(1.0, max(0.0, np.sum(w[v <= cut]))))
        out[cut_key(cut)] = {
            "cut": float(cut),
            "fraction": p,
            "error": float(math.sqrt(p * (1.0 - p) / n_eff)),
            "n_eff": n_eff,
        }
    return out


def taper_region_summary(
    variance,
    weights,
    taper: VarianceTaper,
    *,
    cuts=(),
    reference_kish_ess: float | None = None,
) -> dict[str, object]:
    """Posterior taper-mass diagnostic from per-sample ``sigma^2`` and weights.

    ``weights`` are (unnormalised, non-negative) posterior weights of the
    samples, e.g. dynesty importance weights. Returns the posterior fraction
    inside the taper region, the fraction above the threshold, the posterior
    mean of ``T`` and quantiles of ``sigma^2``. With ``cuts``, also
    ``posterior_mass_below`` (:func:`posterior_mass_below`: ``P(sigma^2 <= c)``
    with its Kish-ESS binomial error, per cut).
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
    out = {
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
    if cuts:
        out["posterior_mass_below"] = posterior_mass_below(
            v, w, cuts, reference_kish_ess=reference_kish_ess
        )
    return out
