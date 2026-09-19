"""Analytic one-dimensional HBI toys for validating the analysis estimators.

Observation model (MODEL_COMPARISON_MATH.md Sec. 6.3): a scalar source
parameter ``x``; the observed datum is ``d = x + sigma_obs * eps`` and a source
is detected iff ``d > d_th``, so ``p_det(x) = Phi((x - d_th) / sigma_obs)``.
PE samples are exact posterior draws under a flat PE prior,
``x_ij ~ N(d_i, sigma_obs)`` (``log_ref_density = 0``); injections are drawn
from ``N(0, s_draw)`` and detected with the same statistic. For Gaussian
populations the per-event likelihood and the selection exposure are analytic::

    ell_i(Lambda) = E_{x ~ N(d_i, sigma_obs)}[p_pop(x | Lambda)] = N(d_i; mu, sqrt(s^2 + sigma_obs^2))
    A(Lambda)     = Phi((mu - d_th) / sqrt(s^2 + sigma_obs^2))

so evidences follow from quadrature and every Monte-Carlo realization of the PE
samples and injections has an exactly known target. The populations are JAX
callables with the ``population_model(samples, hyperparameters)`` contract of
the HBI engine; they are validation fixtures only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping

import numpy as np
from scipy.special import log_ndtr, logsumexp

from gwpop_search.data import Campaign, CoordinateBasis, PosteriorCatalog, SelectionCatalog, SelectionMode

TOY_BASIS = CoordinateBasis(
    name="toy_scalar_x",
    coordinates=("x",),
    frame="toy",
    spin_parameterization="none",
    density_measure="dx",
    version="1",
)
LOG2PI = math.log(2.0 * math.pi)


def _jnp():
    import jax.numpy as jnp

    return jnp


def _log_normal(xp, x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - xp.log(sigma) - 0.5 * LOG2PI


@dataclass(frozen=True)
class GaussianToy:
    """``x ~ N(mu, sigma)``; with ``delta`` the symmetric mixture
    ``0.5 N(mu - delta, sigma) + 0.5 N(mu + delta, sigma)``.

    Hyperparameters ``mu`` and ``sigma``; ``fixed`` supplies values for
    parameters that are not sampled (e.g. ``{"sigma": 1.0}``).
    """

    delta: float | None = None
    fixed: Mapping[str, float] = field(default_factory=dict)
    required_fields = ("x",)

    def _hp(self, hp):
        merged = dict(self.fixed)
        merged.update(hp)
        return merged["mu"], merged["sigma"]

    def __call__(self, samples, hyperparameters):
        jnp = _jnp()
        x = jnp.asarray(samples["x"])
        mu, sigma = self._hp(hyperparameters)
        if self.delta is None:
            return _log_normal(jnp, x, mu, sigma)
        a = _log_normal(jnp, x, mu - self.delta, sigma)
        b = _log_normal(jnp, x, mu + self.delta, sigma)
        return jnp.logaddexp(a, b) + math.log(0.5)

    def to_config(self) -> dict[str, object]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "delta": self.delta,
            "fixed": dict(sorted(self.fixed.items())),
        }

    # -- analytic terms (numpy, broadcasting over hyperparameter arrays) ----

    def exact_event_log_likelihoods(self, d, sigma_obs, hp) -> np.ndarray:
        """``log ell_i`` ``[..., N]`` for data ``d`` ``[N]``."""
        mu, sigma = self._hp(hp)
        mu = np.asarray(mu, dtype=float)[..., None]
        se = np.sqrt(np.asarray(sigma, dtype=float) ** 2 + sigma_obs**2)[..., None]
        d = np.asarray(d, dtype=float)
        if self.delta is None:
            return _log_normal(np, d, mu, se)
        return np.logaddexp(
            _log_normal(np, d, mu - self.delta, se), _log_normal(np, d, mu + self.delta, se)
        ) + math.log(0.5)

    def exact_log_exposure(self, d_th, sigma_obs, hp) -> np.ndarray:
        mu, sigma = self._hp(hp)
        mu = np.asarray(mu, dtype=float)
        se = np.sqrt(np.asarray(sigma, dtype=float) ** 2 + sigma_obs**2)
        if self.delta is None:
            return log_ndtr((mu - d_th) / se)
        return np.logaddexp(
            log_ndtr((mu - self.delta - d_th) / se), log_ndtr((mu + self.delta - d_th) / se)
        ) + math.log(0.5)


@dataclass(frozen=True)
class TruncatedGaussianToy:
    """``x ~ N(mu, sigma)`` truncated to ``x <= xmax`` (a hard population edge).

    Hyperparameters ``mu`` and ``xmax``; ``sigma`` is fixed. An event whose PE
    samples all lie above ``xmax`` has zero likelihood, like the ``m1 <= mmax``
    edge of the mass models.
    """

    sigma: float = 1.0
    required_fields = ("x",)

    def __call__(self, samples, hyperparameters):
        jnp = _jnp()
        from jax.scipy.special import log_ndtr as jlog_ndtr

        x = jnp.asarray(samples["x"])
        mu, xmax = hyperparameters["mu"], hyperparameters["xmax"]
        logp = _log_normal(jnp, x, mu, self.sigma) - jlog_ndtr((xmax - mu) / self.sigma)
        return jnp.where(x <= xmax, logp, -jnp.inf)

    def to_config(self) -> dict[str, object]:
        return {"class": f"{type(self).__module__}.{type(self).__qualname__}", "sigma": self.sigma}


@dataclass(frozen=True)
class PeakMixtureToy:
    """``x ~ (1 - f) N(mu, 1) + f N(m, peak_sigma)``: ``f = 0`` is a boundary null.

    Hyperparameters ``f`` (the peak fraction) and ``m`` (the peak location,
    unidentified at ``f = 0``); the base mean ``mu`` is a hyperparameter when
    ``fit_base_mean`` and 0 otherwise. With ``include_peak=False`` the model is
    the nested ``N(mu, 1)`` (``N(0, 1)`` takes only an unused ``dummy``
    hyperparameter: dynesty needs at least one dimension).
    """

    peak_sigma: float = 0.5
    include_peak: bool = True
    fit_base_mean: bool = False
    required_fields = ("x",)

    def __call__(self, samples, hyperparameters):
        jnp = _jnp()
        x = jnp.asarray(samples["x"])
        mu = hyperparameters["mu"] if self.fit_base_mean else 0.0
        base = _log_normal(jnp, x, mu, 1.0)
        if not self.include_peak:
            return base if self.fit_base_mean else base + 0.0 * hyperparameters["dummy"]
        f, m = hyperparameters["f"], hyperparameters["m"]
        peak = _log_normal(jnp, x, m, self.peak_sigma)
        safe_f = jnp.clip(f, 1e-300, 1.0)
        mixture = jnp.logaddexp(jnp.log1p(-jnp.clip(f, 0.0, 1.0 - 1e-16)) + base, jnp.log(safe_f) + peak)
        return jnp.where(f <= 0.0, base, mixture)

    def to_config(self) -> dict[str, object]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "peak_sigma": self.peak_sigma,
            "include_peak": self.include_peak,
            "fit_base_mean": self.fit_base_mean,
        }


@dataclass(frozen=True)
class ToyObservation:
    """Noise level, detection threshold and injection draw width."""

    sigma_obs: float = 0.5
    d_th: float = 0.5
    s_draw: float = 2.0

    def detect(self, rng: np.random.Generator, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x + self.sigma_obs * rng.normal(size=x.shape) > self.d_th


def toy_observed_data(
    rng: np.random.Generator,
    sampler: Callable[[np.random.Generator, int], np.ndarray],
    n_events: int,
    obs: ToyObservation,
) -> np.ndarray:
    """Observed data ``d`` of ``n_events`` detected sources drawn with ``sampler``."""
    out: list[float] = []
    while len(out) < n_events:
        x = np.asarray(sampler(rng, max(64, 2 * n_events)), dtype=float)
        d = x + obs.sigma_obs * rng.normal(size=x.shape)
        out.extend(d[d > obs.d_th].tolist())
    return np.asarray(out[:n_events])


def toy_posterior_catalog(
    rng: np.random.Generator, d, n_pe: int, obs: ToyObservation, *, names=None
) -> PosteriorCatalog:
    d = np.asarray(d, dtype=float)
    n_events = d.size
    samples = (d[:, None] + obs.sigma_obs * rng.normal(size=(n_events, n_pe))).ravel()
    names = tuple(names) if names is not None else tuple(f"TOY{i:03d}" for i in range(n_events))
    return PosteriorCatalog(
        event_names=names,
        offsets=np.arange(n_events + 1) * n_pe,
        samples={"x": samples},
        log_ref_density=np.zeros(samples.size),
        basis=TOY_BASIS,
        metadata={"toy": "flat-PE-prior gaussian noise", "sigma_obs": obs.sigma_obs},
    )


def toy_selection_catalog(rng: np.random.Generator, n_draw: int, obs: ToyObservation) -> SelectionCatalog:
    x = rng.normal(0.0, obs.s_draw, size=int(n_draw))
    detected = obs.detect(rng, x)
    xd = x[detected]
    log_draw = _log_normal(np, xd, 0.0, obs.s_draw)
    return SelectionCatalog(
        samples={"x": xd},
        log_draw_density=log_draw,
        campaign_id=np.asarray(["TOY"] * xd.size),
        campaigns=(Campaign("TOY", n_draw=int(n_draw), observing_time_yr=1.0),),
        basis=TOY_BASIS,
        mode=SelectionMode.RAW_DRAW,
        metadata={"toy": "gaussian draws, threshold detection", "d_th": obs.d_th},
    )


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid2D:
    """Tensor trapezoid grid on a box inside a uniform prior box."""

    names: tuple[str, str]
    points: np.ndarray  # [G, 2]
    log_quadrature: np.ndarray  # [G] log(trapezoid weight / prior volume)

    @property
    def size(self) -> int:
        return int(self.points.shape[0])


def grid_2d(names, box, prior_box, n: int) -> Grid2D:
    (a0, b0), (a1, b1) = box
    (p0, q0), (p1, q1) = prior_box
    g0 = np.linspace(a0, b0, n)
    g1 = np.linspace(a1, b1, n)
    w0 = np.full(n, g0[1] - g0[0])
    w0[[0, -1]] *= 0.5
    w1 = np.full(n, g1[1] - g1[0])
    w1[[0, -1]] *= 0.5
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    logq = (np.log(w0)[:, None] + np.log(w1)[None, :]).ravel() - math.log((q0 - p0) * (q1 - p1))
    return Grid2D(tuple(names), np.column_stack([G0.ravel(), G1.ravel()]), logq)


def grid_log_evidence(log_likelihood, grid: Grid2D) -> float:
    return float(logsumexp(np.asarray(log_likelihood) + grid.log_quadrature))


def grid_posterior_weights(log_likelihood, grid: Grid2D) -> np.ndarray:
    lp = np.asarray(log_likelihood) + grid.log_quadrature
    return np.exp(lp - logsumexp(lp))
