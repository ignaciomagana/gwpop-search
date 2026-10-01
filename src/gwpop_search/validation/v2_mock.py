"""DAG-consistent closure mocks for the v2 atom search (PILOT_PLAN section (b)).

The generator follows the physical DAG of Essick & Fishbach 2023
(arXiv:2310.02017; the ``mock-data-dag`` rules):

    Lambda (truth) -> theta ~ p_pop(theta | Lambda)
    (run k, projection Theta, noise n)  -- one realisation per system
    rho_obs = A_k Theta g(theta) + n                 (the DATA detection statistic)
    detected  <=>  rho_obs > rho_thr                 (same rule for events and injections)
    x_obs = (ln m1_det, q, chi_eff) + N(0, Sigma(rho_obs))   (PE data; widths from rho_obs only)
    PE: exact draws from  L(rho_obs, x_obs | theta) pi_PE(theta)

* **Detection on data (rule 1)** and **one noise realisation used everywhere
  (rule 2)**: ``rho_obs`` carries the noise draw that decides detection, and
  the PE likelihood contains the same ``rho_obs`` through the amplitude term
  ``N(rho_obs; A_k Theta g(theta), 1)`` (Theta marginalised over its prior,
  as the sky/orientation is in real PE).
* **Same statistic and threshold for injections and events (rule 3)**: both
  are produced by :func:`simulate_detections`. The events are a
  ``p_pop / p_draw``-weighted resample of the detected systems of an event
  pool built by that same function (run labels drawn per system with
  probability ``T_k / T``), i.e. "resample detected injections weighted by
  p_pop / p_draw, per run".
* **Data-derived analyst constants (rule 4)**: the PE widths are
  ``c_j * rho_thr / rho_obs`` and the PE correlation is a constant (the
  median per-event posterior correlation of the real catalog); nothing is
  frozen at the truth.
* **Coverage (rule 6)**: the injection draw distribution covers the fixed v2
  population support (m1 in [3, 300], q in [0.001, 1], z in [0, 1.9],
  chi_eff in [-1, 1]) for every hyperparameter value, so the detected
  population outside the draw support is zero by construction; it is
  checked on the built artifact (:func:`coverage_gate`).

Why the canonical (real) found injections are not reused as the mock
selection: their detection noise realisation (pipeline FAR for O3/O4) is not
available, so it cannot be shared with the synthetic PE (rule 2), and their
optimal SNR is not a function the PE likelihood can evaluate at arbitrary
theta (using the injected value would freeze a truth-dependent constant,
rule 4). Instead the mock campaign is **calibrated to the canonical one**:
per run, the SNR amplitude ``A_k`` reproduces the median redshift of the
canonical detected R0 population, and the run time ``T_k`` its share of the
detections (:data:`CANONICAL_RUN_TARGETS`); one global detector-frame
total-mass roll-off (1 + (M/M_ro)^3)^(-p/3) reproduces the canonical pooled
m1 90 % quantile (M_ro) and the detected fraction above m1 = 100 (p), which
also brings the 99 % quantile and P(m1 > 200) to the canonical values
(:data:`CANONICAL_POOLED_TARGETS`). The x_obs widths are calibrated to the
median per-event posterior widths of the real 259-event catalog
(:data:`REAL_PE_WIDTH_MEDIANS`); the luminosity distance is informed only by
the amplitude datum (with the projection marginalised), so its posterior
width is an outcome, not a calibration.

Selection Monte-Carlo variance: 80 % of the injections are drawn from a
population proxy (the R0 fiducial (m1, q) density tabulated on a grid, its
redshift evolution, a broad chi_eff normal), so the weights p_pop / p_draw at
the truths are nearly constant; sigma^2_lnL at the truth must sit well inside
the v2 sharp cut (gate iv of :func:`build_mock`).

Gates b-0 (:func:`build_mock`): (i) the delta-PE C2-slope profile over an
ensemble of same-pool catalogs (machinery check; binding) with the realised
catalog reported against the pre-declared representativeness rule
|MLE - truth| <= 3 sd_ens; (ii) draw-support and PE-box coverage; (iii) G12 on
every depth-1 node; (iv) sigma^2_lnL at the truth <= 0.5 x the cut threshold
and the per-event term <= 1.5 x the canonical catalog's.

Known simplifications (a mock, not a waveform model): one projection factor
(single-interferometer antenna pattern) for every run, the SNR scaling of
:func:`snr_unit`, Gaussian x_obs in (ln m1_det, q, chi_eff), the same
threshold rho_obs > 10 in every run (the canonical O1/O2 rule; O3/O4 use
FAR < 1/yr), the PE prior ∝ m1_det d_L^2 (d_L <= d_L(z_max)) with a uniform
chi_eff prior, and a RAW_DRAW selection (frozen-selection null replays need
an estimator-ready set, so a mock campaign runs with no null replays).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

MOCK_FORMAT_VERSION = "gwpop-search-v2-closure-mock-1.0"
MOCK_KINDS = ("closure_widthq", "null")

#: LVK GWTC-5.0 parametric fiducial (BP2P + PowerLawRedshift) popsummary
#: medians, as tabulated in staging/v2/PILOT_PLAN.md (read 2026-09-30 from the
#: Zenodo 20292639 popsummary tied in provenance/v2/lvk_tie.json).
LVK_GWTC5_FIDUCIAL_MEDIANS: dict[str, float] = {
    "alpha_1": 1.48, "alpha_2": 5.42, "break_mass": 37.45,
    "mpp_1": 9.91, "sigpp_1": 0.78, "mpp_2": 32.33, "sigpp_2": 5.73,
    "lam_0": 0.40, "lam_1": 0.55,
    "mlow_1": 4.49, "mlow_2": 3.46, "delta_m_1": 3.53, "delta_m_2": 4.81,
    "beta": 1.04, "lamb": 2.54,
}

#: chi_eff truths (DRAFT, operator decision 2026-09-30). Closure: C2, ln-width
#: linear in q with the intercept at q = 1: sigma(1) = 0.05, slope -2.2, so
#: sigma(0.5) = 0.150. Null: R0 (constant width). Its width 0.09 is the
#: detected-population RMS of the closure sigma(q) (0.0913, computed on the
#: canonical selection weighted to the R0 fiducial), so the two mocks have the
#: same detected chi_eff spread and differ only in its q dependence.
CLOSURE_CHI = {"chi_mu": 0.03, "chi_log_sigma": math.log(0.05), "chi_log_sigma_q_slope": -2.2}
NULL_CHI = {"chi_mu": 0.03, "chi_log_sigma": math.log(0.09)}
C2_MUTATION_ID = "v2.chieff.log_sigma_q"
C2_SLOPE = "chi_log_sigma_q_slope"

#: Canonical (real) detected R0 population per run, from
#: staging/v2/canonical/selection.h5 with run labels from
#: gwcat/build/v2r2/selection_row_subsets_index.h5, weights p_pop / p_draw at
#: the null truth (R0 fiducial, q_floor 0.001, chi_eff N(0.03, 0.09)),
#: recomputed 2026-09-30 at the q_floor-0.001 root. ``share`` is the run's
#: fraction of sum(p_pop / p_draw); the lists are the weighted 10/50/90 %
#: quantiles of the found injections. Recompute with
#: :func:`canonical_run_targets`.
CANONICAL_RUN_TARGETS: dict[str, dict[str, object]] = {
    "O1": {"share": 0.005617650603793581, "z": [0.061, 0.1679, 0.3672], "m1_source": [10.16, 32.63, 53.4],
           "q": [0.56, 0.831, 0.972], "chi_eff": [-0.081, 0.037, 0.149]},
    "O2": {"share": 0.025501241771420262, "z": [0.0763, 0.2039, 0.4501], "m1_source": [9.98, 32.83, 53.99],
           "q": [0.563, 0.83, 0.97], "chi_eff": [-0.079, 0.035, 0.151]},
    "O3a": {"share": 0.12024133498023182, "z": [0.1167, 0.3243, 0.7048], "m1_source": [10.05, 32.15, 49.68],
           "q": [0.569, 0.83, 0.971], "chi_eff": [-0.079, 0.034, 0.152]},
    "O3b": {"share": 0.11230558491280278, "z": [0.1195, 0.3379, 0.7368], "m1_source": [10.02, 32.35, 49.92],
           "q": [0.565, 0.832, 0.971], "chi_eff": [-0.078, 0.036, 0.153]},
    "O4a": {"share": 0.3275946946095048, "z": [0.1578, 0.4651, 1.0354], "m1_source": [10.28, 32.64, 49.01],
           "q": [0.565, 0.831, 0.971], "chi_eff": [-0.078, 0.037, 0.153]},
    "O4b": {"share": 0.40873949312224667, "z": [0.1634, 0.4855, 1.1054], "m1_source": [10.31, 32.77, 48.94],
           "q": [0.564, 0.829, 0.971], "chi_eff": [-0.079, 0.038, 0.153]},
}
RUN_LABELS = tuple(CANONICAL_RUN_TARGETS)
#: canonical selection T_obs_yr (sum of the per-component analysis times)
T_TOTAL_YR = 3.0456693791669833

#: Median over the 259 real events (staging/v2/canonical/pe.h5) of the
#: per-event posterior standard deviation of (ln m1_detector, q, ln d_L,
#: chi_eff), and the median per-event posterior correlation matrix.
REAL_PE_WIDTH_MEDIANS = {"ln_m1_detector": 0.1275637, "q": 0.1708303,
                         "ln_luminosity_distance": 0.3320497, "chi_eff": 0.1297908}
REAL_PE_CORRELATION = (
    (1.0, -0.839, 0.103, 0.391),
    (-0.839, 1.0, 0.042, 0.063),
    (0.103, 0.042, 1.0, 0.205),
    (0.391, 0.063, 0.205, 1.0),
)
REAL_PE_COORDINATES = ("ln_m1_detector", "q", "ln_luminosity_distance", "chi_eff")
#: x_obs coordinates, their constant likelihood correlation (the real median
#: posterior correlation restricted to them; positive definite) and the
#: calibration targets of their widths
X_NAMES = ("ln_m1_detector", "q", "chi_eff")
PE_CORRELATION = ((1.0, -0.839, 0.391), (-0.839, 1.0, 0.063), (0.391, 0.063, 1.0))
X_WIDTH_TARGETS = tuple(REAL_PE_WIDTH_MEDIANS[k] for k in X_NAMES)

SNR_THRESHOLD = 10.0
#: sharpness s of the SNR mass roll-off (1 + (M_det / M_ro)^s)^(-p/s): with s = 3
#: the calibrated (M_ro, p) reproduce the canonical detected m1 q90, q99 and
#: P(m1 > 100) together (s = 1 cannot: P(m1 > 100) stays >= 2x canonical for
#: any p; review B item 4)
MASS_ROLLOFF_SHARPNESS = 3.0
#: aligned-spin SNR factor (1 + SPIN_SNR_SLOPE chi_eff): aligned spins are louder
SPIN_SNR_SLOPE = 0.2
#: widths are c_j rho_thr / max(rho_obs, RHO_WIDTH_FLOOR) (the floor only
#: matters for undetected systems, e.g. in the SBC test)
RHO_WIDTH_FLOOR = 5.0

#: Uniform-in-(m1_det, m2_det) and Euclidean-volume PE prior on a box:
#: pi(m1_det, q, d_L, chi_eff) ∝ m1_det d_L^2 (chi_eff uniform). The d_L upper
#: edge used for the mocks is d_L(z_max = 1.9) (:func:`pe_box`), so the PE
#: carries no sample above the declared z_max (review B item 9).
PE_BOX = {"m1_detector": (2.0, 2000.0), "q": (0.001, 1.0),
          "luminosity_distance": (1.0, 30000.0), "chi_eff": (-1.0, 1.0)}


def pe_box(cosmo) -> dict:
    """PE_BOX with the d_L upper edge at d_L(DRAW_ZMAX)."""
    box = dict(PE_BOX)
    box["luminosity_distance"] = (PE_BOX["luminosity_distance"][0], float(cosmo.dL_of_z(DRAW_ZMAX)))
    return box

#: Injection draw distribution (source frame). Its support contains the fixed
#: v2 population support for every q_floor in [0.001, 0.05].
DRAW_M1 = (3.0, 300.0)
DRAW_Q_LO = 0.001
DRAW_ZMAX = 1.9
#: Population-proxy component (fraction DRAW_PROXY_FRACTION of the draws): the
#: R0 fiducial (null truth) (m1, q) density tabulated on a (ln m1, q) grid
#: (piecewise constant, exact density), z ∝ dVc/dz (1+z)^(kappa-1) at the
#: fiducial kappa, and a chi_eff normal broader than either truth (it must also
#: cover the closure's q-dependent width, sigma(q=0.3) = 0.23). It brings the
#: injection weights p_pop/p_draw at the truths close to constant, so the
#: selection Monte-Carlo variance is small for the same number of found
#: injections (review B item 1); the broad component keeps full support.
DRAW_PROXY_FRACTION = 0.8
DRAW_PROXY = {"population": "R0 fiducial (null truth) (m1, q) on a (ln m1, q) grid",
              "grid_ln_m1": 600, "grid_q": 400, "z_kappa": LVK_GWTC5_FIDUCIAL_MEDIANS["lamb"],
              "chi_mu": 0.03, "chi_sigma": 0.15}
#: Kish ESS of the canonical selection (1,566,112 rows) at the two truths
CANONICAL_SELECTION_ESS = {"closure_widthq": 78795.2, "null": 96499.0}

CALIBRATION_SEED = 20261001


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

def truth_model(kind: str):
    """(ModelSpec, sampled hyperparameters) of the generating population."""
    from ..grammar import apply_mutation
    from ..grammar.v2 import V2_MUTATION_TABLE, v2_root_model_spec
    from ..models.declarative import lvk_default_coordinates

    if kind not in MOCK_KINDS:
        raise ValueError(f"kind must be one of {MOCK_KINDS}; got {kind!r}")
    root = v2_root_model_spec()
    hp = lvk_default_coordinates(LVK_GWTC5_FIDUCIAL_MEDIANS, mmin=float(root.support["mmin"]))
    if kind == "null":
        hp.update(NULL_CHI)
        spec = root
    else:
        hp.update(CLOSURE_CHI)
        spec = apply_mutation(root, V2_MUTATION_TABLE[C2_MUTATION_ID])
    missing = set(spec.priors) - set(hp)
    extra = set(hp) - set(spec.priors)
    if missing or extra:
        raise ValueError(f"truth does not match the model priors: missing {missing}, extra {extra}")
    for name, value in hp.items():
        prior = spec.priors[name]
        lo, hi = float(prior.parameters["low"]), float(prior.parameters["high"])
        if not lo <= value <= hi:
            raise ValueError(f"truth {name}={value} outside its prior [{lo}, {hi}]")
    return spec, {k: float(v) for k, v in hp.items()}


def c2_model_spec():
    """The C2 child of R0 (the model whose slope the delta-PE profile scans)."""
    from ..grammar import apply_mutation
    from ..grammar.v2 import V2_MUTATION_TABLE, v2_root_model_spec

    return apply_mutation(v2_root_model_spec(), V2_MUTATION_TABLE[C2_MUTATION_ID])


class LogDensity:
    """Chunked, padded, jitted ln p_pop in the gwcat-v2 detector basis."""

    FIELDS = ("m1_detector", "q", "luminosity_distance", "chi_eff")

    def __init__(self, spec, chunk: int = 262_144):
        import jax

        jax.config.update("jax_enable_x64", True)
        from ..models import compile_model_spec

        self.model = compile_model_spec(spec)
        self.spec = spec
        self.chunk = int(chunk)
        self._fn = jax.jit(lambda s, h: self.model(s, h))

    def __call__(self, cols: Mapping[str, np.ndarray], hp: Mapping[str, float]) -> np.ndarray:
        import jax.numpy as jnp

        n = len(cols["q"])
        out = np.empty(n, dtype=float)
        hpj = {k: jnp.asarray(float(v)) for k, v in hp.items()}
        size = min(self.chunk, max(n, 1))
        for a in range(0, n, size):
            b = min(a + size, n)
            block = {}
            for k in self.FIELDS:
                arr = np.empty(size, dtype=float)
                arr[: b - a] = cols[k][a:b]
                arr[b - a:] = cols[k][a]
                block[k] = jnp.asarray(arr)
            out[a:b] = np.asarray(self._fn(block, hpj))[: b - a]
        return out


# ---------------------------------------------------------------------------
# Cosmology (NumPy mirror of models.cosmology.FlatLambdaCDM, same grids)
# ---------------------------------------------------------------------------

class NumpyCosmology:
    C_KM_S = 299792.458

    def __init__(self, cosmo):
        self.H0, self.Om0 = float(cosmo.H0), float(cosmo.Om0)
        self.zg = np.asarray(cosmo._z_grid, dtype=float)
        self.dg = np.asarray(cosmo._dL_grid, dtype=float)

    def e(self, z):
        return np.sqrt(self.Om0 * (1.0 + z) ** 3 + (1.0 - self.Om0))

    def dL_of_z(self, z):
        return np.interp(z, self.zg, self.dg)

    def z_of_dL(self, d):
        d = np.asarray(d, dtype=float)
        z = np.interp(d, self.dg, self.zg)
        return np.where((d >= 0.0) & (d <= self.dg[-1]), z, np.nan)

    def ddL_dz(self, z):
        dc = self.dL_of_z(z) / (1.0 + z)
        return dc + (1.0 + z) * (self.C_KM_S / self.H0) / self.e(z)

    def dVc_dz(self, z):
        dc = self.dL_of_z(z) / (1.0 + z)
        return 4.0 * np.pi * (self.C_KM_S / self.H0) * dc ** 2 / self.e(z)


def default_cosmology():
    from ..grammar.v2 import v2_root_model_spec
    from ..models import compile_model_spec

    return NumpyCosmology(compile_model_spec(v2_root_model_spec()).cosmology)


# ---------------------------------------------------------------------------
# Injection draw distribution
# ---------------------------------------------------------------------------

def _truncnorm_sample(rng, n, mu, sigma, lo, hi):
    a, b = ndtr((lo - mu) / sigma), ndtr((hi - mu) / sigma)
    return mu + sigma * ndtri(a + (b - a) * rng.uniform(size=n))


def _truncnorm_logpdf(x, mu, sigma, lo, hi):
    z = (x - mu) / sigma
    norm = ndtr((hi - mu) / sigma) - ndtr((lo - mu) / sigma)
    out = -0.5 * z * z - math.log(sigma * math.sqrt(2.0 * math.pi) * norm)
    return np.where((x >= lo) & (x <= hi), out, -np.inf)


class _GridZ:
    """z on [0, DRAW_ZMAX] with density ∝ f(z), sampled and evaluated exactly.

    The CDF is tabulated by the trapezoid rule on a fine grid and inverted by
    linear interpolation, so the sampled density is piecewise constant
    between grid nodes; :meth:`log_pdf` returns exactly that density.
    """

    def __init__(self, f, n: int = 40001):
        z = np.linspace(0.0, DRAW_ZMAX, n)
        pz = f(z)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pz[1:] + pz[:-1]) * np.diff(z))])
        self.z, self.cdf = z, cdf / cdf[-1]
        self._log_dens = np.log(np.diff(self.cdf) / np.diff(z))

    def sample(self, rng, n):
        return np.interp(rng.uniform(size=n), self.cdf, self.z)

    def log_pdf(self, z):
        k = np.clip(np.searchsorted(self.z, z, side="right") - 1, 0, len(self._log_dens) - 1)
        inside = (z > 0.0) & (z <= DRAW_ZMAX)
        return np.where(inside, self._log_dens[k], -np.inf)


class DrawDistribution:
    """Exact source-frame draw density, a broad + population-proxy mixture.

    broad (1 - f): m1 log-uniform on [3, 300], q uniform on [0.001, 1],
    chi_eff uniform on [-1, 1], z ∝ dVc/dz / (1 + z); proxy (f): (m1, q) from
    the R0 fiducial density tabulated on a (ln m1, q) grid (uniform within a
    cell), chi_eff truncated normal, z ∝ dVc/dz (1 + z)^(kappa - 1). The
    density is evaluated in the detector basis (m1_det, q, d_L, chi_eff) with
    the exact Jacobian 1 / ((1 + z) d d_L / dz).
    """

    def __init__(self, cosmo: NumpyCosmology, *, proxy_fraction: float = DRAW_PROXY_FRACTION):
        self.cosmo = cosmo
        self.f_p = float(proxy_fraction)
        kappa = float(DRAW_PROXY["z_kappa"])
        self._zb = _GridZ(lambda z: cosmo.dVc_dz(z) / (1.0 + z))
        self._zp = _GridZ(lambda z: cosmo.dVc_dz(z) * (1.0 + z) ** (kappa - 1.0))
        lo, hi = math.log(DRAW_M1[0]), math.log(DRAW_M1[1])
        self._lnm_bounds = (lo, hi)
        self._build_mq_grid()

    def _build_mq_grid(self):
        spec, hp = truth_model("null")
        nm, nq = int(DRAW_PROXY["grid_ln_m1"]), int(DRAW_PROXY["grid_q"])
        lo, hi = self._lnm_bounds
        self._dlnm = (hi - lo) / nm
        self._dq = (1.0 - DRAW_Q_LO) / nq
        lnm_c = lo + (np.arange(nm) + 0.5) * self._dlnm
        q_c = DRAW_Q_LO + (np.arange(nq) + 0.5) * self._dq
        L, Q = np.meshgrid(lnm_c, q_c, indexing="ij")
        z0, chi0 = 0.5, float(hp["chi_mu"])
        cols = {"m1_detector": np.exp(L.ravel()) * (1.0 + z0), "q": Q.ravel(),
                "luminosity_distance": np.full(L.size, float(self.cosmo.dL_of_z(z0))),
                "chi_eff": np.full(L.size, chi0)}
        # p(m1, q) up to a constant (the model factorises; z and chi_eff fixed)
        logp = LogDensity(spec)(cols, hp).reshape(nm, nq)
        log_cell = np.where(np.isfinite(logp), logp + L, -np.inf)  # x m1 (cell area in ln m1)
        log_cell = log_cell - logsumexp(log_cell)
        self._log_cell = log_cell
        self._cell_cdf = np.cumsum(np.exp(log_cell).ravel())
        self._cell_cdf /= self._cell_cdf[-1]

    def support(self) -> dict[str, float]:
        return {"m1_source_min": DRAW_M1[0], "m1_source_max": DRAW_M1[1], "q_min": DRAW_Q_LO,
                "z_max": DRAW_ZMAX, "chi_eff_min": -1.0, "chi_eff_max": 1.0}

    def sample(self, rng, n: int) -> dict[str, np.ndarray]:
        lo, hi = self._lnm_bounds
        proxy = rng.uniform(size=n) < self.f_p
        n_p = int(proxy.sum())
        n_b = n - n_p
        lnm = np.empty(n)
        q = np.empty(n)
        chi = np.empty(n)
        z = np.empty(n)
        lnm[~proxy] = rng.uniform(lo, hi, n_b)
        q[~proxy] = rng.uniform(DRAW_Q_LO, 1.0, n_b)
        chi[~proxy] = rng.uniform(-1.0, 1.0, n_b)
        z[~proxy] = self._zb.sample(rng, n_b)
        cell = np.minimum(np.searchsorted(self._cell_cdf, rng.uniform(size=n_p), side="right"),
                          self._cell_cdf.size - 1)
        i, j = np.divmod(cell, self._log_cell.shape[1])
        lnm[proxy] = lo + (i + rng.uniform(size=n_p)) * self._dlnm
        q[proxy] = DRAW_Q_LO + (j + rng.uniform(size=n_p)) * self._dq
        chi[proxy] = _truncnorm_sample(rng, n_p, DRAW_PROXY["chi_mu"], DRAW_PROXY["chi_sigma"], -1.0, 1.0)
        z[proxy] = self._zp.sample(rng, n_p)
        return {"m1_source": np.exp(lnm), "q": q, "z": z, "chi_eff": chi}

    def log_density_source(self, m1, q, z, chi):
        """ln p_draw(m1_source, q, z, chi_eff)."""
        lo, hi = self._lnm_bounds
        m1, q, z, chi = (np.asarray(x, dtype=float) for x in (m1, q, z, chi))
        lnm = np.log(m1)
        in_m = (lnm >= lo) & (lnm <= hi)
        in_q = (q >= DRAW_Q_LO) & (q <= 1.0)
        in_c = (chi >= -1.0) & (chi <= 1.0)
        in_z = (z > 0.0) & (z <= DRAW_ZMAX)
        log_b = (-lnm - math.log(hi - lo) - math.log(1.0 - DRAW_Q_LO) - math.log(2.0)
                 + self._zb.log_pdf(z))
        nm, nq = self._log_cell.shape
        with np.errstate(invalid="ignore"):
            fi = np.nan_to_num(np.floor((lnm - lo) / self._dlnm), nan=0.0, posinf=0.0, neginf=0.0)
            fj = np.nan_to_num(np.floor((q - DRAW_Q_LO) / self._dq), nan=0.0, posinf=0.0, neginf=0.0)
        i = np.clip(fi, 0, nm - 1).astype(np.int64)
        j = np.clip(fj, 0, nq - 1).astype(np.int64)
        with np.errstate(invalid="ignore"):
            log_p = (self._log_cell[i, j] - math.log(self._dlnm) - math.log(self._dq) - lnm
                     + _truncnorm_logpdf(chi, DRAW_PROXY["chi_mu"], DRAW_PROXY["chi_sigma"], -1.0, 1.0)
                     + self._zp.log_pdf(z))
        log_mix = np.logaddexp(math.log1p(-self.f_p) + log_b, math.log(self.f_p) + log_p)
        ok = in_m & in_q & in_c & in_z
        return np.where(ok, log_mix, -np.inf)

    def log_density_detector(self, m1, q, z, chi):
        """ln p_draw in dm1_detector dq dd_L dchi_eff (the gwcat-v2 basis)."""
        safe_z = np.where(z > 0, z, 0.5)
        jac = -np.log1p(safe_z) - np.log(self.cosmo.ddL_dz(safe_z))
        return self.log_density_source(m1, q, z, chi) + jac


# ---------------------------------------------------------------------------
# Detection statistic (the same for injections, events and the PE likelihood)
# ---------------------------------------------------------------------------

def projection_factor(rng, n: int) -> np.ndarray:
    """Single-interferometer sky/orientation projection Theta in [0, 1].

    Theta^2 = F+^2 (1 + cos^2 i)^2 / 4 + Fx^2 cos^2 i with isotropic sky,
    polarisation and inclination (the Finn & Chernoff 1993 Theta / 4).
    """
    cth = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    psi = rng.uniform(0.0, np.pi, n)
    ci = rng.uniform(-1.0, 1.0, n)
    a = 0.5 * (1.0 + cth ** 2) * np.cos(2.0 * phi)
    b = cth * np.sin(2.0 * phi)
    fp = a * np.cos(2.0 * psi) - b * np.sin(2.0 * psi)
    fx = a * np.sin(2.0 * psi) + b * np.cos(2.0 * psi)
    return np.sqrt(fp ** 2 * (1.0 + ci ** 2) ** 2 / 4.0 + fx ** 2 * ci ** 2)


def rolloff_factor(m_tot_det, m_rolloff, power: float = 5.0 / 6.0, sharpness: float = 1.0):
    """R(M) = (1 + (M_det / M_ro)^s)^(-p / s) <= 1 (decreasing in M_det)."""
    return (1.0 + (m_tot_det / m_rolloff) ** float(sharpness)) ** (-float(power) / float(sharpness))


def snr_unit(m1_det, q, d_l, chi, m_rolloff, power: float = 5.0 / 6.0, sharpness: float = 1.0):
    """g(theta) at unit amplitude and Theta = 1.

    (Mc_det / 10)^(5/6) (1000 Mpc / d_L) (1 + k chi_eff) R(M_det): the inspiral
    chirp-mass/distance scaling, louder for aligned spins, and a roll-off
    R = (1 + (M_det / M_ro)^s)^(-p/s) above the detector-frame total mass M_ro
    (merger leaving the band). With p > 5/6 the SNR *falls* for M_det >> M_ro,
    as for real detectors; the mocks use s = 3 (:data:`MASS_ROLLOFF_SHARPNESS`)
    with M_ro and p calibrated to the canonical selection (:func:`calibrate`).
    """
    mc = m1_det * q ** 0.6 / (1.0 + q) ** 0.2
    m_tot = m1_det * (1.0 + q)
    return ((mc / 10.0) ** (5.0 / 6.0) * (1000.0 / d_l) * (1.0 + SPIN_SNR_SLOPE * chi)
            * rolloff_factor(m_tot, m_rolloff, power, sharpness))


@dataclass(frozen=True)
class Calibration:
    """Per-run SNR amplitudes A_k and run times T_k, plus the PE width scales."""

    amplitude: tuple[float, ...]
    run_time_yr: tuple[float, ...]
    width_scale: tuple[float, float, float]
    mass_rolloff: float
    labels: tuple[str, ...] = RUN_LABELS
    report: dict = field(default_factory=dict, compare=False)
    #: exponent p and sharpness s of the mass roll-off (:func:`rolloff_factor`)
    mass_rolloff_power: float = 5.0 / 6.0
    mass_rolloff_sharpness: float = 1.0

    @property
    def run_probability(self) -> np.ndarray:
        t = np.asarray(self.run_time_yr, dtype=float)
        return t / t.sum()

    @property
    def total_time_yr(self) -> float:
        return float(np.sum(self.run_time_yr))

    def to_dict(self) -> dict:
        return {"labels": list(self.labels), "amplitude": list(self.amplitude),
                "run_time_yr": list(self.run_time_yr), "width_scale": list(self.width_scale),
                "mass_rolloff": float(self.mass_rolloff),
                "mass_rolloff_power": float(self.mass_rolloff_power),
                "mass_rolloff_sharpness": float(self.mass_rolloff_sharpness), "report": self.report}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Calibration":
        return cls(tuple(float(x) for x in d["amplitude"]), tuple(float(x) for x in d["run_time_yr"]),
                   tuple(float(x) for x in d["width_scale"]), float(d["mass_rolloff"]),
                   tuple(d.get("labels", RUN_LABELS)), dict(d.get("report", {})),
                   float(d.get("mass_rolloff_power", 5.0 / 6.0)),
                   float(d.get("mass_rolloff_sharpness", 1.0)))


def simulate_detections(draw: DrawDistribution, cal: Calibration, rng, n_draw: int,
                        *, chunk: int = 4_000_000) -> dict[str, np.ndarray]:
    """THE detection machinery: draw n_draw systems, keep those with rho_obs > rho_thr.

    Each system gets one run label (probability T_k / T), one projection
    Theta and one noise draw n; rho_obs = A_k Theta g(theta) + n. Returns the
    detected rows (source + detector coordinates, run index, Theta, rho_opt,
    rho_obs and ln p_draw in the detector basis).
    """
    cosmo = draw.cosmo
    amp = np.asarray(cal.amplitude, dtype=float)
    p_run = cal.run_probability
    out: dict[str, list] = {k: [] for k in ("m1_source", "q", "z", "chi_eff", "m1_detector",
                                            "luminosity_distance", "run_index", "theta",
                                            "rho_opt", "rho_obs", "log_draw_density")}
    done = 0
    while done < n_draw:
        n = min(chunk, n_draw - done)
        s = draw.sample(rng, n)
        run = rng.choice(len(amp), size=n, p=p_run)
        theta = projection_factor(rng, n)
        noise = rng.standard_normal(n)
        d_l = cosmo.dL_of_z(s["z"])
        m1_det = s["m1_source"] * (1.0 + s["z"])
        rho_opt = amp[run] * theta * snr_unit(m1_det, s["q"], d_l, s["chi_eff"], cal.mass_rolloff,
                                              cal.mass_rolloff_power, cal.mass_rolloff_sharpness)
        rho_obs = rho_opt + noise
        found = rho_obs > SNR_THRESHOLD
        out["m1_source"].append(s["m1_source"][found])
        out["q"].append(s["q"][found])
        out["z"].append(s["z"][found])
        out["chi_eff"].append(s["chi_eff"][found])
        out["m1_detector"].append(m1_det[found])
        out["luminosity_distance"].append(d_l[found])
        out["run_index"].append(run[found])
        out["theta"].append(theta[found])
        out["rho_opt"].append(rho_opt[found])
        out["rho_obs"].append(rho_obs[found])
        out["log_draw_density"].append(draw.log_density_detector(
            s["m1_source"][found], s["q"][found], s["z"][found], s["chi_eff"][found]))
        done += n
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Synthetic PE
# ---------------------------------------------------------------------------
#
# PE data d = (rho_obs, x_obs) with x_obs = (ln m1_det, q, chi_eff) + N(0, Sigma(rho_obs)).
# Distance information enters only through the amplitude datum rho_obs (with
# the projection Theta marginalised), as in real GW PE. The posterior under the
# box prior pi ∝ m1_det d_L^2 then factorises exactly (derivation in
# :func:`sample_pe`).

def pe_covariance(rho_obs: float, width_scale) -> np.ndarray:
    """Sigma(rho_obs) of x_obs: widths c_j rho_thr / rho_obs, constant correlation."""
    s = SNR_THRESHOLD / max(float(rho_obs), RHO_WIDTH_FLOOR)
    sig = np.asarray(width_scale, dtype=float) * s
    corr = np.asarray(PE_CORRELATION, dtype=float)
    return corr * np.outer(sig, sig)


def x_of(m1_det, q, chi) -> np.ndarray:
    return np.stack([np.log(m1_det), q, chi], axis=-1)


def pe_log_prior_basis(m1_det, d_l, box: Mapping | None = None) -> np.ndarray:
    """ln pi_PE in dm1_det dq dd_L dchi_eff (the canonical log_ref_density convention)."""
    box = PE_BOX if box is None else box
    (m_lo, m_hi), (q_lo, q_hi) = box["m1_detector"], box["q"]
    (d_lo, d_hi), (c_lo, c_hi) = box["luminosity_distance"], box["chi_eff"]
    log_norm = (math.log(0.5 * (m_hi ** 2 - m_lo ** 2)) + math.log(q_hi - q_lo)
                + math.log((d_hi ** 3 - d_lo ** 3) / 3.0) + math.log(c_hi - c_lo))
    inside = ((m1_det >= m_lo) & (m1_det <= m_hi) & (d_l >= d_lo) & (d_l <= d_hi))
    return np.where(inside, np.log(m1_det) + 2.0 * np.log(d_l) - log_norm, -np.inf)


def observe(rng, x_true: np.ndarray, rho_obs: float, width_scale) -> np.ndarray:
    """x_obs = x(theta) + N(0, Sigma(rho_obs))."""
    cov = pe_covariance(rho_obs, width_scale)
    return x_true + np.linalg.cholesky(cov) @ rng.standard_normal(3)


def _log_c(m1_det, q, chi, amplitude, m_rolloff, power: float = 5.0 / 6.0, sharpness: float = 1.0):
    """ln[A d_L g(theta)], so that rho_opt = Theta exp(c) / d_L (see :func:`snr_unit`)."""
    return np.log(amplitude * snr_unit(m1_det, q, 1000.0, chi, m_rolloff, power, sharpness)) + math.log(1000.0)


def rho_lower_cut(rho_obs: float) -> float:
    return max(float(rho_obs) - 8.0, float(rho_obs) / 3.0)


def _sample_rho_opt(rng, rho_obs: float, n: int, grid_size: int = 20001) -> np.ndarray:
    """r ∝ r^-4 N(rho_obs; r, 1) on [rho_lower_cut, rho_obs + 12] by fine-grid inverse CDF."""
    lo, hi = rho_lower_cut(rho_obs), float(rho_obs) + 12.0
    r = np.linspace(lo, hi, grid_size)
    logf = -4.0 * np.log(r) - 0.5 * (rho_obs - r) ** 2
    f = np.exp(logf - logf.max())
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    return np.interp(rng.uniform(size=n) * cdf[-1], cdf, r)


def _sample_theta_cubed(rng, n: int) -> np.ndarray:
    """Theta ∝ Theta^3 p(Theta) (rejection from p(Theta), acceptance Theta^3)."""
    out, got = [], 0
    while got < n:
        t = projection_factor(rng, max(4 * (n - got), 1024) * 4)
        t = t[rng.uniform(size=t.size) < t ** 3]
        out.append(t)
        got += t.size
    return np.concatenate(out)[:n]


def sample_pe(rng, x_obs: np.ndarray, rho_obs: float, amplitude: float, n_samples: int,
              width_scale, m_rolloff: float, *, box: Mapping | None = None, batch: int = 32_768,
              max_proposals: int = 200_000_000, rolloff_power: float = 5.0 / 6.0,
              rolloff_sharpness: float = 1.0):
    """Exact posterior draws of (m1_det, q, d_L, chi_eff) given d = (rho_obs, x_obs).

    With y_r = (ln m1_det, q, chi_eff), y2 = ln d_L, c(y_r) as in :func:`_log_c`
    and rho_opt = Theta exp(c - y2), the posterior is ::

        p ∝ N(x_obs; y_r, Sigma) e^{2 y0} e^{3 y2} p(Theta) N(rho_obs; Theta e^{c - y2}, 1) 1_box

    (e^{2 y0} e^{3 y2} is the prior m1_det d_L^2 times the Jacobian m1_det d_L).
    Substituting r = Theta e^{c - y2} (i.e. y2 = c + ln Theta - ln r) gives
    e^{3 y2} = e^{3c} Theta^3 r^-3 and dy2 = dr / r, so without the box ::

        p ∝ [N(x_obs; y_r, Sigma) e^{2 y0} e^{3 c(y_r)}] [Theta^3 p(Theta)] [r^-4 N(rho_obs; r, 1)]

    three independent factors. e^{3c} ∝ e^{2.5 y0} h(q) (1 + k chi)^3 R(M)^3
    with h(q) = q^1.5 (1 + q)^-0.5 <= h(1) and R(M) = :func:`rolloff_factor`
    <= 1, so y_r is drawn from N(x_obs + Sigma (4.5, 0, 0), Sigma) and accepted
    with probability [h(q) / h(1)] [(1 + k chi) / (1 + k)]^3 R(M)^3; Theta by
    rejection; r on a fine
    grid (truncated below at :func:`rho_lower_cut`, neglected mass < 1e-8).
    The box is applied by rejecting joint draws outside it. Returns
    (samples dict, acceptance fraction).
    """
    box = PE_BOX if box is None else box
    cov = pe_covariance(rho_obs, width_scale)
    chol = np.linalg.cholesky(cov)
    mean = np.asarray(x_obs, dtype=float) + cov @ np.array([4.5, 0.0, 0.0])
    lo = np.array([math.log(box["m1_detector"][0]), box["q"][0], box["chi_eff"][0]])
    hi = np.array([math.log(box["m1_detector"][1]), box["q"][1], box["chi_eff"][1]])
    d_lo, d_hi = box["luminosity_distance"]
    h1 = 2.0 ** -0.5
    kept = {k: [] for k in ("m1_detector", "q", "luminosity_distance", "chi_eff")}
    n_kept = n_prop = 0
    while n_kept < n_samples:
        if n_prop >= max_proposals:
            raise RuntimeError(f"PE sampler: {n_kept} < {n_samples} accepted after {n_prop} proposals")
        y = mean + rng.standard_normal((batch, 3)) @ chol.T
        n_prop += batch
        y = y[np.all((y >= lo) & (y <= hi), axis=1)]
        q, chi = y[:, 1], y[:, 2]
        acc = ((q ** 1.5 * (1.0 + q) ** -0.5 / h1)
               * ((1.0 + SPIN_SNR_SLOPE * chi) / (1.0 + SPIN_SNR_SLOPE)) ** 3
               * rolloff_factor(np.exp(y[:, 0]) * (1.0 + q), m_rolloff, rolloff_power, rolloff_sharpness) ** 3)
        y = y[rng.uniform(size=len(y)) < acc]
        if len(y) == 0:
            continue
        m1 = np.exp(y[:, 0])
        theta = _sample_theta_cubed(rng, len(y))
        r = _sample_rho_opt(rng, rho_obs, len(y))
        d_l = np.exp(_log_c(m1, y[:, 1], y[:, 2], amplitude, m_rolloff, rolloff_power, rolloff_sharpness)) * theta / r
        ok = (d_l >= d_lo) & (d_l <= d_hi)
        kept["m1_detector"].append(m1[ok])
        kept["q"].append(y[ok, 1])
        kept["chi_eff"].append(y[ok, 2])
        kept["luminosity_distance"].append(d_l[ok])
        n_kept += int(ok.sum())
    out = {k: np.concatenate(v)[:n_samples] for k, v in kept.items()}
    return out, n_kept / n_prop


def pe_log_likelihood_unmarginalised(x_obs, rho_obs, amplitude, width_scale, m_rolloff, *,
                                     m1_det, q, d_l, chi, theta, rolloff_power: float = 5.0 / 6.0,
                                     rolloff_sharpness: float = 1.0):
    """ln L(d | theta, Theta) up to a constant (for tests)."""
    cov = pe_covariance(rho_obs, width_scale)
    r = x_of(m1_det, q, chi) - x_obs
    quad = np.einsum("...i,ij,...j->...", r, np.linalg.inv(cov), r)
    rho = theta * np.exp(_log_c(m1_det, q, chi, amplitude, m_rolloff, rolloff_power, rolloff_sharpness)) / d_l
    return -0.5 * quad - 0.5 * (rho_obs - rho) ** 2


# ---------------------------------------------------------------------------
# Calibration to the canonical selection and the real PE widths
# ---------------------------------------------------------------------------

def weighted_quantiles(x, w, qs=(0.1, 0.5, 0.9)):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c = c / c[-1]
    return [float(np.interp(q, c, x[o])) for q in qs]


#: canonical pooled (all runs) detected R0 (null truth) mass statistics: the
#: roll-off calibration targets (q90 for M_ro, P(m1 > 100) for the power p) and
#: reported tail checks. Recompute with :func:`canonical_pooled_targets`.
CANONICAL_POOLED_TARGETS: dict[str, float] = {
    "m1_source_q90": 49.303657647743265,
    "m1_source_q99": 83.68457794189453,
    "p_m1_source_gt_100": 0.0038052795349319706,
    "p_m1_source_gt_200": 4.077543597947834e-05,
    "p_mtot_detector_gt_1000": 0.0,
}

#: sigma^2_lnL of the canonical (real) pair at the two truths with the
#: pipeline's estimator (:func:`taper_variance_at_truth`); recompute with
#: :func:`canonical_taper_variance`.
CANONICAL_TAPER_VARIANCE: dict[str, dict[str, float]] = {
    "closure_widthq": {"events": 0.2722, "selection": 0.8513, "total": 1.1235},
    "null": {"events": 0.2431, "selection": 0.6951, "total": 0.9382},
}


class _SortedQuantile:
    """Weighted quantiles of one fixed column under many masks (presorted, O(n) each)."""

    def __init__(self, x):
        self.order = np.argsort(x)
        self.xs = np.asarray(x)[self.order]

    def __call__(self, w, q):
        c = np.cumsum(np.asarray(w)[self.order])
        return float(self.xs[min(int(np.searchsorted(c, q * c[-1])), len(c) - 1)])


def _calibration_pool(draw, rng, n):
    s = draw.sample(rng, n)
    d_l = draw.cosmo.dL_of_z(s["z"])
    m1_det = s["m1_source"] * (1.0 + s["z"])
    mc = m1_det * s["q"] ** 0.6 / (1.0 + s["q"]) ** 0.2
    s.update(m1_detector=m1_det, luminosity_distance=d_l,
             base=projection_factor(rng, n) * snr_unit(m1_det, s["q"], d_l, s["chi_eff"], np.inf),
             m_tot=m1_det * (1.0 + s["q"]), noise=rng.standard_normal(n),
             log_draw_density=draw.log_density_detector(s["m1_source"], s["q"], s["z"], s["chi_eff"]))
    del mc
    return s


def _calibrate_amplitudes(pool, w, m_rolloff, targets, zq, power: float = 5.0 / 6.0, start=None,
                          *, steps: int = 40, half_width: float | None = None,
                          sharpness: float = 1.0):
    """Per-run A_k matching the canonical median detected redshift (bisection in ln A)."""
    unit = pool["base"] * rolloff_factor(pool["m_tot"], m_rolloff, power, sharpness)
    amps, dets = [], []
    cold = (0.0, math.log(5.0e5))
    for k, label in enumerate(RUN_LABELS):
        target = float(targets[label]["z"][1])
        warm = start is not None and half_width is not None
        lo, hi = (math.log(start[k]) - half_width, math.log(start[k]) + half_width) if warm else cold
        n = steps
        while True:
            lo0, hi0 = lo, hi
            for _ in range(n):
                mid = 0.5 * (lo + hi)
                det = math.exp(mid) * unit + pool["noise"] > SNR_THRESHOLD
                if zq(w * det, 0.5) < target:
                    lo = mid
                else:
                    hi = mid
            # a warm bracket that the solution left: redo with the cold bracket
            if warm and (lo == lo0 or hi == hi0):
                warm, (lo, hi), n = False, cold, 40
                continue
            break
        a = math.exp(0.5 * (lo + hi))
        amps.append(a)
        dets.append(a * unit + pool["noise"] > SNR_THRESHOLD)
    return amps, dets


def _pooled_tail_stats(pool, mix, mq) -> dict:
    """Detected-population mass statistics of the run mixture ``mix`` (weights)."""
    tot = float(np.sum(mix))
    return {
        "m1_source_q90": mq(mix, 0.9), "m1_source_q99": mq(mix, 0.99),
        "p_m1_source_gt_100": float(np.sum(mix[pool["m1_source"] > 100.0]) / tot),
        "p_m1_source_gt_200": float(np.sum(mix[pool["m1_source"] > 200.0]) / tot),
        "p_mtot_detector_gt_1000": float(np.sum(mix[pool["m_tot"] > 1000.0]) / tot),
    }


def calibrate(draw: DrawDistribution, *, n_pool: int = 3_000_000, n_width_events: int = 200,
              n_width_samples: int = 1024, width_iterations: int = 5, seed: int = CALIBRATION_SEED,
              targets: Mapping[str, Mapping] = CANONICAL_RUN_TARGETS,
              pooled_targets: Mapping[str, float] | None = None, power_bounds=(5.0 / 6.0, 8.0),
              power_steps: int = 8, rolloff_steps: int = 16, sharpness: float | None = None,
              log=print) -> Calibration:
    """Calibrate the mock detection to the canonical selection and the PE to the real widths.

    * per run k: A_k so that the detected R0 population has the canonical
      median redshift; T_k so that run k has the canonical share of the
      detections (share_k ∝ T_k beta_k);
    * globally, the mass roll-off (1 + (M_det / M_ro)^s)^(-p/s) with s =
      :data:`MASS_ROLLOFF_SHARPNESS`: for each p, M_ro so
      that the pooled detected m1_source 90 % quantile is the canonical one
      (A_k re-fit at every M_ro); p so that the pooled detected fraction with
      m1_source > 100 is the canonical one (review B item 4: with p = 5/6 the
      SNR never falls with mass and heavy systems were over-detected);
    * PE: the x_obs width scales c_j so that the median per-event posterior
      std of (ln m1_det, q, chi_eff) is the real catalog's.

    Uses the null truth (R0) and a fixed seed, so both mocks share one
    calibration; common random numbers throughout (monotone bisections,
    fixed-point iteration for c_j).
    """
    pooled_targets = dict(CANONICAL_POOLED_TARGETS if pooled_targets is None else pooled_targets)
    sharpness = MASS_ROLLOFF_SHARPNESS if sharpness is None else float(sharpness)
    spec, hp = truth_model("null")
    logp = LogDensity(spec)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    pool = _calibration_pool(draw, rng, n_pool)
    lw = logp(pool, hp) - pool["log_draw_density"]
    w = np.exp(lw - np.max(lw[np.isfinite(lw)]))
    w[~np.isfinite(w)] = 0.0
    zq = _SortedQuantile(pool["z"])
    mq = _SortedQuantile(pool["m1_source"])
    share = np.array([float(targets[k]["share"]) for k in RUN_LABELS])
    state = {"amps": None}

    def pooled(m_ro, power):
        start = state["amps"]
        amps, dets = _calibrate_amplitudes(pool, w, m_ro, targets, zq, power, start,
                                           steps=40 if start is None else 18,
                                           half_width=None if start is None else 1.0,
                                           sharpness=sharpness)
        state["amps"] = amps
        betas = np.array([np.sum(w * d) for d in dets])
        mix = sum(sh / b * (w * d) for sh, b, d in zip(share, betas, dets))
        return _pooled_tail_stats(pool, mix, mq), amps, dets, betas

    def fit_rolloff(power):
        lo, hi = math.log(50.0), math.log(2.0e4)
        for _ in range(rolloff_steps):
            mid = 0.5 * (lo + hi)
            if pooled(math.exp(mid), power)[0]["m1_source_q90"] < pooled_targets["m1_source_q90"]:
                lo = mid
            else:
                hi = mid
        m_ro = math.exp(0.5 * (lo + hi))
        return m_ro, pooled(m_ro, power)

    target_tail = float(pooled_targets["p_m1_source_gt_100"])
    p_lo, p_hi = (float(x) for x in power_bounds)
    history = []
    m_ro, res = fit_rolloff(p_lo)
    history.append({"power": p_lo, "mass_rolloff": m_ro, **res[0]})
    if res[0]["p_m1_source_gt_100"] <= target_tail:
        power = p_lo
    else:
        m_hi, res_hi = fit_rolloff(p_hi)
        history.append({"power": p_hi, "mass_rolloff": m_hi, **res_hi[0]})
        if res_hi[0]["p_m1_source_gt_100"] > target_tail:
            power, m_ro, res = p_hi, m_hi, res_hi
        else:
            for _ in range(power_steps):
                mid = 0.5 * (p_lo + p_hi)
                m_mid, res_mid = fit_rolloff(mid)
                history.append({"power": mid, "mass_rolloff": m_mid, **res_mid[0]})
                if res_mid[0]["p_m1_source_gt_100"] > target_tail:
                    p_lo = mid
                else:
                    p_hi = mid
            power = 0.5 * (p_lo + p_hi)
            m_ro, res = fit_rolloff(power)
        log(f"calibration: roll-off power {power:.4f} M_ro {m_ro:.1f}")
    stats, amps, dets, betas = res
    t = share / betas
    t = t * T_TOTAL_YR / t.sum()
    report = {"mass_rolloff": m_ro, "mass_rolloff_power": power, "mass_rolloff_sharpness": sharpness,
              "pooled_detected_mass": {"achieved": stats, "target": pooled_targets},
              "pooled_m1_source_q90": {"achieved": stats["m1_source_q90"],
                                       "target": pooled_targets["m1_source_q90"]},
              "rolloff_search_history": history}
    for label, a, d, tk in zip(RUN_LABELS, amps, dets, t):
        report[label] = {
            "amplitude": a, "run_time_yr": float(tk),
            "achieved": {k: weighted_quantiles(pool[k][d], w[d]) for k in ("z", "m1_source", "q", "chi_eff")},
            "target": {k: list(targets[label][k]) for k in ("z", "m1_source", "q", "chi_eff")},
            "pool_detected_ess": float(np.sum(w * d) ** 2 / np.sum((w * d) ** 2)),
        }
    cal = Calibration(tuple(amps), tuple(float(x) for x in t), X_WIDTH_TARGETS, m_ro,
                      mass_rolloff_power=power, mass_rolloff_sharpness=sharpness)
    det_frac = float(sum(p * np.mean(d) for p, d in zip(cal.run_probability, dets)))
    log(f"calibration: M_ro {m_ro:.2f} power {power:.4f} amplitudes {np.round(amps, 3).tolist()} run times "
        f"{np.round(t, 4).tolist()} draw detection fraction {det_frac:.5f} pooled {stats}")

    # -- PE widths: match the median per-event posterior std of the real catalog
    ev = _resample_detected(pool, w, cal, rng, n_width_events)
    eps = rng.standard_normal((n_width_events, 3))
    pe_seeds = rng.integers(0, 2 ** 63 - 1, n_width_events)
    box = pe_box(draw.cosmo)
    scale = np.asarray(X_WIDTH_TARGETS, dtype=float).copy()
    history = []
    for it in range(width_iterations + 1):
        stds, corrs, accs = [], [], []
        for i in range(n_width_events):
            cov = pe_covariance(ev["rho_obs"][i], scale)
            x_true = x_of(ev["m1_detector"][i], ev["q"][i], ev["chi_eff"][i])
            x_obs = x_true + np.linalg.cholesky(cov) @ eps[i]
            s, acc = sample_pe(np.random.default_rng(pe_seeds[i]), x_obs, ev["rho_obs"][i],
                               ev["amplitude"][i], n_width_samples, scale, m_ro, batch=8192, box=box,
                               rolloff_power=power, rolloff_sharpness=sharpness)
            ys = np.stack([np.log(s["m1_detector"]), s["q"], np.log(s["luminosity_distance"]), s["chi_eff"]], 1)
            stds.append(ys.std(axis=0))
            corrs.append(np.corrcoef(ys.T))
            accs.append(acc)
        achieved = np.median(np.asarray(stds), axis=0)
        history.append({"scale": scale.tolist(), "median_posterior_std": achieved.tolist(),
                        "median_acceptance": float(np.median(accs))})
        log(f"width iteration {it}: scale {np.round(scale, 4).tolist()} achieved {np.round(achieved, 4).tolist()}")
        if it == width_iterations:
            break
        scale = scale * np.asarray(X_WIDTH_TARGETS) / achieved[[0, 1, 3]]
    report["pe_widths"] = {
        "x_obs_coordinates": list(X_NAMES),
        "coordinates": list(REAL_PE_COORDINATES),
        "target_median_posterior_std": [REAL_PE_WIDTH_MEDIANS[k] for k in REAL_PE_COORDINATES],
        "achieved_median_posterior_std": history[-1]["median_posterior_std"],
        "achieved_median_posterior_correlation": np.median(np.asarray(corrs), axis=0).round(4).tolist(),
        "real_median_posterior_correlation": [list(r) for r in REAL_PE_CORRELATION],
        "note": "ln d_L is not calibrated: its width follows from the amplitude datum and the projection prior",
        "history": history, "n_events": n_width_events, "n_samples": n_width_samples,
    }
    report["draw_detection_fraction"] = det_frac
    report["seed"] = int(seed)
    report["n_pool"] = int(n_pool)
    report["truth"] = "null (R0 fiducial, chi_eff N(0.03, 0.09))"
    return Calibration(cal.amplitude, cal.run_time_yr, tuple(float(x) for x in history[-1]["scale"]),
                       m_ro, RUN_LABELS, report, power, sharpness)


def _resample_detected(pool, w, cal, rng, n):
    """Detected calibration systems (with the calibrated run mix), resampled by weight."""
    run = rng.choice(len(cal.amplitude), size=len(w), p=cal.run_probability)
    amp = np.asarray(cal.amplitude)[run]
    rho_obs = (amp * pool["base"] * rolloff_factor(pool["m_tot"], cal.mass_rolloff, cal.mass_rolloff_power,
                                                   cal.mass_rolloff_sharpness)
               + pool["noise"])
    det = rho_obs > SNR_THRESHOLD
    p = w * det
    idx = rng.choice(len(w), size=n, replace=False, p=p / p.sum())
    return {"m1_detector": pool["m1_detector"][idx], "q": pool["q"][idx],
            "luminosity_distance": pool["luminosity_distance"][idx], "chi_eff": pool["chi_eff"][idx],
            "rho_obs": rho_obs[idx], "amplitude": amp[idx]}


def canonical_run_targets(selection_path, index_path, *, log=print) -> dict:
    """Recompute :data:`CANONICAL_RUN_TARGETS` from the canonical selection (read-only)."""
    import h5py

    from ..data import SelectionCatalog

    spec, hp = truth_model("null")
    sel = SelectionCatalog.from_hdf5(selection_path)
    with h5py.File(index_path, "r") as h:
        runs = h["run_label"][:][h["retained_rows"][:]].astype(str)
    lw = LogDensity(spec)(sel.samples, hp) - sel.log_draw_density
    w = np.exp(lw - np.max(lw[np.isfinite(lw)]))
    w[~np.isfinite(w)] = 0.0
    out = {}
    for label in RUN_LABELS:
        m = runs == label
        out[label] = {"share": float(w[m].sum() / w.sum())}
        for k in ("z", "m1_source", "q", "chi_eff"):
            out[label][k] = weighted_quantiles(sel.samples[k][m], w[m])
    log(json.dumps(out))
    return out


def _truth_weights(samples, log_draw_density, kind: str) -> np.ndarray:
    spec, hp = truth_model(kind)
    lw = LogDensity(spec)(samples, hp) - log_draw_density
    w = np.exp(lw - np.max(lw[np.isfinite(lw)]))
    w[~np.isfinite(w)] = 0.0
    return w


def canonical_pooled_targets(selection_path, *, log=print) -> dict:
    """Recompute :data:`CANONICAL_POOLED_TARGETS` (pooled detected R0 mass statistics; read-only).

    The canonical selection is one estimator-ready campaign, so the found
    rows weighted by p_pop / p_draw at the null truth are the detected
    population.
    """
    from ..data import SelectionCatalog

    sel = SelectionCatalog.from_hdf5(selection_path)
    w = _truth_weights(sel.samples, sel.log_draw_density, "null")
    m1 = np.asarray(sel.samples["m1_source"])
    pool = {"m1_source": m1, "m_tot": np.asarray(sel.samples["m1_detector"]) * (1.0 + np.asarray(sel.samples["q"]))}
    out = _pooled_tail_stats(pool, w, _SortedQuantile(m1))
    log(json.dumps(out))
    return out


def taper_variance_at_truth(posterior, selection, kind: str) -> dict:
    """sigma^2_lnL at the truth with the pipeline's estimator (NumPy backend, v2 sharp cut).

    Returns the per-event sum, the selection term N^2 Var[ln xi], their total
    and whether the v2 cut (sigma^2 <= 1) keeps the truth.
    """
    from ..hbi import HBIConfig
    from ..hbi.numpy_backend import shape_log_likelihood
    from ..inference.v2_numerics import v2_variance_taper
    from ..models import compile_model_spec

    spec, hp = truth_model(kind)
    taper = v2_variance_taper()
    result = shape_log_likelihood(posterior, selection, compile_model_spec(spec), hp,
                                  config=HBIConfig(selection_chunk_size=None, variance_taper=taper))
    events = float(result.terms.variance.event_variance)
    total = float(result.taper_variance)
    return {"events": events, "selection": total - events, "total": total,
            "threshold": float(taper.threshold), "kept_by_cut": bool(total <= taper.threshold)}


def canonical_taper_variance(pe_path, selection_path, *, log=print) -> dict:
    """Recompute :data:`CANONICAL_TAPER_VARIANCE` on the canonical pair (read-only)."""
    from ..data import PosteriorCatalog, SelectionCatalog

    pe = PosteriorCatalog.from_hdf5(pe_path)
    sel = SelectionCatalog.from_hdf5(selection_path)
    out = {kind: taper_variance_at_truth(pe, sel, kind) for kind in MOCK_KINDS}
    log(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def selection_log_xi(logp: LogDensity, selection, hp) -> float:
    """ln xi with the pipeline's raw-draw estimator, ln(T/N) + logsumexp(ln p_pop - ln p_draw)."""
    from ..hbi.common import selection_log_factors

    lp = logp(selection.samples, hp)
    return float(logsumexp(lp - selection.log_draw_density + selection_log_factors(selection)))


def profile_summary(grid, lnl, truth) -> dict:
    grid, lnl = np.asarray(grid, float), np.asarray(lnl, float)
    i = int(np.nanargmax(lnl))
    mle = float(grid[i])
    if 0 < i < len(grid) - 1:
        c = np.polyfit(grid[i - 1:i + 2], lnl[i - 1:i + 2], 2)
        if c[0] < 0:
            mle = float(-c[1] / (2 * c[0]))
    dl = lnl[i] - lnl
    inside = grid[dl <= 2.0]
    lo, hi = float(inside.min()), float(inside.max())
    t_ln = float(np.interp(truth, grid, lnl))
    return {"grid": grid.tolist(), "lnL": lnl.tolist(), "mle": mle, "max_lnL": float(lnl[i]),
            "interval_dlnL_le_2": [lo, hi], "truth": float(truth),
            "delta_lnL_at_truth": float(lnl[i] - t_ln),
            "truth_inside": bool(lo <= truth <= hi),
            "interval_at_grid_edge": bool(i == 0 or i == len(grid) - 1 or lo <= grid[0] or hi >= grid[-1])}


def delta_profile(events: Mapping[str, np.ndarray], selection, kind: str, grid, *,
                  logp: LogDensity | None = None, log_xi_grid=None):
    """ln L(slope) = sum_i ln p_pop(theta_i) - N ln xi on delta-function PE (C2 slope scan)."""
    _, hp_truth = truth_model(kind)
    spec = c2_model_spec()
    logp = logp or LogDensity(spec)
    base = dict(hp_truth)
    base.setdefault(C2_SLOPE, 0.0)
    truth = float(base[C2_SLOPE])
    if log_xi_grid is None:
        log_xi_grid = [selection_log_xi(logp, selection, {**base, C2_SLOPE: float(v)}) for v in grid]
    n = len(events["q"])
    lnl = [float(np.sum(logp(events, {**base, C2_SLOPE: float(v)}))) - n * lx
           for v, lx in zip(grid, log_xi_grid)]
    return profile_summary(grid, lnl, truth), log_xi_grid


def coverage_gate(draw: DrawDistribution, selection, event_truths, *, graph_nodes=None,
                  box: Mapping | None = None) -> dict:
    """Rule 6 on the built artifact: the draw support contains every node's population support.

    Also checks that the PE prior box contains the population support (the
    PE prior must not truncate where the population has mass): m1_det up to
    mmax (1 + zmax), d_L up to d_L(zmax), q down to the population's q edge,
    chi_eff on [-1, 1].
    """
    from ..grammar.v2 import enumerate_v2_depth1

    nodes = graph_nodes if graph_nodes is not None else list(enumerate_v2_depth1().nodes)
    sup = draw.support()
    per_node, ok_all = [], True
    for node in nodes:
        s = node.support
        ok = (float(s["mmin"]) >= sup["m1_source_min"] and float(s["mmax"]) <= sup["m1_source_max"]
              and float(s["q_floor"]) >= sup["q_min"] and float(s["zmax"]) <= sup["z_max"])
        ok_all &= ok
        per_node.append({"model_hash": node.model_hash, "pass": bool(ok)})
    # grid over the population support (largest: the smallest q_floor of any node)
    s0 = nodes[0].support
    q_floor = min(float(n.support["q_floor"]) for n in nodes)
    m = np.geomspace(float(s0["mmin"]), float(s0["mmax"]), 61)
    z = np.linspace(1e-4, float(s0["zmax"]), 61)
    c = np.linspace(-1.0, 1.0, 41)
    M, Z, C, U = np.meshgrid(m, z, c, np.linspace(0.0, 1.0, 21), indexing="ij")
    qmin = np.maximum(q_floor, float(s0["mmin"]) / M)
    Q = qmin + U * (1.0 - qmin)
    lp = draw.log_density_source(M.ravel(), Q.ravel(), Z.ravel(), C.ravel())
    n_bad_grid = int(np.sum(~np.isfinite(lp)))
    lp_ev = draw.log_density_detector(event_truths["m1_source"], event_truths["q"], event_truths["z"],
                                      event_truths["chi_eff"])
    n_bad_events = int(np.sum(~np.isfinite(lp_ev)))
    n_bad_sel = int(np.sum(~np.isfinite(selection.log_draw_density)))
    box = pe_box(draw.cosmo) if box is None else box
    m1_det_max = max(float(n.support["mmax"]) * (1.0 + float(n.support["zmax"])) for n in nodes)
    m1_det_min = min(float(n.support["mmin"]) for n in nodes)
    d_max = float(draw.cosmo.dL_of_z(max(float(n.support["zmax"]) for n in nodes)))
    q_edge = min(max(float(n.support["q_floor"]), float(n.support["mmin"]) / float(n.support["mmax"]))
                 for n in nodes)
    box_ok = {
        "m1_detector": bool(box["m1_detector"][0] <= m1_det_min and box["m1_detector"][1] >= m1_det_max),
        "luminosity_distance": bool(box["luminosity_distance"][1] >= d_max * (1.0 - 1e-9)),
        "q": bool(box["q"][0] <= q_edge and box["q"][1] >= 1.0),
        "chi_eff": bool(box["chi_eff"][0] <= -1.0 and box["chi_eff"][1] >= 1.0),
    }
    passed = bool(ok_all and n_bad_grid == 0 and n_bad_events == 0 and n_bad_sel == 0 and all(box_ok.values()))
    return {
        "pass": passed,
        "pe_box": {k: list(v) for k, v in box.items()},
        "pe_box_contains_population_support": box_ok,
        "draw_support": sup,
        "nodes_support_inside_draw_support": bool(ok_all),
        "per_node": per_node,
        "support_grid_points": int(lp.size),
        "support_grid_points_with_zero_draw_density": n_bad_grid,
        "event_truths_with_zero_draw_density": n_bad_events,
        "selection_rows_with_zero_draw_density": n_bad_sel,
        "fraction_detected_population_outside_draw_support": 0.0 if passed else None,
        "note": ("the v2 population support (mmin, mmax, q_floor, zmax; chi_eff in [-1, 1]) is fixed, "
                 "independent of the hyperparameters, so containment of the support in the draw "
                 "support covers every trial hyperparameter of the prior box"),
    }


def g12_gate(posterior, selection) -> dict:
    """G12 data-support check of every depth-1 graph node on the mock pair."""
    from ..grammar.v2 import enumerate_v2_depth1
    from ..models.data_support import v2_data_support_report

    reports = [v2_data_support_report(node, posterior, selection) for node in enumerate_v2_depth1().nodes]
    return {"pass": all(r["pass"] for r in reports), "n_models": len(reports),
            "failed": {r["model_hash"]: [c for c in r["checks"] if not c["passed"]]
                       for r in reports if not r["pass"]},
            "reported_root": reports[0]["reported"]}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

C2_SLOPE_GRID = tuple(np.round(np.arange(-6.0, 2.0001, 0.1), 10))
MOCK_CAMPAIGN_ID = "MOCK_O1_O4b"


def _git_commit() -> str | None:
    import subprocess

    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover - provenance only
        return None


def refuse_protected(path: Path) -> None:
    parts = Path(path).resolve().parts
    if "frozen" in parts or "v2r2" in parts:
        raise SystemExit(f"refusing to write a mock under a frozen/ or v2r2 directory: {path}")


def build_selection(inj: Mapping[str, np.ndarray], n_draw: int, cal: Calibration, draw: DrawDistribution,
                    *, seed: int):
    from ..data import Campaign, SelectionCatalog, SelectionMode
    from ..data.adapters.gwcat_v2 import basis_for_spin

    samples = {
        "m1_detector": inj["m1_detector"], "q": inj["q"], "luminosity_distance": inj["luminosity_distance"],
        "chi_eff": inj["chi_eff"], "m1_source": inj["m1_source"], "m2_source": inj["q"] * inj["m1_source"],
        "z": inj["z"], "run_index": inj["run_index"].astype(float),
    }
    return SelectionCatalog(
        samples=samples,
        log_draw_density=inj["log_draw_density"],
        campaign_id=np.full(len(inj["q"]), MOCK_CAMPAIGN_ID),
        campaigns=(Campaign(MOCK_CAMPAIGN_ID, n_draw=int(n_draw), observing_time_yr=cal.total_time_yr),),
        basis=basis_for_spin("chieff", sky_marginal=True),
        mode=SelectionMode.RAW_DRAW,
        metadata={
            "mock": True, "mock_format_version": MOCK_FORMAT_VERSION, "z_max": DRAW_ZMAX, "sky_marginal": True,
            "draw_support": {k: draw.support()[k] for k in ("m1_source_min", "m1_source_max", "q_min", "z_max")},
            "detection_rule": f"rho_obs = A_k Theta g(theta) + N(0, 1) > {SNR_THRESHOLD}",
            "run_labels": list(cal.labels), "run_amplitude": list(cal.amplitude),
            "run_time_yr": list(cal.run_time_yr), "mass_rolloff": cal.mass_rolloff,
            "run_index_column": "run_index (index into run_labels; the run is drawn with probability T_k / T)",
            "seed": int(seed),
        },
    )


def _event_truth_record(name, pool, j, cal, x_obs=None, acc=None):
    rec = {"name": name, "run": cal.labels[int(pool["run_index"][j])]}
    for k in ("m1_source", "q", "z", "chi_eff", "m1_detector", "luminosity_distance", "theta", "rho_opt",
              "rho_obs", "log_draw_density"):
        rec[k] = float(pool[k][j])
    if x_obs is not None:
        rec["x_obs"] = dict(zip(X_NAMES, (float(v) for v in x_obs)))
    if acc is not None:
        rec["pe_acceptance"] = float(acc)
    return rec


def pe_profile(posterior, selection, kind: str, grid, log_xi_grid, logp: LogDensity) -> dict:
    """Untapered shape ln L(slope) on the synthetic PE (other parameters at the truth)."""
    _, hp_truth = truth_model(kind)
    base = dict(hp_truth)
    base.setdefault(C2_SLOPE, 0.0)
    offsets = posterior.offsets
    lnl = []
    for v, lx in zip(grid, log_xi_grid):
        lp = logp(posterior.samples, {**base, C2_SLOPE: float(v)}) - posterior.log_ref_density
        ev = [logsumexp(lp[offsets[i]:offsets[i + 1]]) - math.log(offsets[i + 1] - offsets[i])
              for i in range(posterior.n_events)]
        lnl.append(float(np.sum(ev)) - posterior.n_events * lx)
    return profile_summary(grid, lnl, float(base[C2_SLOPE]))


#: b-0 taper gates (review B items 1 and 5; DRAFT): sigma^2_lnL at the truth must
#: be at most this fraction of the v2 cut threshold (the truth must sit well
#: inside the sharp cut), and the per-event term at most this multiple of the
#: canonical (real) catalog's at the same truth.
TRUTH_VARIANCE_MAX_FRACTION = 0.5
EVENT_VARIANCE_MAX_RATIO = 1.5
#: representativeness of the realised catalog: |MLE - truth| <= this x sd_ens
REPRESENTATIVE_N_SD = 3.0
DEFAULT_N_PE = 8192
DEFAULT_TARGET_FOUND = 1_500_000
DEFAULT_ENSEMBLE = 200


def _package_versions() -> dict:
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "jax", "jaxlib", "h5py"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover
            out[name] = None
    return out


def build_mock(output_dir, *, kind: str = "closure_widthq", n_events: int = 259, n_pe: int = DEFAULT_N_PE,
               target_found: int = DEFAULT_TARGET_FOUND, seed: int = 20261030, delta_pe: bool = False,
               calibration: Calibration | None = None, calibration_source: str | None = None,
               n_pool: int | None = None, run_gates: bool = True, ensemble: int = DEFAULT_ENSEMBLE,
               grid=C2_SLOPE_GRID, calibration_kwargs: Mapping | None = None, log=print) -> dict:
    """Generate one canonical-format mock pair (pe.h5, selection.h5) plus truth, gates and manifest.

    ``calibration_source`` is recorded in the manifest (the path of a reused
    calibration.json); without a ``calibration`` the calibration is computed.
    """
    from ..data import PosteriorCatalog, validate_pair
    from ..data.adapters.gwcat_v2 import basis_for_spin

    out = Path(output_dir).resolve()
    refuse_protected(out)
    out.mkdir(parents=True, exist_ok=True)
    cosmo = default_cosmology()
    draw = DrawDistribution(cosmo)
    if calibration is None:
        calibration = calibrate(draw, log=log, **dict(calibration_kwargs or {}))
        cal_record = {"computed": True, "seed": int(calibration.report.get("seed", CALIBRATION_SEED))}
    elif calibration_source:
        cal_record = {"reused": str(calibration_source), "sha256": sha256_file(calibration_source)}
    else:
        cal_record = {"provided_in_memory": True}
    cal = calibration
    box = pe_box(cosmo)
    (out / "calibration.json").write_text(json.dumps(cal.to_dict(), indent=2, sort_keys=True) + "\n")
    spec, hp = truth_model(kind)
    logp = LogDensity(spec)
    ss = np.random.SeedSequence(int(seed))
    inj_ss, pool_ss, pe_ss, ens_ss = ss.spawn(4)

    # -- injections (the selection set) ---------------------------------------
    det_frac = float(cal.report.get("draw_detection_fraction") or 0.05)
    n_draw = int(math.ceil(target_found / det_frac))
    inj = simulate_detections(draw, cal, np.random.default_rng(inj_ss), n_draw)
    selection = build_selection(inj, n_draw, cal, draw, seed=seed)
    log(f"injections: n_draw {n_draw} found {selection.n_selected}")

    # -- events: resample detected systems of an event pool (same machinery) ---
    n_pool = int(n_pool or max(n_draw // 2, 50 * n_events))
    pool_rng = np.random.default_rng(pool_ss)
    pool = simulate_detections(draw, cal, pool_rng, n_pool)
    lw = logp(pool, hp) - pool["log_draw_density"]
    w = np.exp(lw - np.max(lw[np.isfinite(lw)]))
    w[~np.isfinite(w)] = 0.0
    pool_ess = float(w.sum() ** 2 / np.sum(w ** 2))
    picks = pool_rng.choice(len(w), size=n_events, replace=False, p=w / w.sum())
    log(f"event pool: n_draw {n_pool} detected {len(w)} ESS {pool_ess:.0f}")

    names = [f"MOCK{i:03d}" for i in range(n_events)]
    cols = {k: [] for k in ("m1_detector", "q", "luminosity_distance", "chi_eff")}
    log_ref, records = [], []
    for i, (name, j, ev_ss) in enumerate(zip(names, picks, pe_ss.spawn(n_events))):
        rng = np.random.default_rng(ev_ss)
        amp = cal.amplitude[int(pool["run_index"][j])]
        x_true = x_of(pool["m1_detector"][j], pool["q"][j], pool["chi_eff"][j])
        x_obs = observe(rng, x_true, pool["rho_obs"][j], cal.width_scale)
        if delta_pe:
            s = {k: np.array([pool[k][j]]) for k in cols}
            acc = None
        else:
            s, acc = sample_pe(rng, x_obs, pool["rho_obs"][j], amp, n_pe, cal.width_scale, cal.mass_rolloff,
                               box=box, rolloff_power=cal.mass_rolloff_power,
                               rolloff_sharpness=cal.mass_rolloff_sharpness)
        for k in cols:
            cols[k].append(s[k])
        log_ref.append(pe_log_prior_basis(s["m1_detector"], s["luminosity_distance"], box))
        records.append(_event_truth_record(name, pool, j, cal, x_obs, acc))
    samples = {k: np.concatenate(v) for k, v in cols.items()}
    z = cosmo.z_of_dL(samples["luminosity_distance"])
    samples["z"] = z
    samples["m1_source"] = samples["m1_detector"] / (1.0 + z)
    samples["m2_source"] = samples["q"] * samples["m1_source"]
    per = 1 if delta_pe else n_pe
    posterior = PosteriorCatalog(
        event_names=tuple(names), offsets=np.arange(n_events + 1) * per, samples=samples,
        log_ref_density=np.concatenate(log_ref), basis=basis_for_spin("chieff", sky_marginal=True),
        metadata={"mock": True, "mock_format_version": MOCK_FORMAT_VERSION, "mock_kind": kind,
                  "sky_marginal": True, "z_max": DRAW_ZMAX, "seed": int(seed), "delta_pe": bool(delta_pe),
                  "log_ref_density": "ln pi_PE(m1_det, q, d_L, chi_eff) in dm1_det dq dd_L dchi_eff; "
                                     f"pi ∝ m1_det d_L^2 on the box {box}",
                  "pe_box": {k: list(v) for k, v in box.items()},
                  "z_max_note": "the PE prior box ends at d_L(z_max), so no PE sample lies above z_max"},
    )
    validate_pair(posterior, selection, ("m1_detector", "q", "luminosity_distance", "chi_eff"))
    posterior.to_hdf5(out / "pe.h5")
    selection.to_hdf5(out / "selection.h5")
    truth = {
        "format_version": MOCK_FORMAT_VERSION, "kind": kind, "delta_pe": bool(delta_pe),
        "generating_model_hash": spec.model_hash, "generating_model_spec": spec.to_dict(),
        "truth_hyperparameters": hp, "lvk_fiducial_medians": LVK_GWTC5_FIDUCIAL_MEDIANS,
        "c2_slope_truth": float(hp.get(C2_SLOPE, 0.0)), "c2_model_hash": c2_model_spec().model_hash,
        "events": records,
    }
    (out / "mock_truth.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")

    # -- gates (b-0) -------------------------------------------------------------
    gates = {"run": bool(run_gates)}
    if run_gates:
        c2 = c2_model_spec()
        logp_c2 = LogDensity(c2)
        ev_truth = {k: pool[k][picks] for k in ("m1_detector", "q", "luminosity_distance", "chi_eff",
                                                "m1_source", "z")}
        prof, log_xi_grid = delta_profile(ev_truth, selection, kind, grid, logp=logp_c2)
        # (i) machinery gate on an ensemble of same-pool catalogs drawn like the
        # real one (without replacement); the realised catalog is reported
        # against the pre-declared representativeness rule (review B item 2)
        ens_rng = np.random.default_rng(ens_ss)
        ens = []
        for _ in range(int(ensemble)):
            idx = ens_rng.choice(len(w), size=n_events, replace=False, p=w / w.sum())
            e = {k: pool[k][idx] for k in ("m1_detector", "q", "luminosity_distance", "chi_eff")}
            ens.append(delta_profile(e, selection, kind, grid, logp=logp_c2, log_xi_grid=log_xi_grid)[0])
        mles = np.array([p["mle"] for p in ens])
        truth_slope = prof["truth"]
        ens_summary = {
            "n_catalogs": int(ensemble), "n_events_each": n_events, "draw": "without replacement",
            "mle_mean": float(mles.mean()) if len(mles) else None,
            "mle_sd": float(mles.std(ddof=1)) if len(mles) > 1 else None,
            "coverage_dlnL_le_2": float(np.mean([p["truth_inside"] for p in ens])) if ens else None,
            "n_interval_at_grid_edge": int(sum(p["interval_at_grid_edge"] for p in ens)),
            "mles": mles.round(4).tolist(),
        }
        if len(mles) > 1:
            se = ens_summary["mle_sd"] / math.sqrt(len(mles))
            ens_summary["mle_mean_minus_truth_over_se"] = float((mles.mean() - truth_slope) / se)
            ens_summary["pass"] = bool(abs(mles.mean() - truth_slope) <= 3.0 * se
                                       and ens_summary["coverage_dlnL_le_2"] >= 0.8)
        sd = ens_summary.get("mle_sd")
        representative = {
            "rule": f"|MLE - truth| <= {REPRESENTATIVE_N_SD:g} sd_ens (pre-declared, SEED_DECLARATION.json)",
            "mle": prof["mle"], "truth": truth_slope, "sd_ens": sd,
            "n_sd": None if not sd else float(abs(prof["mle"] - truth_slope) / sd),
            "ensemble_quantile_of_mle": float(np.mean(mles <= prof["mle"])) if len(mles) else None,
            "pass": bool(sd and abs(prof["mle"] - truth_slope) <= REPRESENTATIVE_N_SD * sd),
        }
        gates["i_delta_pe_profile"] = {
            "parameter": C2_SLOPE, "model": "C2 (other hyperparameters at the truth)",
            "this_catalog": prof, "ensemble": ens_summary, "realised_catalog_representative": representative,
            "pass": bool(ens_summary.get("pass", False)),
            "note": ("binding: the ensemble machinery check; the realised catalog's own dlnL <= 2 interval "
                     "misses the truth for ~5 % of honest seeds and is reported, not gated"),
        }
        if not delta_pe:
            gates["pe_profile_reported"] = pe_profile(posterior, selection, kind, grid, log_xi_grid, logp_c2)
        gates["ii_coverage"] = coverage_gate(draw, selection, ev_truth, box=box)
        gates["iii_g12"] = g12_gate(posterior, selection)
        lpd = logp(selection.samples, hp) - selection.log_draw_density
        wsel = np.exp(lpd - lpd.max())
        gates["selection_diagnostics"] = {
            "n_found": int(selection.n_selected), "n_draw": int(n_draw),
            "ess_at_truth": float(wsel.sum() ** 2 / np.sum(wsel ** 2)),
            "n_events_sq_over_ess": float(n_events ** 2 / (wsel.sum() ** 2 / np.sum(wsel ** 2))),
            "canonical_selection_ess_at_truth": CANONICAL_SELECTION_ESS[kind],
            "event_pool_ess": pool_ess,
        }
        if not delta_pe:
            tv = taper_variance_at_truth(posterior, selection, kind)
            canon = CANONICAL_TAPER_VARIANCE[kind]
            gates["iv_taper_at_truth"] = {
                "estimator": "hbi.numpy_backend.shape_log_likelihood / taper_variance with the v2 sharp cut",
                "mock": tv, "canonical_at_same_truth": canon,
                "max_total": TRUTH_VARIANCE_MAX_FRACTION * tv["threshold"],
                "event_ratio_to_canonical": tv["events"] / canon["events"],
                "max_event_ratio": EVENT_VARIANCE_MAX_RATIO,
                "pass_total": bool(tv["total"] <= TRUTH_VARIANCE_MAX_FRACTION * tv["threshold"]),
                "pass_events": bool(tv["events"] <= EVENT_VARIANCE_MAX_RATIO * canon["events"]),
            }
            gates["iv_taper_at_truth"]["pass"] = bool(gates["iv_taper_at_truth"]["pass_total"]
                                                      and gates["iv_taper_at_truth"]["pass_events"])
        rep = gates["i_delta_pe_profile"]["realised_catalog_representative"]
        gates["b0_pass"] = bool(gates["i_delta_pe_profile"]["pass"] and rep["pass"]
                                and gates["ii_coverage"]["pass"] and gates["iii_g12"]["pass"]
                                and gates.get("iv_taper_at_truth", {"pass": True})["pass"])
        (out / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
        log(f"gates: b0_pass {gates['b0_pass']} (ensemble {gates['i_delta_pe_profile']['pass']}, "
            f"representative {rep['pass']} ({rep['n_sd']}), coverage {gates['ii_coverage']['pass']}, "
            f"G12 {gates['iii_g12']['pass']}, taper {gates.get('iv_taper_at_truth', {}).get('mock')})")

    files = {name: sha256_file(out / name) for name in ("pe.h5", "selection.h5", "mock_truth.json",
                                                        "calibration.json")}
    if run_gates:
        files["gates.json"] = sha256_file(out / "gates.json")
    manifest = {
        "format_version": MOCK_FORMAT_VERSION, "kind": kind, "delta_pe": bool(delta_pe),
        "status": "STAGING MOCK (not frozen)", "code_commit": _git_commit(),
        "software_versions": _package_versions(),
        "calibration_source": cal_record,
        "generator": "scripts/v2_closure_mock.py / gwpop_search.validation.v2_mock",
        "seeds": {"master": int(seed), "calibration": int(cal.report.get("seed", CALIBRATION_SEED)),
                  "streams": "SeedSequence(master).spawn(4) = injections, event pool, per-event PE, ensemble"},
        "sizes": {"n_events": n_events, "n_pe": per, "n_draw": n_draw, "n_found": int(selection.n_selected),
                  "event_pool_draws": n_pool, "event_pool_detected": int(len(w))},
        "truth": {"model_hash": spec.model_hash, "hyperparameters": hp,
                  "c2_slope": float(hp.get(C2_SLOPE, 0.0))},
        "dag": {
            "detection": f"rho_obs = A_k Theta g(theta) + n, n ~ N(0, 1); detected iff rho_obs > {SNR_THRESHOLD}; "
                         "identical function for injections and events",
            "events": "p_pop/p_draw-weighted resample (without replacement) of the detected systems of an "
                      "event pool drawn with the injection machinery (run k with probability T_k/T)",
            "pe_data": "(rho_obs, x_obs), x_obs = (ln m1_det, q, chi_eff) + N(0, Sigma(rho_obs)); "
                       "Sigma widths c_j rho_thr/rho_obs, constant correlation",
            "pe_posterior": "exact draws from N(x_obs; x, Sigma) E_Theta[N(rho_obs; A_k Theta g(theta), 1)] "
                            "pi_PE(theta); pi_PE ∝ m1_det d_L^2 on pe_box (d_L <= d_L(z_max))",
            "snr_mass_rolloff": f"(1 + (M_det / M_ro)^s)^(-p/s), M_ro = {cal.mass_rolloff:.2f}, "
                                f"p = {cal.mass_rolloff_power:.4f}, s = {cal.mass_rolloff_sharpness:g}",
            "pe_box": {k: list(v) for k, v in box.items()}, "snr_threshold": SNR_THRESHOLD,
            "spin_snr_slope": SPIN_SNR_SLOPE,
            "selection_mode": ("RAW_DRAW (mock); frozen-selection null replays require estimator_ready, "
                               "so the mock campaign uses --max-null-replays 0"),
            "draw_distribution": {"support": draw.support(), "proxy_fraction": DRAW_PROXY_FRACTION,
                                  "proxy": DRAW_PROXY},
        },
        "calibration": cal.to_dict(),
        "gates_b0_pass": gates.get("b0_pass"),
        "files_sha256": files,
    }
    (out / "mock_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
