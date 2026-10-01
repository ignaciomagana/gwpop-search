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
total-mass roll-off reproduces the canonical pooled m1 90 % quantile. The
x_obs widths are calibrated to the median per-event posterior widths of the
real 259-event catalog (:data:`REAL_PE_WIDTH_MEDIANS`); the luminosity
distance is informed only by the amplitude datum (with the projection
marginalised), so its posterior width is an outcome, not a calibration.

Known simplifications (a mock, not a waveform model): one projection factor
(single-interferometer antenna pattern) for every run, the SNR scaling of
:func:`snr_unit`, Gaussian x_obs in (ln m1_det, q, chi_eff), the same
threshold rho_obs > 10 in every run (the canonical O1/O2 rule; O3/O4 use
FAR < 1/yr), and the PE prior ∝ m1_det d_L^2 with a uniform chi_eff prior.
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
#: staging/v2/canonical/selection.h5 (sha256 below) with run labels from
#: gwcat/build/v2r2/selection_row_subsets_index.h5, weights p_pop / p_draw at
#: the R0 fiducial with chi_eff N(0.03, 0.08) (2026-09-30). ``share`` is the
#: run's fraction of sum(p_pop / p_draw); ``z`` the weighted 10/50/90 %
#: quantiles of the found-injection redshift. Recompute with
#: :func:`canonical_run_targets`.
CANONICAL_RUN_TARGETS: dict[str, dict[str, object]] = {
    "O1": {"share": 0.005627703616347835, "z": [0.0609, 0.1677, 0.3669], "m1_source": [10.21, 32.60, 53.36],
           "q": [0.560, 0.831, 0.971], "chi_eff": [-0.068, 0.036, 0.135]},
    "O2": {"share": 0.025523314137665745, "z": [0.0764, 0.2036, 0.4498], "m1_source": [9.97, 32.82, 53.99],
           "q": [0.563, 0.830, 0.970], "chi_eff": [-0.067, 0.034, 0.138]},
    "O3a": {"share": 0.12035247013469419, "z": [0.1166, 0.3240, 0.7033], "m1_source": [10.05, 32.13, 49.67],
            "q": [0.570, 0.830, 0.970], "chi_eff": [-0.068, 0.033, 0.137]},
    "O3b": {"share": 0.11233453249639039, "z": [0.1193, 0.3375, 0.7353], "m1_source": [10.00, 32.30, 49.87],
            "q": [0.565, 0.832, 0.971], "chi_eff": [-0.067, 0.035, 0.138]},
    "O4a": {"share": 0.327651594307621, "z": [0.1575, 0.4646, 1.0339], "m1_source": [10.27, 32.62, 49.00],
            "q": [0.565, 0.831, 0.971], "chi_eff": [-0.067, 0.036, 0.139]},
    "O4b": {"share": 0.4085103853072808, "z": [0.1634, 0.4850, 1.1040], "m1_source": [10.31, 32.78, 48.92],
            "q": [0.564, 0.829, 0.970], "chi_eff": [-0.068, 0.037, 0.138]},
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
#: aligned-spin SNR factor (1 + SPIN_SNR_SLOPE chi_eff): aligned spins are louder
SPIN_SNR_SLOPE = 0.2
#: widths are c_j rho_thr / max(rho_obs, RHO_WIDTH_FLOOR) (the floor only
#: matters for undetected systems, e.g. in the SBC test)
RHO_WIDTH_FLOOR = 5.0

#: Uniform-in-(m1_det, m2_det) and Euclidean-volume PE prior on a box:
#: pi(m1_det, q, d_L, chi_eff) ∝ m1_det d_L^2 (chi_eff uniform).
PE_BOX = {"m1_detector": (2.0, 2000.0), "q": (0.001, 1.0),
          "luminosity_distance": (1.0, 30000.0), "chi_eff": (-1.0, 1.0)}

#: Injection draw distribution (source frame). Its support contains the fixed
#: v2 population support for every q_floor in [0.001, 0.05].
DRAW_M1 = (3.0, 300.0)
DRAW_Q_LO = 0.001
DRAW_ZMAX = 1.9
#: Proxy settings chosen so the selection ESS at the two truths (about 104k /
#: 117k of 1.5M rows, closure / null) is comparable to the canonical set's
#: (:data:`CANONICAL_SELECTION_ESS`); the broad component keeps full support.
DRAW_PROXY_FRACTION = 0.8
DRAW_PROXY = {"ln_m1_mu": math.log(25.0), "ln_m1_sigma": 0.6, "q_power": 1.0,
              "chi_mu": 0.03, "chi_sigma": 0.08}
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


class DrawDistribution:
    """Exact source-frame draw density, a broad + population-proxy mixture.

    z ~ dVc/dz / (1 + z) on [0, 1.9] in both components; broad: m1
    log-uniform on [3, 300], q uniform on [0.001, 1], chi_eff uniform on
    [-1, 1]; proxy: ln m1 truncated normal, q ∝ q, chi_eff truncated normal.
    The density is evaluated in the detector basis (m1_det, q, d_L, chi_eff)
    with the exact Jacobian 1 / ((1 + z) d d_L / dz).
    """

    def __init__(self, cosmo: NumpyCosmology, *, proxy_fraction: float = DRAW_PROXY_FRACTION):
        self.cosmo = cosmo
        self.f_p = float(proxy_fraction)
        z = np.linspace(0.0, DRAW_ZMAX, 40001)
        pz = cosmo.dVc_dz(z) / (1.0 + z)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pz[1:] + pz[:-1]) * np.diff(z))])
        self._z, self._cdf = z, cdf / cdf[-1]
        self._log_pz_norm = float(np.log(cdf[-1]))
        lo, hi = math.log(DRAW_M1[0]), math.log(DRAW_M1[1])
        self._lnm_bounds = (lo, hi)

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
        lnm[~proxy] = rng.uniform(lo, hi, n_b)
        q[~proxy] = rng.uniform(DRAW_Q_LO, 1.0, n_b)
        chi[~proxy] = rng.uniform(-1.0, 1.0, n_b)
        lnm[proxy] = _truncnorm_sample(rng, n_p, DRAW_PROXY["ln_m1_mu"], DRAW_PROXY["ln_m1_sigma"], lo, hi)
        k = DRAW_PROXY["q_power"] + 1.0
        q[proxy] = (DRAW_Q_LO ** k + rng.uniform(size=n_p) * (1.0 - DRAW_Q_LO ** k)) ** (1.0 / k)
        chi[proxy] = _truncnorm_sample(rng, n_p, DRAW_PROXY["chi_mu"], DRAW_PROXY["chi_sigma"], -1.0, 1.0)
        z = np.interp(rng.uniform(size=n), self._cdf, self._z)
        return {"m1_source": np.exp(lnm), "q": q, "z": z, "chi_eff": chi}

    def log_density_source(self, m1, q, z, chi):
        """ln p_draw(m1_source, q, z, chi_eff)."""
        lo, hi = self._lnm_bounds
        lnm = np.log(m1)
        in_m = (lnm >= lo) & (lnm <= hi)
        in_q = (q >= DRAW_Q_LO) & (q <= 1.0)
        in_c = (chi >= -1.0) & (chi <= 1.0)
        in_z = (z > 0.0) & (z <= DRAW_ZMAX)
        log_b = -lnm - math.log(hi - lo) - math.log(1.0 - DRAW_Q_LO) - math.log(2.0)
        k = DRAW_PROXY["q_power"] + 1.0
        log_qp = math.log(k) + (k - 1.0) * np.log(np.where(q > 0, q, 1.0)) - math.log(1.0 - DRAW_Q_LO ** k)
        log_p = (_truncnorm_logpdf(lnm, DRAW_PROXY["ln_m1_mu"], DRAW_PROXY["ln_m1_sigma"], lo, hi) - lnm
                 + log_qp + _truncnorm_logpdf(chi, DRAW_PROXY["chi_mu"], DRAW_PROXY["chi_sigma"], -1.0, 1.0))
        log_mix = np.logaddexp(math.log1p(-self.f_p) + log_b, math.log(self.f_p) + log_p)
        safe_z = np.where(in_z, z, 0.5)
        log_z = np.log(self.cosmo.dVc_dz(safe_z)) - np.log1p(safe_z) - self._log_pz_norm
        ok = in_m & in_q & in_c & in_z
        return np.where(ok, log_mix + log_z, -np.inf)

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


def snr_unit(m1_det, q, d_l, chi, m_rolloff):
    """g(theta) at unit amplitude and Theta = 1.

    (Mc_det / 10)^(5/6) (1000 Mpc / d_L) (1 + k chi_eff) (1 + M_det / M_ro)^(-5/6):
    the inspiral chirp-mass/distance scaling, louder for aligned spins, and a
    roll-off above the detector-frame total mass M_ro (merger leaving the
    band; for M_det >> M_ro the SNR no longer grows with mass).
    """
    mc = m1_det * q ** 0.6 / (1.0 + q) ** 0.2
    m_tot = m1_det * (1.0 + q)
    return ((mc / 10.0) ** (5.0 / 6.0) * (1000.0 / d_l) * (1.0 + SPIN_SNR_SLOPE * chi)
            * (1.0 + m_tot / m_rolloff) ** (-5.0 / 6.0))


@dataclass(frozen=True)
class Calibration:
    """Per-run SNR amplitudes A_k and run times T_k, plus the PE width scales."""

    amplitude: tuple[float, ...]
    run_time_yr: tuple[float, ...]
    width_scale: tuple[float, float, float]
    mass_rolloff: float
    labels: tuple[str, ...] = RUN_LABELS
    report: dict = field(default_factory=dict, compare=False)

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
                "mass_rolloff": float(self.mass_rolloff), "report": self.report}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Calibration":
        return cls(tuple(float(x) for x in d["amplitude"]), tuple(float(x) for x in d["run_time_yr"]),
                   tuple(float(x) for x in d["width_scale"]), float(d["mass_rolloff"]),
                   tuple(d.get("labels", RUN_LABELS)), dict(d.get("report", {})))


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
        rho_opt = amp[run] * theta * snr_unit(m1_det, s["q"], d_l, s["chi_eff"], cal.mass_rolloff)
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


def _log_c(m1_det, q, chi, amplitude, m_rolloff):
    """ln[A d_L g(theta)], so that rho_opt = Theta exp(c) / d_L (see :func:`snr_unit`)."""
    return np.log(amplitude * snr_unit(m1_det, q, 1000.0, chi, m_rolloff)) + math.log(1000.0)


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
              max_proposals: int = 200_000_000):
    """Exact posterior draws of (m1_det, q, d_L, chi_eff) given d = (rho_obs, x_obs).

    With y_r = (ln m1_det, q, chi_eff), y2 = ln d_L, c(y_r) as in :func:`_log_c`
    and rho_opt = Theta exp(c - y2), the posterior is ::

        p ∝ N(x_obs; y_r, Sigma) e^{2 y0} e^{3 y2} p(Theta) N(rho_obs; Theta e^{c - y2}, 1) 1_box

    (e^{2 y0} e^{3 y2} is the prior m1_det d_L^2 times the Jacobian m1_det d_L).
    Substituting r = Theta e^{c - y2} (i.e. y2 = c + ln Theta - ln r) gives
    e^{3 y2} = e^{3c} Theta^3 r^-3 and dy2 = dr / r, so without the box ::

        p ∝ [N(x_obs; y_r, Sigma) e^{2 y0} e^{3 c(y_r)}] [Theta^3 p(Theta)] [r^-4 N(rho_obs; r, 1)]

    three independent factors. e^{3c} ∝ e^{2.5 y0} h(q) (1 + k chi)^3 R(M)^3
    with h(q) = q^1.5 (1 + q)^-0.5 <= h(1) and R(M) = (1 + M_det / M_ro)^(-5/6)
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
               * (1.0 + np.exp(y[:, 0]) * (1.0 + q) / m_rolloff) ** -2.5)
        y = y[rng.uniform(size=len(y)) < acc]
        if len(y) == 0:
            continue
        m1 = np.exp(y[:, 0])
        theta = _sample_theta_cubed(rng, len(y))
        r = _sample_rho_opt(rng, rho_obs, len(y))
        d_l = np.exp(_log_c(m1, y[:, 1], y[:, 2], amplitude, m_rolloff)) * theta / r
        ok = (d_l >= d_lo) & (d_l <= d_hi)
        kept["m1_detector"].append(m1[ok])
        kept["q"].append(y[ok, 1])
        kept["chi_eff"].append(y[ok, 2])
        kept["luminosity_distance"].append(d_l[ok])
        n_kept += int(ok.sum())
    out = {k: np.concatenate(v)[:n_samples] for k, v in kept.items()}
    return out, n_kept / n_prop


def pe_log_likelihood_unmarginalised(x_obs, rho_obs, amplitude, width_scale, m_rolloff, *,
                                     m1_det, q, d_l, chi, theta):
    """ln L(d | theta, Theta) up to a constant (for tests)."""
    cov = pe_covariance(rho_obs, width_scale)
    r = x_of(m1_det, q, chi) - x_obs
    quad = np.einsum("...i,ij,...j->...", r, np.linalg.inv(cov), r)
    rho = theta * np.exp(_log_c(m1_det, q, chi, amplitude, m_rolloff)) / d_l
    return -0.5 * quad - 0.5 * (rho_obs - rho) ** 2


# ---------------------------------------------------------------------------
# Calibration to the canonical selection and the real PE widths
# ---------------------------------------------------------------------------

def weighted_quantiles(x, w, qs=(0.1, 0.5, 0.9)):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    c = c / c[-1]
    return [float(np.interp(q, c, x[o])) for q in qs]


#: canonical pooled (all runs) detected R0 m1_source 90 % quantile: the
#: roll-off calibration target
CANONICAL_POOLED_M1_Q90 = 49.287


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


def _calibrate_amplitudes(pool, w, m_rolloff, targets, zq):
    unit = pool["base"] * (1.0 + pool["m_tot"] / m_rolloff) ** (-5.0 / 6.0)
    amps, dets = [], []
    for label in RUN_LABELS:
        target = float(targets[label]["z"][1])
        lo, hi = 0.0, math.log(5000.0)
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            det = math.exp(mid) * unit + pool["noise"] > SNR_THRESHOLD
            if zq(w * det, 0.5) < target:
                lo = mid
            else:
                hi = mid
        a = math.exp(0.5 * (lo + hi))
        amps.append(a)
        dets.append(a * unit + pool["noise"] > SNR_THRESHOLD)
    return amps, dets


def calibrate(draw: DrawDistribution, *, n_pool: int = 3_000_000, n_width_events: int = 200,
              n_width_samples: int = 1024, width_iterations: int = 5, seed: int = CALIBRATION_SEED,
              targets: Mapping[str, Mapping] = CANONICAL_RUN_TARGETS,
              pooled_m1_q90: float = CANONICAL_POOLED_M1_Q90, log=print) -> Calibration:
    """Calibrate the mock detection to the canonical selection and the PE to the real widths.

    * per run k: A_k so that the detected R0 population has the canonical
      median redshift; T_k so that run k has the canonical share of the
      detections (share_k ∝ T_k beta_k);
    * globally: the mass roll-off M_ro so that the pooled detected m1_source
      90 % quantile is the canonical one (A_k re-fit at every M_ro);
    * PE: the x_obs width scales c_j so that the median per-event posterior
      std of (ln m1_det, q, chi_eff) is the real catalog's.

    Uses the null truth (R0) and a fixed seed, so both mocks share one
    calibration; common random numbers throughout (monotone bisections,
    fixed-point iteration for c_j).
    """
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

    def pooled_q90(m_ro):
        amps, dets = _calibrate_amplitudes(pool, w, m_ro, targets, zq)
        betas = np.array([np.sum(w * d) for d in dets])
        mix = sum(sh / b * (w * d) for sh, b, d in zip(share, betas, dets))
        return mq(mix, 0.9), amps, dets, betas

    lo, hi = math.log(20.0), math.log(5000.0)
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if pooled_q90(math.exp(mid))[0] < pooled_m1_q90:
            lo = mid
        else:
            hi = mid
    m_ro = math.exp(0.5 * (lo + hi))
    q90, amps, dets, betas = pooled_q90(m_ro)
    t = share / betas
    t = t * T_TOTAL_YR / t.sum()
    report = {"mass_rolloff": m_ro, "pooled_m1_source_q90": {"achieved": q90, "target": pooled_m1_q90}}
    for label, a, d, tk in zip(RUN_LABELS, amps, dets, t):
        report[label] = {
            "amplitude": a, "run_time_yr": float(tk),
            "achieved": {k: weighted_quantiles(pool[k][d], w[d]) for k in ("z", "m1_source", "q", "chi_eff")},
            "target": {k: list(targets[label][k]) for k in ("z", "m1_source", "q", "chi_eff")},
            "pool_detected_ess": float(np.sum(w * d) ** 2 / np.sum((w * d) ** 2)),
        }
    cal = Calibration(tuple(amps), tuple(float(x) for x in t), X_WIDTH_TARGETS, m_ro)
    det_frac = float(sum(p * np.mean(d) for p, d in zip(cal.run_probability, dets)))
    log(f"calibration: M_ro {m_ro:.2f} amplitudes {np.round(amps, 3).tolist()} run times "
        f"{np.round(t, 4).tolist()} draw detection fraction {det_frac:.5f}")

    # -- PE widths: match the median per-event posterior std of the real catalog
    ev = _resample_detected(pool, w, cal, rng, n_width_events)
    eps = rng.standard_normal((n_width_events, 3))
    pe_seeds = rng.integers(0, 2 ** 63 - 1, n_width_events)
    scale = np.asarray(X_WIDTH_TARGETS, dtype=float).copy()
    history = []
    for it in range(width_iterations + 1):
        stds, corrs, accs = [], [], []
        for i in range(n_width_events):
            cov = pe_covariance(ev["rho_obs"][i], scale)
            x_true = x_of(ev["m1_detector"][i], ev["q"][i], ev["chi_eff"][i])
            x_obs = x_true + np.linalg.cholesky(cov) @ eps[i]
            s, acc = sample_pe(np.random.default_rng(pe_seeds[i]), x_obs, ev["rho_obs"][i],
                               ev["amplitude"][i], n_width_samples, scale, m_ro, batch=8192)
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
                       m_ro, RUN_LABELS, report)


def _resample_detected(pool, w, cal, rng, n):
    """Detected calibration systems (with the calibrated run mix), resampled by weight."""
    run = rng.choice(len(cal.amplitude), size=len(w), p=cal.run_probability)
    amp = np.asarray(cal.amplitude)[run]
    rho_obs = amp * pool["base"] * (1.0 + pool["m_tot"] / cal.mass_rolloff) ** (-5.0 / 6.0) + pool["noise"]
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


def coverage_gate(draw: DrawDistribution, selection, event_truths, *, graph_nodes=None) -> dict:
    """Rule 6 on the built artifact: the draw support contains every node's population support."""
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
    passed = bool(ok_all and n_bad_grid == 0 and n_bad_events == 0 and n_bad_sel == 0)
    return {
        "pass": passed,
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


def build_mock(output_dir, *, kind: str = "closure_widthq", n_events: int = 259, n_pe: int = 4096,
               target_found: int = 1_500_000, seed: int = 20261010, delta_pe: bool = False,
               calibration: Calibration | None = None, n_pool: int | None = None, run_gates: bool = True,
               ensemble: int = 50, grid=C2_SLOPE_GRID, calibration_kwargs: Mapping | None = None,
               log=print) -> dict:
    """Generate one canonical-format mock pair (pe.h5, selection.h5) plus truth, gates and manifest."""
    from ..data import PosteriorCatalog, validate_pair
    from ..data.adapters.gwcat_v2 import basis_for_spin

    out = Path(output_dir).resolve()
    refuse_protected(out)
    out.mkdir(parents=True, exist_ok=True)
    cosmo = default_cosmology()
    draw = DrawDistribution(cosmo)
    if calibration is None:
        calibration = calibrate(draw, log=log, **dict(calibration_kwargs or {}))
    cal = calibration
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
            s, acc = sample_pe(rng, x_obs, pool["rho_obs"][j], amp, n_pe, cal.width_scale, cal.mass_rolloff)
        for k in cols:
            cols[k].append(s[k])
        log_ref.append(pe_log_prior_basis(s["m1_detector"], s["luminosity_distance"]))
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
                                     f"pi ∝ m1_det d_L^2 on the box {PE_BOX}"},
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
        ens_rng = np.random.default_rng(ens_ss)
        ens = []
        for _ in range(int(ensemble)):
            idx = ens_rng.choice(len(w), size=n_events, replace=True, p=w / w.sum())
            e = {k: pool[k][idx] for k in ("m1_detector", "q", "luminosity_distance", "chi_eff")}
            ens.append(delta_profile(e, selection, kind, grid, logp=logp_c2, log_xi_grid=log_xi_grid)[0])
        mles = np.array([p["mle"] for p in ens])
        truth_slope = prof["truth"]
        ens_summary = {
            "n_catalogs": int(ensemble), "n_events_each": n_events,
            "mle_mean": float(mles.mean()) if len(mles) else None,
            "mle_sd": float(mles.std(ddof=1)) if len(mles) > 1 else None,
            "coverage_dlnL_le_2": float(np.mean([p["truth_inside"] for p in ens])) if ens else None,
            "mles": mles.round(4).tolist(),
        }
        if len(mles) > 1:
            se = ens_summary["mle_sd"] / math.sqrt(len(mles))
            ens_summary["mle_mean_minus_truth_over_se"] = float((mles.mean() - truth_slope) / se)
            ens_summary["pass"] = bool(abs(mles.mean() - truth_slope) <= 3.0 * se
                                       and ens_summary["coverage_dlnL_le_2"] >= 0.8)
        gates["i_delta_pe_profile"] = {
            "parameter": C2_SLOPE, "model": "C2 (other hyperparameters at the truth)",
            "this_catalog": prof, "ensemble": ens_summary,
            "pass": bool(prof["truth_inside"] and ens_summary.get("pass", True)),
        }
        if not delta_pe:
            gates["pe_profile_reported"] = pe_profile(posterior, selection, kind, grid, log_xi_grid, logp_c2)
        gates["ii_coverage"] = coverage_gate(draw, selection, ev_truth)
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
        gates["b0_pass"] = bool(gates["i_delta_pe_profile"]["pass"] and gates["ii_coverage"]["pass"]
                                and gates["iii_g12"]["pass"])
        (out / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
        log(f"gates: b0_pass {gates['b0_pass']} (delta {gates['i_delta_pe_profile']['pass']}, coverage "
            f"{gates['ii_coverage']['pass']}, G12 {gates['iii_g12']['pass']})")

    files = {name: sha256_file(out / name) for name in ("pe.h5", "selection.h5", "mock_truth.json",
                                                        "calibration.json")}
    if run_gates:
        files["gates.json"] = sha256_file(out / "gates.json")
    manifest = {
        "format_version": MOCK_FORMAT_VERSION, "kind": kind, "delta_pe": bool(delta_pe),
        "status": "STAGING MOCK (not frozen)", "code_commit": _git_commit(),
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
                            "pi_PE(theta); pi_PE ∝ m1_det d_L^2 on PE_BOX",
            "pe_box": PE_BOX, "snr_threshold": SNR_THRESHOLD, "spin_snr_slope": SPIN_SNR_SLOPE,
            "draw_distribution": {"support": draw.support(), "proxy_fraction": DRAW_PROXY_FRACTION,
                                  "proxy": DRAW_PROXY},
        },
        "calibration": cal.to_dict(),
        "gates_b0_pass": gates.get("b0_pass"),
        "files_sha256": files,
    }
    (out / "mock_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
