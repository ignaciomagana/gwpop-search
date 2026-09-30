"""G2a: the ported v2 BP2P root against the LVK GWTC-5.0 ``rates_on_grids``.

Port of populations-research ``popres/tests/test_rates_on_grids.py`` (gate
G2a, gates/G2a_grids.md) to ``gwpop_search.models``. For draw i of a tied
release file (sha256 checked), with hyperparameters Lambda_i and local rate R0_i:

  mass_1      dR/dm1 = R0 trapz_q p(m1, q)    on m1 = linspace(3, M, 1000),
  mass_ratio  dR/dq  = R0 trapz_m1 p(m1, q)   q = linspace(0.001, 1, 500)
  redshift    R(z)   = R0 psi(z)              on linspace(1e-6, 1.9, 2500)

Pass: max relative deviation < 1e-6 at every node where the released value
exceeds 1e-6 of its maximum, over 256 draws per file (seed 20260926). The
model is built with the LVK run's normalisation range (minimum_mass = 3,
maximum_mass = M; var_cut 4 has M = 200 with the power law still normalised to
mmax = 300) and q_floor = 0.001, which makes the v2 pairing grid the LVK one.

Skipped when the release directory is not mounted (override with
``GWPOP_LVK_RELEASE_DIR``).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
h5py = pytest.importorskip("h5py")
import jax.numpy as jnp  # noqa: E402

from gwpop_search.grammar import v2_root_model_spec  # noqa: E402
from gwpop_search.models import (  # noqa: E402
    compile_model_spec,
    lvk_default_coordinates,
    lvk_default_physical,
    v2_mass_logpdf,
)
from gwpop_search.models.components import (  # noqa: E402
    BrokenPowerLawPeaksMass,
    LVKMassGrid,
    TaperedPowerLawPairing,
    madau_dickinson_log_psi,
)

RELEASE_DIR = Path(os.environ.get(
    "GWPOP_LVK_RELEASE_DIR",
    "/hildafs/home/magana/tmp_ondemand_hildafs_phy220048p_symlink/share/"
    "LVK_population_analyses/popsummary_files",
))
_TP = ("mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_{z}_magnitude_iid_spin_"
       "magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5")
#: key -> (file, sha256 tied in gates/G2a_grids.md section 1)
FILES = {
    "default": ("gwtc5_updated_default_mmax_" + _TP.format(z="PowerLawRedshift"),
                "aa159c4259ccdbb9140d4ab65382095f4744ef42d7531dc727f19e75d19ef048"),
    "madau_dickinson": ("gwtc5_updated_madau_dickinson_mmax_" + _TP.format(z="MadauDickinsonRedshift"),
                        "a00aa97e6be8a03a6900ef1b076e01788c8c49e48e98bc33bea48df885dc9fb3"),
    "var_4": ("gwtc5_updated_default_var_4_" + _TP.format(z="PowerLawRedshift"),
              "8e592be41f689c12214cdc7e56d58baf597e1d56698cd9b0d43db7476f55cdbd"),
}
LVK_MASS_KEYS = ("alpha_1", "alpha_2", "break_mass", "mlow_1", "delta_m_1", "mlow_2", "delta_m_2",
                 "lam_0", "lam_1", "mpp_1", "sigpp_1", "mpp_2", "sigpp_2", "beta")
N_DRAWS = 256
SEED = 20260926
TOL = 1e-6
FLOOR = 1e-6
LVK_Q_FLOOR = 0.001
_trapz = getattr(np, "trapezoid", None) or np.trapz

pytestmark = pytest.mark.skipif(not RELEASE_DIR.is_dir(), reason="LVK GWTC-5 release not mounted")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(f):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["hyperparameters"]]


def _maxdev(model, release):
    ok = release > FLOOR * release.max()
    return float(np.max(np.abs(model[ok] / release[ok] - 1.0)))


def _load(key):
    name, sha = FILES[key]
    path = RELEASE_DIR / name
    assert _sha256(path) == sha, f"{name} is not the tied release file"
    with h5py.File(path, "r") as f:
        names = _names(f)
        x = f["posterior/hyperparameter_samples"][:]
        rows = np.sort(np.random.default_rng(SEED).choice(x.shape[0], size=min(N_DRAWS, x.shape[0]),
                                                          replace=False))
        grids = {k: (f[f"posterior/rates_on_grids/{k}/positions"][0],
                     f[f"posterior/rates_on_grids/{k}/rates"][rows, :])
                 for k in ("mass_1", "mass_ratio", "redshift")}
    return {n: i for i, n in enumerate(names)}, x, rows, grids


def _component_model(m_lo, m_hi):
    grid = LVKMassGrid(mmin=m_lo, mmax=m_hi, n_m1=1000, n_q=500, q_floor=LVK_Q_FLOOR, m1_grid="geomspace")
    mass = BrokenPowerLawPeaksMass("broken", ("p10", "p35"), grid, continuum_high=300.0)
    pairing = TaperedPowerLawPairing("constant", grid)

    def logp(m, q, p):
        lm, lq = jnp.log(m), jnp.log(q)
        terms = mass.log_component_terms(m, lm, p)
        lp_m1 = mass.log_prob_from_terms(m, terms, p)
        lp_q = pairing.log_prob(q, lq, m, lm, p, mass_terms=terms)
        return jnp.where(jnp.isneginf(lp_m1), -jnp.inf, lp_m1 + lp_q)

    return jax.jit(logp)


@pytest.mark.parametrize("key", ["default", "madau_dickinson", "var_4"])
def test_g2a_mass_and_redshift_grids_reproduce_the_lvk_release(key):
    col, x, rows, grids = _load(key)
    m, rm = grids["mass_1"]
    q, rq = grids["mass_ratio"]
    z, rz = grids["redshift"]
    assert m.size == 1000 and q.size == 500 and z.size == 2500
    assert q[0] == LVK_Q_FLOOR and q[-1] == 1.0
    mm, qq = np.meshgrid(m, q)
    fn = _component_model(float(m[0]), float(m[-1]))
    log1pz = np.log1p(z)
    dm1, dq, dz = [], [], []
    for j, i in enumerate(rows):
        p = lvk_default_physical({k: float(x[i, col[k]]) for k in LVK_MASS_KEYS})
        rate = float(x[i, col["rate"]])
        dens = np.exp(np.asarray(fn(jnp.asarray(mm), jnp.asarray(qq), p))) * rate
        dm1.append(_maxdev(_trapz(dens, q, axis=0), rm[j]))
        dq.append(_maxdev(_trapz(dens, m, axis=1), rq[j]))
        if key == "madau_dickinson":
            log_psi = np.asarray(madau_dickinson_log_psi(
                jnp.asarray(log1pz), x[i, col["gamma"]], x[i, col["kappa"]], x[i, col["z_peak"]]))
        else:
            log_psi = x[i, col["lamb"]] * log1pz
        dz.append(_maxdev(rate * np.exp(log_psi), rz[j]))
    assert len(rows) >= 200
    assert max(dm1) < TOL, (key, max(dm1), float(np.median(dm1)))
    assert max(dq) < TOL, (key, max(dq), float(np.median(dq)))
    assert max(dz) < TOL, (key, max(dz))


def test_g2a_declarative_root_coordinates_reproduce_the_release():
    """The sampled v2 coordinates (Dirichlet unit weights, mlow_2 fraction) map
    the LVK draws onto the same density through the compiled root."""
    col, x, rows, grids = _load("default")
    m, rm = grids["mass_1"]
    q, _ = grids["mass_ratio"]
    root = v2_root_model_spec()
    spec = replace(root, support={**root.support, "q_floor": LVK_Q_FLOOR})
    model = compile_model_spec(spec)
    mm, qq = np.meshgrid(m, q)
    fn = jax.jit(lambda hp: v2_mass_logpdf(model, jnp.asarray(mm), jnp.asarray(qq), hp))
    devs = []
    for j, i in enumerate(rows[:32]):
        lvk = {k: float(x[i, col[k]]) for k in LVK_MASS_KEYS}
        assert 3.0 <= lvk["mlow_2"] <= lvk["mlow_1"]  # GWTC-5 Table 5: m2,low ~ U(3, m1,low)
        hp = lvk_default_coordinates(lvk)
        dens = np.exp(np.asarray(fn(hp))) * float(x[i, col["rate"]])
        devs.append(_maxdev(_trapz(dens, q, axis=0), rm[j]))
    assert max(devs) < TOL, max(devs)
