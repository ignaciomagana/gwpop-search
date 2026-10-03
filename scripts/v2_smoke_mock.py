#!/usr/bin/env python
"""Tiny DAG-consistent MOCK catalog for the v2 end-to-end CPU smoke test.

    python scripts/v2_smoke_mock.py --output-dir <dir> [--n-events 12] [--seed 0]

Writes ``<dir>/pe.h5`` and ``<dir>/selection.h5`` in the canonical
sky-marginal gwcat-v2 basis (the basis of the approved v2r2 canonical staging
products) plus ``<dir>/mock_truth.json``. This is a pipeline smoke fixture, not
a validation catalog: it is small (few events, thinned injections) and its only
job is to exercise enumeration -> tapered dynesty -> claim table on CPU.

DAG (mock-data-dag): one draw distribution ``p_draw`` in the source frame
(m1 log-uniform on [mmin, mmax] = [3, 300], q uniform on [q_floor, 1] =
[0.001, 1], z ~ dVc/dz/(1+z) on [0, zmax = 1.9], chi_eff uniform on [-1, 1];
the bounds are read from the v2 model support and asserted to cover it) is
mapped to the density basis
(m1_detector, q, luminosity_distance, chi_eff) with its exact Jacobian. The
same deterministic detection rule (a chirp-mass/distance SNR proxy > 10) is
applied to the injections and to the events. Events are drawn from the v2 root
R0 at the truth below by importance resampling a separate draw pool with
weights p_pop / p_draw (same density basis, same cosmology as the model), then
the detection rule is applied. PE: Gaussian likelihood in the basis
coordinates around a noisy observation (observation = truth + noise), uniform
PE prior on a box, so the posterior is the truncated Gaussian and
``log_ref_density`` is the log of the uniform prior density.

SMOKE ONLY -- not for the step-6b mock closure. Known departures from the
mock-data-dag rules, acceptable for a pipeline smoke test only: the PE widths
scale with the true values (``sigma ~ w * truth``, a misspecified Gaussian
likelihood), and detection is a deterministic step on the true parameters with
no noise shared with the PE. The closure mock needs a noisy detection
statistic shared with the PE likelihood and truth-independent PE widths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Truth in sampled v2 root coordinates (LVK-like BP2P; lam = 0.4/0.5/0.1).
TRUTH = {
    "alpha_1": 1.5, "alpha_2": 5.4, "m_break": 37.5,
    "mu_p10": 9.9, "sigma_p10": 0.8, "mu_p35": 33.0, "sigma_p35": 4.0,
    "lam_u_pl": float(-np.expm1(-0.4)), "lam_u_p10": float(-np.expm1(-0.5)),
    "lam_u_p35": float(-np.expm1(-0.1)),
    "mlow_1": 5.0, "delta_m_1": 4.0, "mlow_2_frac": 0.5, "delta_m_2": 4.0,
    "beta": 1.1, "kappa": 3.0, "chi_mu": 0.06, "chi_log_sigma": float(np.log(0.1)),
}

def _support():
    from gwpop_search.grammar.v2 import V2_SUPPORT

    return V2_SUPPORT


# The draw distribution covers the v2 population support exactly (mock-data-dag:
# the draw support must contain the population support).
M_LO, M_HI = float(_support()["mmin"]), float(_support()["mmax"])
Q_LO = float(_support()["q_floor"])
Z_MAX = float(_support()["zmax"])
SNR_THRESHOLD = 10.0
PE_BOX = {  # uniform PE prior box in the density basis
    "m1_detector": (2.0, 600.0),
    "q": (0.02, 1.0),
    "luminosity_distance": (1.0, 20000.0),
    "chi_eff": (-1.0, 1.0),
}
PE_WIDTH = {"m1_detector": 0.08, "q": 0.12, "luminosity_distance": 0.25, "chi_eff": 0.08}


class DrawDistribution:
    """Source-frame draw distribution and its density in the detector basis."""

    def __init__(self, cosmology):
        self.cosmology = cosmology
        z = np.linspace(0.0, Z_MAX, 4001)
        pz = np.asarray(cosmology.dVc_dz(z), dtype=float) / (1.0 + z)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pz[1:] + pz[:-1]) * np.diff(z))])
        self._z, self._cdf = z, cdf / cdf[-1]
        self._pz_norm = float(cdf[-1])

    def sample(self, rng, n):
        m1 = np.exp(rng.uniform(np.log(M_LO), np.log(M_HI), n))
        q = rng.uniform(Q_LO, 1.0, n)
        z = np.interp(rng.uniform(0.0, 1.0, n), self._cdf, self._z)
        chi = rng.uniform(-1.0, 1.0, n)
        return m1, q, z, chi

    def log_density_detector(self, m1_source, z):
        """ln p_draw in dm1_detector dq ddL dchi_eff (q and chi_eff uniform)."""
        log_m1 = -np.log(m1_source) - np.log(np.log(M_HI / M_LO))
        log_q = -np.log(1.0 - Q_LO)
        log_z = (np.log(np.asarray(self.cosmology.dVc_dz(z), dtype=float)) - np.log1p(z)
                 - np.log(self._pz_norm))
        log_chi = -np.log(2.0)
        jac = -np.log1p(z) - np.log(np.asarray(self.cosmology.ddL_dz(z), dtype=float))
        return log_m1 + log_q + log_z + log_chi + jac


def snr_proxy(m1_source, q, z, d_l):
    """Deterministic detection statistic (no noise): chirp-mass/distance scaling."""
    m2 = q * m1_source
    mc_det = (m1_source * m2) ** 0.6 / (m1_source + m2) ** 0.2 * (1.0 + z)
    return 9.0 * (mc_det / 10.0) ** (5.0 / 6.0) * (1000.0 / d_l)


def build(output_dir: Path, *, n_events: int, n_pe: int, n_draw: int, seed: int) -> dict:
    import jax

    jax.config.update("jax_enable_x64", True)
    from gwpop_search.data import Campaign, PosteriorCatalog, SelectionCatalog, SelectionMode
    from gwpop_search.data.adapters.gwcat_v2 import basis_for_spin
    from gwpop_search.grammar.v2 import v2_root_model_spec
    from gwpop_search.models import compile_model_spec

    rng = np.random.default_rng(seed)
    root = v2_root_model_spec()
    support = root.support
    if not (M_LO <= float(support["mmin"]) and M_HI >= float(support["mmax"])
            and Q_LO <= float(support["q_floor"]) and Z_MAX >= float(support["zmax"])):
        raise SystemExit(f"mock draw support [{M_LO}, {M_HI}] x [{Q_LO}, 1] x [0, {Z_MAX}] does not "
                         f"cover the model support {dict(support)}")
    model = compile_model_spec(root)
    cosmo = model.cosmology
    draw = DrawDistribution(cosmo)
    basis = basis_for_spin("chieff", sky_marginal=True)

    # -- injections (thinned: only the detected rows are kept) --------------
    m1, q, z, chi = draw.sample(rng, n_draw)
    d_l = np.asarray(cosmo.dL_of_z(z), dtype=float)
    found = snr_proxy(m1, q, z, d_l) > SNR_THRESHOLD
    sel_samples = {
        "m1_detector": m1[found] * (1.0 + z[found]),
        "q": q[found],
        "luminosity_distance": d_l[found],
        "chi_eff": chi[found],
        "m1_source": m1[found],
        "z": z[found],
    }
    selection = SelectionCatalog(
        samples=sel_samples,
        log_draw_density=draw.log_density_detector(m1[found], z[found]),
        campaign_id=np.full(int(found.sum()), "MOCK"),
        campaigns=(Campaign("MOCK", n_draw=int(n_draw), observing_time_yr=1.0),),
        basis=basis,
        mode=SelectionMode.RAW_DRAW,
        metadata={"mock": True, "z_max": Z_MAX, "sky_marginal": True,
                  "detection_rule": f"snr_proxy > {SNR_THRESHOLD}", "seed": int(seed)},
    )

    # -- events: resample a draw pool to R0 truth, then detect --------------
    pool = 400_000
    pm1, pq, pz, pchi = draw.sample(rng, pool)
    pdl = np.asarray(cosmo.dL_of_z(pz), dtype=float)
    det = {"m1_detector": pm1 * (1.0 + pz), "q": pq, "luminosity_distance": pdl, "chi_eff": pchi}
    log_pop = np.asarray(model({k: np.asarray(v) for k, v in det.items()}, TRUTH), dtype=float)
    log_w = log_pop - draw.log_density_detector(pm1, pz)
    w = np.exp(log_w - np.max(log_w[np.isfinite(log_w)]))
    w[~np.isfinite(w)] = 0.0
    detectable = snr_proxy(pm1, pq, pz, pdl) > SNR_THRESHOLD
    w_det = w * detectable
    ess_pool = float(w_det.sum() ** 2 / np.sum(w_det ** 2))
    picks = rng.choice(pool, size=n_events, replace=False, p=w_det / w_det.sum())

    names, cols, log_ref = [], {k: [] for k in (*PE_BOX, "m1_source", "z")}, []
    log_prior = -sum(np.log(hi - lo) for lo, hi in PE_BOX.values())
    for i, j in enumerate(picks):
        truth = {k: float(det[k][j]) for k in PE_BOX}
        sig = {k: PE_WIDTH[k] * (truth[k] if k in ("m1_detector", "luminosity_distance") else 1.0)
               for k in PE_BOX}
        obs = {k: truth[k] + sig[k] * rng.normal() for k in PE_BOX}
        draws = {}
        for k in PE_BOX:
            lo, hi = PE_BOX[k]
            vals = np.empty(0)
            while vals.size < n_pe:  # truncated Gaussian by rejection
                cand = obs[k] + sig[k] * rng.normal(size=4 * n_pe)
                vals = np.concatenate([vals, cand[(cand > lo) & (cand < hi)]])
            draws[k] = vals[:n_pe]
        zz = np.asarray(cosmo.z_of_dL(draws["luminosity_distance"]), dtype=float)
        for k in PE_BOX:
            cols[k].append(draws[k])
        cols["m1_source"].append(draws["m1_detector"] / (1.0 + zz))
        cols["z"].append(zz)
        log_ref.append(np.full(n_pe, log_prior))
        names.append(f"MOCK{i:03d}")
    posterior = PosteriorCatalog(
        event_names=tuple(names),
        offsets=np.arange(n_events + 1) * n_pe,
        samples={k: np.concatenate(v) for k, v in cols.items()},
        log_ref_density=np.concatenate(log_ref),
        basis=basis,
        metadata={"mock": True, "sky_marginal": True, "z_max": Z_MAX, "seed": int(seed)},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior.to_hdf5(output_dir / "pe.h5")
    selection.to_hdf5(output_dir / "selection.h5")
    truth = {
        "format_version": "gwpop-search-v2-smoke-mock-1.0",
        "root_model_hash": root.model_hash,
        "truth": TRUTH,
        "n_events": n_events, "n_pe": n_pe, "n_draw": n_draw,
        "n_found_injections": int(found.sum()), "seed": int(seed),
        "pool_detected_ess": ess_pool,
        "basis_identity": basis.identity,
    }
    (output_dir / "mock_truth.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")
    return truth


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-events", type=int, default=12)
    parser.add_argument("--n-pe", type=int, default=256)
    parser.add_argument("--n-draw", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    if "frozen" in out.parts:
        raise SystemExit("refusing to write a mock under a frozen/ directory")
    print(json.dumps(build(out, n_events=args.n_events, n_pe=args.n_pe, n_draw=args.n_draw,
                           seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
