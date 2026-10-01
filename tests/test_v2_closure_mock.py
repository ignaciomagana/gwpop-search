"""v2 closure / null mock generator (PILOT_PLAN (b), mock-data-dag rules), small sizes."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from scipy.stats import chisquare  # noqa: E402

from gwpop_search.validation import v2_mock as M  # noqa: E402

#: a fixed small calibration (no calibrate() call): realistic magnitudes
CAL = M.Calibration(
    amplitude=(10.1, 12.3, 19.7, 20.6, 28.3, 29.4),
    run_time_yr=(0.22, 0.56, 0.62, 0.49, 0.55, 0.61),
    width_scale=(0.21, 0.33, 0.17),
    mass_rolloff=409.0,
    report={"draw_detection_fraction": 0.05, "seed": 0},
)


@pytest.fixture(scope="module")
def draw():
    return M.DrawDistribution(M.default_cosmology())


def test_truth_models():
    spec_c, hp_c = M.truth_model("closure_widthq")
    spec_n, hp_n = M.truth_model("null")
    from gwpop_search.grammar.v2 import v2_root_model_spec

    assert spec_n.model_hash == v2_root_model_spec().model_hash
    assert spec_c.model_hash == M.c2_model_spec().model_hash != spec_n.model_hash
    assert hp_c[M.C2_SLOPE] == -2.2 and M.C2_SLOPE not in hp_n
    sigma = lambda q: math.exp(hp_c["chi_log_sigma"] + hp_c[M.C2_SLOPE] * (q - 1.0))  # noqa: E731
    assert sigma(1.0) == pytest.approx(0.05) and sigma(0.5) == pytest.approx(0.1502, abs=1e-3)
    assert math.exp(hp_n["chi_log_sigma"]) == pytest.approx(0.09)
    # LVK mapping: lam_u = 1 - exp(-lam), mlow_2_frac = (mlow_2 - 3) / (mlow_1 - 3)
    assert hp_n["lam_u_pl"] == pytest.approx(-math.expm1(-0.40))
    assert hp_n["mlow_2_frac"] == pytest.approx((3.46 - 3.0) / (4.49 - 3.0))
    with pytest.raises(ValueError):
        M.truth_model("other")


def test_draw_density_is_normalised_with_exact_jacobian(draw):
    """E_draw[p_pop / p_draw] = 1 in the detector basis (normalised population inside the draw support)."""
    spec, hp = M.truth_model("null")
    rng = np.random.default_rng(3)
    s = draw.sample(rng, 400_000)
    d_l = draw.cosmo.dL_of_z(s["z"])
    cols = {"m1_detector": s["m1_source"] * (1 + s["z"]), "q": s["q"], "luminosity_distance": d_l,
            "chi_eff": s["chi_eff"]}
    lw = M.LogDensity(spec)(cols, hp) - draw.log_density_detector(s["m1_source"], s["q"], s["z"], s["chi_eff"])
    w = np.exp(lw)
    ess = w.sum() ** 2 / np.sum(w ** 2)
    assert w.mean() == pytest.approx(1.0, abs=4.0 / math.sqrt(ess))
    # the source density integrates to one on the support (uniform-box importance estimate)
    n = 400_000
    m1 = rng.uniform(3, 300, n)
    q = rng.uniform(0.001, 1, n)
    z = rng.uniform(1e-6, 1.9, n)
    c = rng.uniform(-1, 1, n)
    vol = 297 * 0.999 * 1.9 * 2
    p = np.exp(draw.log_density_source(m1, q, z, c))
    assert p.mean() * vol == pytest.approx(1.0, rel=0.03)


def test_detection_is_on_data_and_shared_machinery(draw):
    rng = np.random.default_rng(4)
    det = M.simulate_detections(draw, CAL, rng, 300_000, chunk=100_000)
    assert np.all(det["rho_obs"] > M.SNR_THRESHOLD)
    amp = np.asarray(CAL.amplitude)[det["run_index"]]
    g = M.snr_unit(det["m1_detector"], det["q"], det["luminosity_distance"], det["chi_eff"], CAL.mass_rolloff)
    np.testing.assert_allclose(det["rho_opt"], amp * det["theta"] * g)
    # rule 1: some detected systems have optimal SNR below threshold (noise up-fluctuations)
    assert np.any(det["rho_opt"] < M.SNR_THRESHOLD)
    # run labels follow T_k / T among draws, louder runs are over-represented among detections
    counts = np.bincount(det["run_index"], minlength=6)
    assert counts[5] / counts[0] > CAL.run_time_yr[5] / CAL.run_time_yr[0]


@np.errstate(invalid="ignore", divide="ignore")
def _reference_moments(x_obs, rho_obs, amp, box, n=1_000_000, seed=7):
    """Importance-sampling posterior moments from the un-factorised likelihood x prior.

    Proposal: y_r = (ln m1, q, chi) ~ N(x_obs, 4 Sigma), ln d_L uniform; the amplitude
    likelihood is marginalised over Theta by brute-force Monte Carlo on a grid in s.
    """
    rng = np.random.default_rng(seed)
    cov = 4.0 * M.pe_covariance(rho_obs, CAL.width_scale)
    yr = rng.multivariate_normal(x_obs, cov, size=n)
    (d_lo, d_hi) = box["luminosity_distance"]
    lnd = rng.uniform(math.log(d_lo), math.log(d_hi), n)
    m1, q, c, d = np.exp(yr[:, 0]), yr[:, 1], yr[:, 2], np.exp(lnd)
    inside = ((m1 >= box["m1_detector"][0]) & (m1 <= box["m1_detector"][1]) & (q >= box["q"][0])
              & (q <= box["q"][1]) & (c >= box["chi_eff"][0]) & (c <= box["chi_eff"][1]))
    diff = yr - x_obs
    log_prop = -0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
    # prior ∝ m1 d^2 in the basis -> m1^2 d^3 in (ln m1, q, ln d, chi)
    logw = 2 * np.log(m1) + 3 * lnd - log_prop
    base = M.pe_log_likelihood_unmarginalised(x_obs, rho_obs, amp, CAL.width_scale, CAL.mass_rolloff,
                                              m1_det=m1, q=q, d_l=d, chi=c, theta=0.0) + 0.5 * rho_obs ** 2
    theta = M.projection_factor(rng, 200_000)
    s_grid = np.exp(np.linspace(math.log(rho_obs / 3), math.log(rho_obs * 400), 3000))
    l_grid = np.array([np.mean(np.exp(-0.5 * (rho_obs - sg * theta) ** 2)) for sg in s_grid])
    s = np.exp(M._log_c(m1, q, c, amp, CAL.mass_rolloff)) / d
    amp_like = np.interp(np.log(s), np.log(s_grid), l_grid, left=0.0, right=l_grid[-1])
    lw = np.where(inside, logw + base + np.log(amp_like + 1e-300), -np.inf)
    w = np.exp(lw - lw.max())
    y = np.stack([np.log(m1), q, lnd, c], 1)
    mean = (w[:, None] * y).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (y - mean) ** 2).sum(0) / w.sum())
    return mean, sd, w.sum() ** 2 / np.sum(w ** 2)


def test_pe_sampler_matches_unfactorised_posterior():
    """The factorised exact sampler reproduces the posterior of the raw likelihood x prior."""
    amp, m_ro = 20.0, CAL.mass_rolloff
    m1, q, chi, d_l, theta = 40.0, 0.7, 0.05, 800.0, 0.6
    rho_opt = theta * amp * M.snr_unit(m1, q, d_l, chi, m_ro)
    rho_obs = float(rho_opt + 0.4)
    x_obs = M.x_of(m1, q, chi) + np.array([0.02, -0.03, 0.01])
    box = {"m1_detector": (15.0, 100.0), "q": (0.2, 1.0), "luminosity_distance": (60.0, 6000.0),
           "chi_eff": (-0.6, 0.7)}
    s, acc = M.sample_pe(np.random.default_rng(11), x_obs, rho_obs, amp, 40_000, CAL.width_scale, m_ro,
                         box=box)
    assert acc > 0.01
    y = np.stack([np.log(s["m1_detector"]), s["q"], np.log(s["luminosity_distance"]), s["chi_eff"]], 1)
    ref_mean, ref_sd, ess = _reference_moments(x_obs, rho_obs, amp, box)
    assert ess > 2000
    np.testing.assert_allclose(y.mean(0), ref_mean, atol=0.06 * ref_sd.max(), rtol=0)
    assert np.all(np.abs(y.mean(0) - ref_mean) < 0.08 * ref_sd)
    np.testing.assert_allclose(y.std(0), ref_sd, rtol=0.08)


def test_pe_sbc_conditional_on_detection():
    """SBC: truths drawn from the PE prior, kept if rho_obs > 10 (a data cut), ranks uniform."""
    rng = np.random.default_rng(21)
    box = {"m1_detector": (20.0, 80.0), "q": (0.3, 1.0), "luminosity_distance": (200.0, 2500.0),
           "chi_eff": (-0.5, 0.5)}
    amp, n_post, ranks = 20.0, 99, []
    while len(ranks) < 160:
        m1 = math.sqrt(rng.uniform(box["m1_detector"][0] ** 2, box["m1_detector"][1] ** 2))
        q = rng.uniform(*box["q"])
        d = rng.uniform(box["luminosity_distance"][0] ** 3, box["luminosity_distance"][1] ** 3) ** (1 / 3)
        c = rng.uniform(*box["chi_eff"])
        th = M.projection_factor(rng, 1)[0]
        rho_obs = th * amp * M.snr_unit(m1, q, d, c, CAL.mass_rolloff) + rng.standard_normal()
        if rho_obs <= M.SNR_THRESHOLD:
            continue
        x_obs = M.observe(rng, M.x_of(m1, q, c), rho_obs, CAL.width_scale)
        s, _ = M.sample_pe(rng, x_obs, rho_obs, amp, n_post, CAL.width_scale, CAL.mass_rolloff, box=box,
                           batch=4096)
        truth = (m1, q, d, c)
        ranks.append([int(np.sum(s[k] < t)) for k, t in zip(("m1_detector", "q", "luminosity_distance",
                                                                "chi_eff"), truth)])
    ranks = np.asarray(ranks)
    for j in range(4):
        hist = np.bincount(ranks[:, j] // 10, minlength=10)
        assert chisquare(hist).pvalue > 1e-3, (j, hist)


def test_pe_log_prior_matches_box_density():
    rng = np.random.default_rng(5)
    box = M.PE_BOX
    # density ∝ m1 d^2 integrates to one on the box (importance check in log coordinates)
    n = 400_000
    m1 = np.exp(rng.uniform(math.log(box["m1_detector"][0]), math.log(box["m1_detector"][1]), n))
    d = np.exp(rng.uniform(math.log(box["luminosity_distance"][0]), math.log(box["luminosity_distance"][1]), n))
    vol = (math.log(box["m1_detector"][1] / box["m1_detector"][0]) * (box["q"][1] - box["q"][0])
           * math.log(box["luminosity_distance"][1] / box["luminosity_distance"][0]) * 2.0)
    integrand = np.exp(M.pe_log_prior_basis(m1, d)) * m1 * d  # Jacobian to (ln m1, ln d)
    assert integrand.mean() * vol == pytest.approx(1.0, rel=0.05)
    assert np.isneginf(M.pe_log_prior_basis(np.array([1.0]), np.array([100.0]))[0])


def test_refuses_protected_paths(tmp_path):
    with pytest.raises(SystemExit):
        M.refuse_protected(tmp_path / "frozen" / "x")
    with pytest.raises(SystemExit):
        M.refuse_protected(tmp_path / "gwcat" / "build" / "v2r2" / "x")


@pytest.mark.parametrize("delta_pe", [False, True])
def test_small_build_end_to_end(tmp_path, delta_pe):
    from gwpop_search.data import PosteriorCatalog, SelectionCatalog

    out = tmp_path / ("delta" if delta_pe else "mock")
    grid = tuple(np.round(np.arange(-5.0, 1.01, 0.25), 6))
    man = M.build_mock(out, kind="closure_widthq", n_events=30, n_pe=256, target_found=20_000, seed=5,
                       delta_pe=delta_pe, calibration=CAL, ensemble=4, grid=grid, log=lambda *_: None)
    pe = PosteriorCatalog.from_hdf5(out / "pe.h5")
    sel = SelectionCatalog.from_hdf5(out / "selection.h5")
    assert pe.n_events == 30 and pe.offsets[-1] == 30 * (1 if delta_pe else 256)
    assert sel.mode.value == "raw_draw" and sel.campaigns[0].n_draw == man["sizes"]["n_draw"]
    assert sel.campaigns[0].observing_time_yr == pytest.approx(sum(CAL.run_time_yr))
    assert tuple(pe.basis.coordinates) == ("m1_detector", "q", "luminosity_distance", "chi_eff")
    assert np.all(np.isfinite(pe.log_ref_density))
    for name, digest in man["files_sha256"].items():
        assert M.sha256_file(out / name) == digest
    truth = json.loads((out / "mock_truth.json").read_text())
    assert truth["c2_slope_truth"] == -2.2 and len(truth["events"]) == 30
    assert all(e["rho_obs"] > M.SNR_THRESHOLD for e in truth["events"])
    if delta_pe:  # delta PE is the truth itself
        np.testing.assert_allclose(pe.samples["q"], [e["q"] for e in truth["events"]])
        np.testing.assert_allclose(pe.samples["luminosity_distance"],
                                   [e["luminosity_distance"] for e in truth["events"]])
    gates = json.loads((out / "gates.json").read_text())
    assert gates["ii_coverage"]["pass"] and gates["iii_g12"]["pass"]
    assert gates["ii_coverage"]["support_grid_points_with_zero_draw_density"] == 0
    prof = gates["i_delta_pe_profile"]["this_catalog"]
    assert prof["truth"] == -2.2 and len(prof["lnL"]) == len(grid)
    if not delta_pe:
        assert "pe_profile_reported" in gates
    # the selection estimator in the gate equals the pipeline's (numpy HBI) ln xi
    from gwpop_search.hbi import evaluate_selection

    spec, hp = M.truth_model("closure_widthq")
    logp = M.LogDensity(spec)
    ref = evaluate_selection(sel, logp.model, hp).log_exposure
    assert M.selection_log_xi(logp, sel, hp) == pytest.approx(ref, rel=1e-10, abs=1e-10)


def test_calibration_reproduces_targets(draw):
    cal = M.calibrate(draw, n_pool=150_000, n_width_events=12, n_width_samples=256, width_iterations=1,
                      log=lambda *_: None)
    assert sum(cal.run_time_yr) == pytest.approx(M.T_TOTAL_YR)
    for label in M.RUN_LABELS:
        rep = cal.report[label]
        assert rep["achieved"]["z"][1] == pytest.approx(M.CANONICAL_RUN_TARGETS[label]["z"][1], rel=0.03)
    assert cal.report["pooled_m1_source_q90"]["achieved"] == pytest.approx(M.CANONICAL_POOLED_M1_Q90, rel=0.05)
    assert list(cal.amplitude) == sorted(cal.amplitude) or cal.amplitude[-1] > cal.amplitude[0]
    rt = M.Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert rt.amplitude == cal.amplitude and rt.mass_rolloff == cal.mass_rolloff
