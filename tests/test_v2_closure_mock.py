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


#: the same detection calibration with the generator-2.0 PE structure
CAL_2 = M.Calibration(
    amplitude=CAL.amplitude, run_time_yr=CAL.run_time_yr, width_scale=(0.30, 0.33, 0.23),
    mass_rolloff=CAL.mass_rolloff, report=CAL.report, pe_structure=M.default_pe_structure(),
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
    # C2 pivots at q = 0.7 (operator decision 2026-10-02): chi_log_sigma is ln sigma(0.7); the
    # physical truth is unchanged (sigma(1) = 0.05, slope -2.2)
    assert spec_c.chieff.options["q_pivot"] == 0.7
    sigma = lambda q: math.exp(hp_c["chi_log_sigma"] + hp_c[M.C2_SLOPE] * (q - 0.7))  # noqa: E731
    assert sigma(1.0) == pytest.approx(0.05) and sigma(0.5) == pytest.approx(0.1502, abs=1e-3)
    assert sigma(0.7) == pytest.approx(0.0967, abs=1e-4)
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
    # the source density integrates to one on the support (box importance estimate, ln m1 uniform)
    n = 800_000
    lnm = rng.uniform(math.log(3), math.log(300), n)
    q = rng.uniform(0.001, 1, n)
    z = rng.uniform(1e-6, 1.9, n)
    c = rng.uniform(-1, 1, n)
    vol = math.log(100.0) * 0.999 * 1.9 * 2
    p = np.exp(draw.log_density_source(np.exp(lnm), q, z, c)) * np.exp(lnm)
    assert p.mean() * vol == pytest.approx(1.0, rel=0.03)
    # sampler and density agree: E_draw[u(x) / p_draw(x)] = 1 for the uniform box density u
    # (the weights are bounded because the broad component has full support)
    lnu = -math.log(vol) - np.log(s["m1_source"])
    ratio = np.exp(lnu - draw.log_density_source(s["m1_source"], s["q"], s["z"], s["chi_eff"]))
    assert ratio.mean() == pytest.approx(1.0, abs=5.0 * ratio.std() / math.sqrt(ratio.size))
    # the proxy makes the injection weights at the truth nearly constant: high ESS fraction
    assert ess / w.size > 0.3


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
def _reference_moments(x_obs, rho_obs, amp, box, n=1_000_000, seed=7, power=5.0 / 6.0, m_ro=None, sharp=1.0):
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
    m_ro = CAL.mass_rolloff if m_ro is None else m_ro
    base = M.pe_log_likelihood_unmarginalised(x_obs, rho_obs, amp, CAL.width_scale, m_ro,
                                              m1_det=m1, q=q, d_l=d, chi=c, theta=0.0,
                                              rolloff_power=power, rolloff_sharpness=sharp) + 0.5 * rho_obs ** 2
    theta = M.projection_factor(rng, 200_000)
    s_grid = np.exp(np.linspace(math.log(rho_obs / 3), math.log(rho_obs * 400), 3000))
    l_grid = np.array([np.mean(np.exp(-0.5 * (rho_obs - sg * theta) ** 2)) for sg in s_grid])
    s = np.exp(M._log_c(m1, q, c, amp, m_ro, power, sharp)) / d
    amp_like = np.interp(np.log(s), np.log(s_grid), l_grid, left=0.0, right=l_grid[-1])
    lw = np.where(inside, logw + base + np.log(amp_like + 1e-300), -np.inf)
    w = np.exp(lw - lw.max())
    y = np.stack([np.log(m1), q, lnd, c], 1)
    mean = (w[:, None] * y).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (y - mean) ** 2).sum(0) / w.sum())
    return mean, sd, w.sum() ** 2 / np.sum(w ** 2)


@pytest.mark.parametrize("power,m_ro,sharp", [(5.0 / 6.0, CAL.mass_rolloff, 1.0), (2.5, 70.0, 3.0)])
def test_pe_sampler_matches_unfactorised_posterior(power, m_ro, sharp):
    """The factorised exact sampler reproduces the posterior of the raw likelihood x prior."""
    amp = 20.0 if power < 1 else 40.0
    m1, q, chi, d_l, theta = 40.0, 0.7, 0.05, 800.0, 0.6
    rho_opt = theta * amp * M.snr_unit(m1, q, d_l, chi, m_ro, power, sharp)
    rho_obs = float(rho_opt + 0.4)
    x_obs = M.x_of(m1, q, chi) + np.array([0.02, -0.03, 0.01])
    box = {"m1_detector": (15.0, 100.0), "q": (0.2, 1.0), "luminosity_distance": (60.0, 6000.0),
           "chi_eff": (-0.6, 0.7)}
    s, acc = M.sample_pe(np.random.default_rng(11), x_obs, rho_obs, amp, 40_000, CAL.width_scale, m_ro,
                         box=box, rolloff_power=power, rolloff_sharpness=sharp)
    assert acc > 0.01
    y = np.stack([np.log(s["m1_detector"]), s["q"], np.log(s["luminosity_distance"]), s["chi_eff"]], 1)
    ref_mean, ref_sd, ess = _reference_moments(x_obs, rho_obs, amp, box, power=power, m_ro=m_ro, sharp=sharp)
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
    cal = CAL if delta_pe else CAL_2
    man = M.build_mock(out, kind="closure_widthq", n_events=30, n_pe=256, target_found=20_000, seed=5,
                       delta_pe=delta_pe, calibration=cal, ensemble=4, ensemble_n_pe=64, grid=grid,
                       log=lambda *_: None)
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
    assert all(gates["ii_coverage"]["pe_box_contains_population_support"].values())
    prof = gates["delta_pe_profile_reported"]["this_catalog"]
    assert prof["truth"] == -2.2 and len(prof["lnL"]) == len(grid)
    gi = gates["i_noisy_pe_ensemble"]
    assert man["format_version"] == M.MOCK_FORMAT_VERSION and man["generator_version"] == "2.0"
    assert man["pe_generator_version"] == ("1.0" if delta_pe else "2.0")
    if not delta_pe:
        names = gi["model"]["parameters"]
        assert names == ["chi_mu", "chi_log_sigma", "chi_log_sigma_q_slope"]
        assert gi["ensemble"]["n_catalogs"] == 4 and gi["ensemble"]["n_pe"] == 64
        assert len(gi["ensemble"]["estimates"]) == 4 and gi["realised"]["n_pe"] == 256
        assert gi["pass"] == gi["unbiased"]["pass"]
        assert gi["tail_catalog"] == (not all(v["pass"] for v in gi["position"].values()))
        assert gi["ensemble"]["pe_model"].startswith("2.0")
        dec = M.ensemble_decision(gi["realised"]["values"], gi["ensemble"]["estimates"],
                                  [gi["model"]["truth"][n] for n in names], names)
        assert dec["tail_catalog"] == gi["tail_catalog"]
        assert "chi_log_sigma_at_q1" in gi["derived_reported"]
        assert man["dag"]["pe_structure"] == CAL_2.pe_structure.to_dict()
    else:
        assert gi["run"] is False
    # the PE box ends at d_L(z_max): no PE sample above the declared z_max
    assert pe.metadata["z_max"] == M.DRAW_ZMAX
    assert np.nanmax(pe.samples["z"]) <= M.DRAW_ZMAX + 1e-9
    assert man["software_versions"]["numpy"] == np.__version__
    assert man["calibration_source"] == {"provided_in_memory": True}
    if not delta_pe:
        assert "pe_profile_reported" in gates
        # sigma^2 at the truth: the gate's NumPy value is the JAX likelihood's variance
        tv = gates["iv_taper_at_truth"]
        from gwpop_search.hbi import HBIConfig
        from gwpop_search.hbi.jax_backend import build_shape_log_likelihood_components
        from gwpop_search.inference.v2_numerics import v2_variance_taper
        from gwpop_search.models import compile_model_spec

        spec_t, hp_t = M.truth_model("closure_widthq")
        comps = build_shape_log_likelihood_components(
            pe, sel, compile_model_spec(spec_t),
            config=HBIConfig(selection_chunk_size=None, variance_taper=v2_variance_taper()))
        variance = float(comps({k: jax.numpy.asarray(v) for k, v in hp_t.items()})[2])
        assert tv["mock"]["total"] == pytest.approx(variance, rel=1e-8)
        assert tv["pass"] == (tv["pass_total"] and tv["pass_events"])
        assert gates["b0_pass"] is (gi["pass"] and not gi["tail_catalog"]
                                    and tv["pass"] and gates["ii_coverage"]["pass"] and gates["iii_g12"]["pass"])
    else:
        assert "iv_taper_at_truth" not in gates
    # the selection estimator in the gate equals the pipeline's (numpy HBI) ln xi
    from gwpop_search.hbi import evaluate_selection

    spec, hp = M.truth_model("closure_widthq")
    logp = M.LogDensity(spec)
    ref = evaluate_selection(sel, logp.model, hp).log_exposure
    assert M.selection_log_xi(logp, sel, hp) == pytest.approx(ref, rel=1e-10, abs=1e-10)


def test_calibration_reproduces_targets(draw):
    cal = M.calibrate(draw, n_pool=300_000, n_width_events=12, n_width_samples=256, width_iterations=1,
                      power_steps=5, rolloff_steps=12, log=lambda *_: None)
    assert sum(cal.run_time_yr) == pytest.approx(M.T_TOTAL_YR)
    for label in M.RUN_LABELS:
        rep = cal.report[label]
        # small pool: the faint runs (O1/O2) have few detected rows
        assert rep["achieved"]["z"][1] == pytest.approx(M.CANONICAL_RUN_TARGETS[label]["z"][1], rel=0.08)
    target = M.CANONICAL_POOLED_TARGETS
    assert cal.report["pooled_m1_source_q90"]["achieved"] == pytest.approx(target["m1_source_q90"], rel=0.05)
    # the roll-off power is fitted to the heavy tail (or sits at a bound, reported)
    assert 5.0 / 6.0 <= cal.mass_rolloff_power <= 8.0
    tail = cal.report["pooled_detected_mass"]["achieved"]["p_m1_source_gt_100"]
    if 5.0 / 6.0 < cal.mass_rolloff_power < 8.0:
        assert tail == pytest.approx(target["p_m1_source_gt_100"], rel=0.3)
    assert list(cal.amplitude) == sorted(cal.amplitude) or cal.amplitude[-1] > cal.amplitude[0]
    rt = M.Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert rt.amplitude == cal.amplitude and rt.mass_rolloff == cal.mass_rolloff
    assert rt.mass_rolloff_power == cal.mass_rolloff_power
    assert rt.mass_rolloff_sharpness == cal.mass_rolloff_sharpness == M.MASS_ROLLOFF_SHARPNESS


def test_coverage_gate_refuses_a_pe_box_that_truncates_the_population(draw):
    sel_like = type("S", (), {"log_draw_density": np.zeros(3)})()
    truths = {"m1_source": np.array([30.0]), "q": np.array([0.8]), "z": np.array([0.3]),
              "chi_eff": np.array([0.0])}
    ok = M.coverage_gate(draw, sel_like, truths)
    assert ok["pass"]
    bad_box = dict(M.pe_box(draw.cosmo), m1_detector=(2.0, 500.0))
    bad = M.coverage_gate(draw, sel_like, truths, box=bad_box)
    assert not bad["pass"] and not bad["pe_box_contains_population_support"]["m1_detector"]
    assert M.pe_box(draw.cosmo)["luminosity_distance"][1] == pytest.approx(draw.cosmo.dL_of_z(M.DRAW_ZMAX))


# ---------------------------------------------------------------------------
# generator 2.0: real-like PE structure (operator decision 2026-10-02)
# ---------------------------------------------------------------------------


def test_pe_structure_is_validated_and_interpolates_the_real_correlations():
    st = M.default_pe_structure()
    assert st.q_nodes == M.REAL_PE_REF_Q_BIN_MEDIANS and st.corr_nodes == M.REAL_PE_REF_CORR_BY_Q
    assert all(f[1] == 1.0 for f in st.width_factor_nodes)
    for k, q in enumerate(st.q_nodes):
        r = st.correlation(q)
        assert (r[0, 1], r[0, 2], r[1, 2]) == pytest.approx(M.REAL_PE_REF_CORR_BY_Q[k])
    # constant outside the nodes; positive definite everywhere (convex combinations)
    np.testing.assert_array_equal(st.correlation(-0.4), st.correlation(st.q_nodes[0]))
    np.testing.assert_array_equal(st.correlation(1.7), st.correlation(st.q_nodes[-1]))
    for q in np.linspace(-0.5, 1.5, 81):
        assert np.min(np.linalg.eigvalsh(st.correlation(q))) > 0
    # widths: (rho_thr / rho)^a with the real SNR powers; the q width does not depend on x_obs_q
    s10 = st.sigmas(10.0, 0.3, (1.0, 1.0, 1.0))
    s20 = st.sigmas(20.0, 0.3, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(np.log(s20 / s10) / math.log(2.0), -np.asarray(st.snr_power))
    assert st.sigmas(15.0, 0.2, CAL_2.width_scale)[1] == st.sigmas(15.0, 0.95, CAL_2.width_scale)[1]
    with pytest.raises(ValueError, match="q width"):
        M.PEStructure(width_factor_nodes=((1.0, 1.1, 1.0),) * 4)
    with pytest.raises(ValueError, match="positive definite"):
        M.PEStructure(corr_nodes=((-0.99, 0.99, 0.99),) * 4)
    assert M.PEStructure.from_dict(st.to_dict()) == st
    rt = M.Calibration.from_dict(json.loads(json.dumps(CAL_2.to_dict())))
    assert rt.pe_structure == st and rt.generator_version == "2.0"
    # a 1.0 calibration.json (no pe_structure) keeps the 1.0 PE
    legacy = {k: v for k, v in CAL.to_dict().items() if k not in ("pe_structure", "generator_version")}
    assert M.Calibration.from_dict(legacy).pe_structure is None
    assert M.Calibration.from_dict(legacy).generator_version == "1.0"


def test_sequential_data_model_has_the_stated_likelihood():
    """p(x_q | theta) p(x_m, x_chi | x_q, theta) = N(x_obs; x(theta), C(rho, x_obs_q)) for every theta:
    the covariance is a function of the data (rho_obs, x_obs_q) only."""
    from scipy.stats import multivariate_normal, norm

    st = M.default_pe_structure()
    ws = CAL_2.width_scale
    rng = np.random.default_rng(8)
    for _ in range(25):
        rho = rng.uniform(10.0, 40.0)
        x_obs = np.array([rng.uniform(2.5, 4.5), rng.uniform(0.1, 1.05), rng.uniform(-0.4, 0.6)])
        theta = x_obs + rng.normal(0.0, 0.2, 3)
        cov = st.covariance(rho, x_obs[1], ws)
        s_q = st.sigmas(rho, 0.0, ws)[1]
        assert cov[1, 1] == pytest.approx(s_q ** 2)
        r = [0, 2]
        b = cov[r, 1] / cov[1, 1]
        cond = cov[np.ix_(r, r)] - np.outer(cov[r, 1], cov[1, r]) / cov[1, 1]
        seq = (norm.logpdf(x_obs[1], theta[1], s_q)
               + multivariate_normal.logpdf(x_obs[r], theta[r] + b * (x_obs[1] - theta[1]), cond))
        assert seq == pytest.approx(multivariate_normal.logpdf(x_obs, theta, cov), abs=1e-9)
    # observe() draws from that model: residuals standardised by the conditional structure are N(0, 1)
    theta = np.array([3.4, 0.62, 0.05])
    rho = 14.0
    xs = np.array([M.observe(rng, theta, rho, ws, st) for _ in range(20000)])
    s_q = st.sigmas(rho, 0.0, ws)[1]
    e1 = (xs[:, 1] - theta[1]) / s_q
    assert abs(e1.mean()) < 0.03 and abs(e1.std() - 1.0) < 0.03
    z = []
    for x in xs[:4000]:
        cov = st.covariance(rho, x[1], ws)
        r = [0, 2]
        b = cov[r, 1] / cov[1, 1]
        cond = cov[np.ix_(r, r)] - np.outer(cov[r, 1], cov[1, r]) / cov[1, 1]
        z.append(np.linalg.solve(np.linalg.cholesky(cond), x[r] - theta[r] - b * (x[1] - theta[1])))
    z = np.asarray(z)
    assert np.all(np.abs(z.mean(0)) < 0.06) and np.all(np.abs(z.std(0) - 1.0) < 0.05)


def test_structured_pe_sampler_matches_unfactorised_posterior():
    """With the 2.0 covariance C(rho_obs, x_obs_q) the factorised sampler still draws the exact posterior."""
    st = M.default_pe_structure()
    amp, m_ro = 20.0, CAL.mass_rolloff
    m1, q, chi, d_l, theta = 40.0, 0.45, 0.2, 800.0, 0.6
    rho_opt = theta * amp * M.snr_unit(m1, q, d_l, chi, m_ro)
    rho_obs = float(rho_opt + 0.4)
    x_obs = M.x_of(m1, q, chi) + np.array([0.02, -0.03, 0.01])
    box = {"m1_detector": (15.0, 100.0), "q": (0.05, 1.0), "luminosity_distance": (60.0, 6000.0),
           "chi_eff": (-0.6, 0.9)}
    ws = (0.15, 0.12, 0.1)
    s, acc = M.sample_pe(np.random.default_rng(12), x_obs, rho_obs, amp, 40_000, ws, m_ro, box=box, structure=st)
    y = np.stack([np.log(s["m1_detector"]), s["q"], np.log(s["luminosity_distance"]), s["chi_eff"]], 1)
    cov = st.covariance(rho_obs, x_obs[1], ws)
    corr = np.corrcoef(y[:, [0, 1, 3]].T)
    assert corr[1, 2] < -0.5  # the q < 0.5 node: strong q - chi_eff anticorrelation survives into the PE
    ref_mean, ref_sd, ess = _reference_moments_cov(x_obs, rho_obs, amp, box, cov, ws, st)
    assert ess > 2000
    assert np.all(np.abs(y.mean(0) - ref_mean) < 0.08 * ref_sd)
    np.testing.assert_allclose(y.std(0), ref_sd, rtol=0.08)


@np.errstate(invalid="ignore", divide="ignore")
def _reference_moments_cov(x_obs, rho_obs, amp, box, cov, ws, st, n=1_000_000, seed=7):
    rng = np.random.default_rng(seed)
    prop = 4.0 * cov
    yr = rng.multivariate_normal(x_obs, prop, size=n)
    (d_lo, d_hi) = box["luminosity_distance"]
    lnd = rng.uniform(math.log(d_lo), math.log(d_hi), n)
    m1, q, c, d = np.exp(yr[:, 0]), yr[:, 1], yr[:, 2], np.exp(lnd)
    inside = ((m1 >= box["m1_detector"][0]) & (m1 <= box["m1_detector"][1]) & (q >= box["q"][0])
              & (q <= box["q"][1]) & (c >= box["chi_eff"][0]) & (c <= box["chi_eff"][1]))
    diff = yr - x_obs
    log_prop = -0.5 * np.einsum("ni,ij,nj->n", diff, np.linalg.inv(prop), diff)
    logw = 2 * np.log(m1) + 3 * lnd - log_prop
    base = M.pe_log_likelihood_unmarginalised(x_obs, rho_obs, amp, ws, CAL.mass_rolloff, m1_det=m1, q=q, d_l=d,
                                              chi=c, theta=0.0, structure=st) + 0.5 * rho_obs ** 2
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


def test_structured_pe_sbc_conditional_on_detection():
    """SBC with the 2.0 data model: truths from the PE prior, detection on the data, ranks uniform."""
    st = M.default_pe_structure()
    ws = (0.2, 0.25, 0.18)
    rng = np.random.default_rng(31)
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
        x_obs = M.observe(rng, M.x_of(m1, q, c), rho_obs, ws, st)
        s, _ = M.sample_pe(rng, x_obs, rho_obs, amp, n_post, ws, CAL.mass_rolloff, box=box, batch=4096,
                           structure=st)
        truth = (m1, q, d, c)
        ranks.append([int(np.sum(s[k] < t)) for k, t in zip(("m1_detector", "q", "luminosity_distance",
                                                                "chi_eff"), truth)])
    ranks = np.asarray(ranks)
    for j in range(4):
        hist = np.bincount(ranks[:, j] // 10, minlength=10)
        assert chisquare(hist).pvalue > 1e-3, (j, hist)


# ---------------------------------------------------------------------------
# gate b-0(i): noisy-PE ensemble (operator decision 2026-10-02)
# ---------------------------------------------------------------------------


def test_ensemble_decision_flags_a_tail_catalog_and_checks_unbiasedness():
    rng = np.random.default_rng(2)
    names = ("chi_mu", "chi_log_sigma", "chi_log_sigma_q_slope")
    truth = np.array([0.03, -2.34, -2.2])
    est = truth + rng.normal(0.0, [0.01, 0.1, 0.7], (30, 3))
    med, sd = np.median(est, 0), est.std(0, ddof=1)
    ok = M.ensemble_decision(med + np.array([0.0, 2.4, 0.0]) * sd, est, truth, names)
    assert not ok["tail_catalog"] and ok["unbiased_pass"]
    assert ok["position"]["chi_log_sigma"]["n_sd_from_median"] == pytest.approx(2.4)
    tail = M.ensemble_decision(med + np.array([0.0, 0.0, -2.6]) * sd, est, truth, names)
    assert tail["tail_catalog"] and not tail["position"]["chi_log_sigma_q_slope"]["pass"]
    biased = M.ensemble_decision(med, est + np.array([0.0, 0.0, 1.0]), truth, names)
    assert not biased["unbiased_pass"] and biased["unbiased"]["chi_log_sigma_q_slope"]["pass"] is False
    assert M.TAIL_CATALOG_N_SD == 2.5 and M.ENSEMBLE_UNBIASED_N_SE == 3.0
    assert M.ENSEMBLE_N_CATALOGS == 30 and M.ENSEMBLE_N_PE == 2048


def test_chi_conditional_likelihood_is_the_pipeline_likelihood(draw, tmp_path):
    """The separable chi_eff likelihood equals the pipeline's untapered shape ln L (other hp at truth)."""
    from gwpop_search.data import PosteriorCatalog, SelectionCatalog
    from gwpop_search.hbi import HBIConfig
    from gwpop_search.hbi.numpy_backend import shape_log_likelihood
    from gwpop_search.models import compile_model_spec

    out = tmp_path / "m"
    M.build_mock(out, kind="closure_widthq", n_events=20, n_pe=128, target_found=20_000, seed=9,
                 calibration=CAL_2, run_gates=False, log=lambda *_: None)
    pe = PosteriorCatalog.from_hdf5(out / "pe.h5")
    sel = SelectionCatalog.from_hdf5(out / "selection.h5")
    spec, hp = M.truth_model("closure_widthq")
    like = M.ChiConditionalLikelihood(spec, hp, sel, draw.cosmo)
    assert like.names == ("chi_mu", "chi_log_sigma", "chi_log_sigma_q_slope")
    lnl = like.events({k: pe.samples[k] for k in M.LogDensity.FIELDS}, pe.log_ref_density, pe.offsets)
    model = compile_model_spec(spec)

    def pipeline(values):
        h = {**hp, **dict(zip(like.names, values))}
        return float(shape_log_likelihood(pe, sel, model, h, config=HBIConfig(selection_chunk_size=None)).log_likelihood)

    for values in (like.truth, like.truth + np.array([0.05, 0.3, -1.0]), np.array([-0.1, -1.5, 1.0])):
        assert lnl(values) - lnl(like.truth) == pytest.approx(pipeline(values) - pipeline(like.truth), abs=1e-7)
    fit = like.mle(lnl)
    assert len(fit["values"]) == 3 and np.all(like.bounds[:, 0] <= fit["values"])
    assert lnl(fit["values"]) >= lnl(like.truth) - 1e-6


def test_ensemble_gate_script_never_writes_inside_the_mock(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "v2_mock_ensemble_gate.py"
    spec = importlib.util.spec_from_file_location("v2_mock_ensemble_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mock = tmp_path / "mock"
    mock.mkdir()
    with pytest.raises(SystemExit, match="inside the mock directory"):
        mod.main(["--mock-dir", str(mock), "--output", str(mock / "gate.json")])
    with pytest.raises(SystemExit, match="frozen"):
        mod.main(["--mock-dir", str(mock), "--output", str(tmp_path / "frozen" / "gate.json")])
