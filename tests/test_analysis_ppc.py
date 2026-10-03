"""Posterior predictive checks (v2 D6) on a selection-aware mock catalog.

The mock follows the mock-data-dag rules: sources from a known population,
a noise-realised detection statistic applied identically to the catalog and
to the injections, and PE samples that are exact posterior draws of a noisy
datum under a flat PE prior (``log_ref_density = 0``).
"""

import json
import math

import numpy as np
import pytest
from scipy import stats

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gwpop_search.analysis._common import AnalysisInputError, posterior_from_equal_weight_draws  # noqa: E402
from gwpop_search.analysis.ppc import (  # noqa: E402
    BINDING_STATISTICS,
    PPC_FORMAT,
    PREDECLARED_STATISTICS,
    PPCConfig,
    ks_uniform_distance,
    posterior_predictive_check,
    ppc_criterion,
    ppc_report,
    spearman_rho,
    two_sided_ppp,
    family_wise_false_fail_bound,
    observable_samples,
    ppc_draws_resolve_alpha,
    statistic_ppp,
    upper_tail_ppp,
)
from gwpop_search.analysis.terms import CatalogWeightEvaluator, pad_catalog  # noqa: E402
from gwpop_search.data import Campaign, PosteriorCatalog, SelectionCatalog, SelectionMode  # noqa: E402
from gwpop_search.data.fixtures import TOY_BASIS  # noqa: E402

M_LO, M_HI = 5.0, 80.0
Q_LO = 0.1
Z_HI = 1.5
SIG = {"m1_source": 1.0, "q": 0.02, "z": 0.02, "chi_eff": 0.02}


class MockPopulation:
    """m1 power law on [5, 80], q and z uniform, chi_eff ~ N(mu + dmu (q - 1), sigma)."""

    required_fields = ("m1_source", "q", "z", "chi_eff")

    def __init__(self, correlated: bool):
        self.correlated = correlated

    def __call__(self, samples, hp):
        xp = jnp if isinstance(hp["alpha"], jax.Array) or isinstance(samples["q"], jax.Array) else np
        m1, q, z, chi = samples["m1_source"], samples["q"], samples["z"], samples["chi_eff"]
        a = hp["alpha"]
        norm = xp.log((a - 1.0) / (M_LO ** (1.0 - a) - M_HI ** (1.0 - a)))
        lp_m = norm - a * xp.log(m1)
        mu = hp["mu_chi"] + (hp["dmu_q"] * (q - 1.0) if self.correlated else 0.0)
        sigma = xp.exp(hp["log_sigma_chi"])
        lp_c = -0.5 * ((chi - mu) / sigma) ** 2 - xp.log(sigma) - 0.5 * math.log(2 * math.pi)
        value = lp_m - math.log(1.0 - Q_LO) - math.log(Z_HI) + lp_c
        inside = (m1 >= M_LO) & (m1 <= M_HI) & (q >= Q_LO) & (q <= 1.0) & (z >= 0.0) & (z <= Z_HI)
        return xp.where(inside, value, -xp.inf)

    def to_config(self):
        return {"class": "MockPopulation", "correlated": self.correlated}


TRUTH = {"alpha": 2.3, "mu_chi": 0.06, "log_sigma_chi": math.log(0.08), "dmu_q": -0.6}


def _draw_sources(rng, n, truth):
    u = rng.random(n)
    a = truth["alpha"]
    m1 = (M_LO ** (1 - a) + u * (M_HI ** (1 - a) - M_LO ** (1 - a))) ** (1.0 / (1 - a))
    q = rng.uniform(Q_LO, 1.0, n)
    z = rng.uniform(0.0, Z_HI, n)
    chi = truth["mu_chi"] + truth["dmu_q"] * (q - 1.0) + math.exp(truth["log_sigma_chi"]) * rng.normal(size=n)
    return {"m1_source": m1, "q": q, "z": z, "chi_eff": chi}


def _detected(rng, src):
    snr = 30.0 * (src["m1_source"] / 20.0) ** 0.9 * (0.4 + 0.6 * src["q"]) / (1.0 + src["z"] / 0.15)
    return snr + rng.normal(size=snr.shape) > 12.0


def make_mock(seed=3, n_events=150, n_pe=192, n_inj=150_000):
    rng = np.random.default_rng(seed)
    events = {k: [] for k in SIG}
    while len(events["q"]) < n_events:
        src = _draw_sources(rng, 4000, TRUTH)
        keep = _detected(rng, src)
        for k in SIG:
            events[k].extend(src[k][keep].tolist())
    events = {k: np.asarray(v[:n_events]) for k, v in events.items()}
    samples = {}
    for k, s in SIG.items():
        datum = events[k] + s * rng.normal(size=n_events)
        samples[k] = (datum[:, None] + s * rng.normal(size=(n_events, n_pe))).ravel()
    pe = PosteriorCatalog(
        event_names=tuple(f"GW{190500 + i:06d}_000000" for i in range(n_events)),
        offsets=np.arange(n_events + 1) * n_pe,
        samples=samples,
        log_ref_density=np.zeros(n_events * n_pe),
        basis=TOY_BASIS,
    )
    inj = {
        "m1_source": rng.uniform(M_LO, M_HI, n_inj),
        "q": rng.uniform(Q_LO, 1.0, n_inj),
        "z": rng.uniform(0.0, Z_HI, n_inj),
        "chi_eff": rng.uniform(-1.0, 1.0, n_inj),
    }
    found = _detected(rng, inj)
    log_draw = -math.log(M_HI - M_LO) - math.log(1.0 - Q_LO) - math.log(Z_HI) - math.log(2.0)
    sel = SelectionCatalog(
        samples={k: v[found] for k, v in inj.items()},
        log_draw_density=np.full(int(found.sum()), log_draw),
        campaign_id=np.asarray(["MOCK"] * int(found.sum())),
        campaigns=(Campaign("MOCK", n_draw=n_inj, observing_time_yr=1.0),),
        basis=TOY_BASIS,
        mode=SelectionMode.RAW_DRAW,
    )
    return pe, sel, events


def _near(rng, centre, names, n=600, scale=0.02):
    draws = np.asarray([[centre[k] for k in names]] * n) + scale * rng.normal(size=(n, len(names)))
    return posterior_from_equal_weight_draws(names, draws)


@pytest.fixture(scope="module")
def mock():
    return make_mock()


def test_statistics_match_scipy():
    rng = np.random.default_rng(0)
    u = rng.random(57)
    assert ks_uniform_distance(u) == pytest.approx(stats.kstest(u, "uniform").statistic, abs=1e-14)
    x, y = rng.normal(size=40), rng.normal(size=40)
    y[:5] = y[0]  # ties
    assert spearman_rho(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-14)
    assert spearman_rho(np.ones(5), x[:5]) == 0.0
    t_obs = np.array([1.0, 1.0, 1.0, 1.0])
    p = two_sided_ppp(t_obs, np.array([0.0, 2.0, 2.0, 1.0]))
    assert p["p_upper"] == pytest.approx(0.5 + 0.125)
    assert p["p_lower"] == pytest.approx(0.25 + 0.125)
    assert p["p_value"] == pytest.approx(0.75)
    assert two_sided_ppp(t_obs, np.zeros(4))["p_value"] == 0.0


def test_log_weights_jax_matches_numpy(mock):
    pe, sel, _ = mock
    model = MockPopulation(correlated=True)
    names = ("alpha", "mu_chi", "log_sigma_chi", "dmu_q")
    cat = pad_catalog(pe, sel, model)
    X = np.array([[2.3, 0.06, math.log(0.08), -0.6], [1.9, 0.0, math.log(0.1), -0.3]])
    a = CatalogWeightEvaluator(cat, model, names, batch_size=1).log_weights(X)
    b = CatalogWeightEvaluator(cat, model, names, backend="numpy").log_weights(X)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        finite = np.isfinite(y)
        assert np.array_equal(finite, np.isfinite(x))
        np.testing.assert_allclose(x[finite], y[finite], rtol=0, atol=1e-10)


def test_true_model_passes_every_predeclared_statistic(mock):
    pe, sel, _ = mock
    names = ("alpha", "mu_chi", "log_sigma_chi", "dmu_q")
    sample = _near(np.random.default_rng(1), TRUTH, names)
    result = posterior_predictive_check(
        sample, pe, sel, MockPopulation(correlated=True),
        config=PPCConfig(n_draws=1000, seed=4, batch_size=50), verify_identity=False,
    )
    summary = result.summary()
    p = {k: v["p_value"] for k, v in summary["statistics"].items()}
    assert set(p) == set(PREDECLARED_STATISTICS)
    assert summary["failed_statistics"] == [], p
    assert summary["diagnostics"]["reliable"]
    # the selection matters: detected chi_eff-q correlation is reproduced
    corr = summary["statistics"]["spearman_chi_eff_q"]
    assert corr["observed"]["q50"] < -0.4 and corr["predicted"]["q50"] < -0.4
    # without identity verification the check cannot pass D6
    payload = result.to_dict()
    assert payload["status"] == "pass"
    assert ppc_criterion(payload)["status"] == "incomplete"


def test_missing_correlation_fails_the_rank_correlation_statistic(mock):
    pe, sel, events = mock
    names = ("alpha", "mu_chi", "log_sigma_chi")
    centre = {"alpha": 2.3, "mu_chi": float(np.mean(events["chi_eff"])),
              "log_sigma_chi": math.log(float(np.std(events["chi_eff"])))}
    sample = _near(np.random.default_rng(2), centre, names)
    result = posterior_predictive_check(
        sample, pe, sel, MockPopulation(correlated=False),
        config=PPCConfig(n_draws=500, seed=5, batch_size=50), verify_identity=False,
    )
    payload = result.to_dict()
    assert payload["status"] == "fail"
    assert "spearman_chi_eff_q" in payload["failed_statistics"]
    assert payload["statistics"]["spearman_chi_eff_q"]["p_value"] < 0.01
    crit = ppc_criterion(payload)
    assert crit["status"] == "fail" and "spearman_chi_eff_q" in crit["failed_statistics"]


def test_wrong_mass_slope_fails_the_m1_marginal(mock):
    pe, sel, _ = mock
    names = ("alpha", "mu_chi", "log_sigma_chi", "dmu_q")
    wrong = dict(TRUTH, alpha=1.2)
    sample = _near(np.random.default_rng(3), wrong, names)
    result = posterior_predictive_check(
        sample, pe, sel, MockPopulation(correlated=True),
        config=PPCConfig(n_draws=500, seed=6, batch_size=50), verify_identity=False,
    )
    summary = result.summary()
    assert "ks_m1" in summary["failed_statistics"]
    assert "spearman_chi_eff_q" not in summary["failed_statistics"]


def test_identity_is_required_by_default_and_columns_are_checked(mock):
    pe, sel, _ = mock
    names = ("alpha", "mu_chi", "log_sigma_chi", "dmu_q")
    sample = _near(np.random.default_rng(1), TRUTH, names, n=10)
    with pytest.raises(AnalysisInputError, match="no likelihood identity"):
        posterior_predictive_check(sample, pe, sel, MockPopulation(True), config=PPCConfig(n_draws=10))
    with pytest.raises(AnalysisInputError, match="observable column"):
        posterior_predictive_check(
            sample, pe, sel, MockPopulation(True), verify_identity=False,
            config=PPCConfig(n_draws=10, columns={"m1": "mass_1_source", "q": "q", "chi_eff": "chi_eff", "z": "z"}),
        )


def test_ppc_criterion_and_report_contract(mock):
    pe, sel, _ = mock
    names = ("alpha", "mu_chi", "log_sigma_chi", "dmu_q")
    sample = _near(np.random.default_rng(1), TRUTH, names, n=50)
    result = posterior_predictive_check(
        sample, pe, sel, MockPopulation(True), verify_identity=False,
        config=PPCConfig(n_draws=100, seed=1, batch_size=50),
    )
    payload = result.to_dict(include_draws=False)
    assert "draws" not in payload
    # 100 draws cannot resolve alpha / 2 = 0.005 (needs n_draws * alpha / 2 >= 5)
    if not payload["failed_statistics"]:
        assert payload["status"] == "insufficient_draws"
    forged = dict(payload, identity_verified=True, status="pass")
    assert ppc_criterion(dict(forged, diagnostics=dict(payload["diagnostics"], alpha_resolvable=True,
                                                       reliable=True)))["status"] == "incomplete"
    forged["config"] = dict(payload["config"], n_draws=1000)
    forged["diagnostics"] = dict(payload["diagnostics"], alpha_resolvable=True, reliable=True)
    forged["statistics"] = {k: dict(v, p_value=0.5) for k, v in payload["statistics"].items()}
    assert ppc_criterion(forged)["status"] == "pass"
    forged["statistics"]["ks_z"]["p_value"] = 0.009
    assert ppc_criterion(forged)["status"] == "fail"
    forged["config"] = dict(forged["config"], alpha=0.05)
    assert ppc_criterion(forged)["status"] == "incomplete"
    assert ppc_criterion(None)["status"] == "missing"
    del forged["statistics"]["ks_q"]
    assert ppc_criterion(forged)["status"] == "incomplete"
    report = ppc_report([result])
    assert list(report["models"]) == ["model"]
    with pytest.raises(AnalysisInputError):
        ppc_report([result, result])


def test_ks_p_values_are_one_sided_and_correlations_two_sided():
    rng = np.random.default_rng(7)
    t_pred = rng.random(2000)
    # observed distances far *below* the predicted ones: fits better than predicted -> not a failure
    better = np.full(2000, -1.0)
    assert statistic_ppp("ks_m1", better, t_pred)["p_value"] == pytest.approx(1.0)
    assert statistic_ppp("ks_m1", better, t_pred)["sidedness"] == "upper"
    # ... but the same numbers as a correlation statistic are a two-sided failure
    assert statistic_ppp("spearman_chi_eff_q", better, t_pred)["p_value"] == 0.0
    # observed distances above every predicted one fail the one-sided KS test
    worse = np.full(2000, 2.0)
    assert statistic_ppp("ks_q", worse, t_pred)["p_value"] == 0.0
    mid = np.full(2000, 0.5)
    one = upper_tail_ppp(mid, t_pred)
    assert one["p_value"] == pytest.approx(np.mean(t_pred >= 0.5), abs=1e-12)
    assert one["mc_standard_error"] == pytest.approx(math.sqrt(one["p_value"] * (1 - one["p_value"]) / 2000))
    assert ppc_draws_resolve_alpha(1000) and not ppc_draws_resolve_alpha(999)
    # D6 is decided by the six binding statistics (operator decision 2026-10-02): ~6%
    assert family_wise_false_fail_bound() == pytest.approx(1 - 0.99**6)
    assert family_wise_false_fail_bound(n_statistics=18) == pytest.approx(1 - 0.99**18)


def test_source_frame_is_derived_at_the_population_cosmology_even_if_stored():
    from types import SimpleNamespace

    from gwpop_search.models.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM()
    d_l = np.array([400.0, 2000.0, 9000.0])
    z_true = np.asarray(cosmo.z_of_dL(d_l))
    m1_det = np.array([30.0, 45.0, 80.0])
    stored = {"z": z_true * (1 + 2e-3), "m1_source": m1_det / (1 + z_true) * (1 - 1e-3)}
    data = SimpleNamespace(samples={"m1_detector": m1_det, "luminosity_distance": d_l, "q": np.ones(3),
                                    "chi_eff": np.zeros(3), **stored})
    model = SimpleNamespace(cosmology=cosmo)
    columns = {"m1": "m1_source", "q": "q", "chi_eff": "chi_eff", "z": "z"}
    values, derived, diff = observable_samples(data, columns, model, what="PE")
    assert derived == ["m1_source", "z"]
    np.testing.assert_allclose(values["z"], z_true, rtol=1e-14)
    np.testing.assert_allclose(values["m1_source"], m1_det / (1 + z_true), rtol=1e-14)
    assert diff["z"] == pytest.approx(2e-3, rel=1e-6)
    assert diff["m1_source"] == pytest.approx(1e-3, rel=1e-6)
    # without the detector-frame columns the stored ones are used (nothing derived)
    bare = SimpleNamespace(samples={k: data.samples[k] for k in ("q", "chi_eff", "z", "m1_source")})
    values, derived, diff = observable_samples(bare, columns, model, what="PE")
    assert derived == [] and diff == {}
    np.testing.assert_array_equal(values["z"], stored["z"])


# ---------------------------------------------------------------------------
# width-sensitive statistics (operator decision 2026-10-02)
# ---------------------------------------------------------------------------

from gwpop_search.analysis.ppc import (  # noqa: E402
    STATISTIC_FAMILIES,
    WIDTH_STATISTICS,
    absdev_trend,
    replicate_family_wise_rate,
    replicate_p_values,
    statistic_kind,
    tercile_spreads,
    width_statistics,
)


class MockWidthPopulation(MockPopulation):
    """As :class:`MockPopulation` with chi_eff ~ N(mu, sigma(q)), ln sigma(q) = ls + slope (q - 1)."""

    def __init__(self, q_dependent_width: bool):
        super().__init__(correlated=False)
        self.q_dependent_width = q_dependent_width

    def __call__(self, samples, hp):
        xp = jnp if isinstance(hp["alpha"], jax.Array) or isinstance(samples["q"], jax.Array) else np
        hp = dict(hp)
        slope = hp.pop("log_sigma_q_slope", 0.0) if self.q_dependent_width else 0.0
        base = super().__call__(samples, dict(hp, log_sigma_chi=hp["log_sigma_chi"] + 0.0 * samples["q"]))
        if not self.q_dependent_width:
            return base
        # replace the constant-width Gaussian by the q-dependent one
        q, chi = samples["q"], samples["chi_eff"]
        s0, s1 = xp.exp(hp["log_sigma_chi"]), xp.exp(hp["log_sigma_chi"] + slope * (q - 1.0))
        lp0 = -0.5 * ((chi - hp["mu_chi"]) / s0) ** 2 - xp.log(s0)
        lp1 = -0.5 * ((chi - hp["mu_chi"]) / s1) ** 2 - xp.log(s1)
        return base - lp0 + lp1

    def to_config(self):
        return {"class": "MockWidthPopulation", "q_dependent_width": self.q_dependent_width}


WIDTH_TRUTH = {"alpha": 2.3, "mu_chi": 0.03, "log_sigma_chi": math.log(0.05), "log_sigma_q_slope": -2.2}


def make_width_mock(seed=11, n_events=200, n_pe=192, n_inj=150_000):
    """Closure-like mock: chi_eff width grows towards low q (sigma(0.5) = 0.15, sigma(1) = 0.05)."""
    rng = np.random.default_rng(seed)
    events = {k: [] for k in SIG}
    while len(events["q"]) < n_events:
        src = _draw_sources(rng, 4000, dict(TRUTH, dmu_q=0.0, mu_chi=WIDTH_TRUTH["mu_chi"]))
        sigma = np.exp(WIDTH_TRUTH["log_sigma_chi"] + WIDTH_TRUTH["log_sigma_q_slope"] * (src["q"] - 1.0))
        src["chi_eff"] = WIDTH_TRUTH["mu_chi"] + sigma * rng.normal(size=src["q"].size)
        keep = _detected(rng, src)
        for k in SIG:
            events[k].extend(src[k][keep].tolist())
    events = {k: np.asarray(v[:n_events]) for k, v in events.items()}
    pe, sel, _ = make_mock(seed=seed + 1, n_events=n_events, n_pe=n_pe, n_inj=n_inj)
    samples = {}
    for k, s in SIG.items():
        datum = events[k] + s * rng.normal(size=n_events)
        samples[k] = (datum[:, None] + s * rng.normal(size=(n_events, n_pe))).ravel()
    pe = PosteriorCatalog(event_names=pe.event_names, offsets=pe.offsets, samples=samples,
                          log_ref_density=np.zeros(n_events * n_pe), basis=TOY_BASIS)
    return pe, sel, events


@pytest.fixture(scope="module")
def width_mock():
    return make_width_mock()


def test_width_statistics_definitions():
    rng = np.random.default_rng(0)
    x = rng.permutation((np.arange(300) + 0.5) / 300)  # exactly 100 values per tercile
    y = rng.normal(size=300) * np.where(x < 1 / 3, 3.0, 1.0)
    spreads = tercile_spreads(x, y)
    order = np.argsort(x, kind="stable")
    low = y[order[:100]]
    assert spreads[0] == pytest.approx(np.quantile(low, 0.75) - np.quantile(low, 0.25))
    assert spreads[0] > 2 * spreads[1] and spreads[0] > 2 * spreads[2]
    # wider at low x: |y - median| anti-correlated with x
    assert absdev_trend(x, y) < -0.2
    assert absdev_trend(x, y) == pytest.approx(stats.spearmanr(x, np.abs(y - np.median(y))).statistic, abs=1e-12)
    stats_ = width_statistics({"q": x, "z": 1.0 - x, "m1": x, "chi_eff": y})
    assert set(stats_) == set(WIDTH_STATISTICS) and len(WIDTH_STATISTICS) == 12
    assert stats_["iqr_chi_eff_q_t1"] == pytest.approx(spreads[0])
    assert stats_["iqr_chi_eff_z_t3"] == pytest.approx(spreads[0])  # z decreases with x
    assert stats_["spearman_absdev_chi_eff_z"] == pytest.approx(-stats_["spearman_absdev_chi_eff_q"])
    assert len(PREDECLARED_STATISTICS) == 18
    assert set(STATISTIC_FAMILIES["width"]) == set(WIDTH_STATISTICS)
    assert statistic_kind("iqr_chi_eff_m1_t2") == "width_iqr"
    assert statistic_kind("spearman_absdev_chi_eff_z") == "width_spearman_absdev"
    assert statistic_kind("ks_q") == "ks_marginal" and statistic_kind("spearman_chi_eff_q") == "spearman"
    # width statistics are two-sided
    t_pred = rng.random(2000)
    assert statistic_ppp("iqr_chi_eff_q_t1", np.full(2000, -1.0), t_pred)["p_value"] == 0.0
    with pytest.raises(ValueError):
        tercile_spreads(x[:5], y[:5])


def test_replicate_family_wise_rate_keeps_the_correlations():
    rng = np.random.default_rng(1)
    s, k = 4000, 12
    # independent statistics: the rate approaches the independence bound
    independent = {f"spearman_x{i}": rng.normal(size=s) for i in range(k)}
    rate = replicate_family_wise_rate(independent, 0.01, names=list(independent))
    assert rate["rate"] == pytest.approx(1 - 0.99**k, abs=0.025)
    # perfectly correlated copies: the rate is that of one statistic
    one = rng.normal(size=s)
    copies = {f"spearman_x{i}": one for i in range(k)}
    rate = replicate_family_wise_rate(copies, 0.01, names=list(copies))
    assert rate["rate"] == pytest.approx(0.01, abs=0.004)
    # one-sided KS: only the upper tail counts
    p = replicate_p_values("ks_m1", np.arange(100.0))
    assert p[-1] == 0.0 and p[0] == 1.0
    p = replicate_p_values("spearman_chi_eff_q", np.arange(100.0))
    assert p[0] == 0.0 and p[-1] == 0.0 and p[50] == pytest.approx(1.0, abs=0.03)
    # ties count 1/2
    p = replicate_p_values("spearman_chi_eff_q", np.ones(10))
    np.testing.assert_allclose(p, 1.0)


def test_constant_width_model_is_flagged_by_a_width_statistic_which_is_not_binding(width_mock):
    pe, sel, events = width_mock
    names = ("alpha", "mu_chi", "log_sigma_chi", "log_sigma_q_slope")
    sample = _near(np.random.default_rng(5), WIDTH_TRUTH, names)
    good = posterior_predictive_check(
        sample, pe, sel, MockWidthPopulation(q_dependent_width=True),
        config=PPCConfig(n_draws=1000, seed=7, batch_size=50), verify_identity=False,
    ).summary()
    p_good = {k: v["p_value"] for k, v in good["statistics"].items()}
    assert good["failed_statistics"] == [], p_good
    # the constant-width fit (R0 analogue): mean and width at the detected catalog's values
    names0 = ("alpha", "mu_chi", "log_sigma_chi")
    centre = {"alpha": 2.3, "mu_chi": float(np.median(events["chi_eff"])),
              "log_sigma_chi": math.log(float(np.std(events["chi_eff"])))}
    bad = posterior_predictive_check(
        _near(np.random.default_rng(6), centre, names0), pe, sel, MockWidthPopulation(q_dependent_width=False),
        config=PPCConfig(n_draws=1000, seed=8, batch_size=50), verify_identity=False,
    ).summary()
    # the width statistics see the misfit and are reported below alpha ...
    flagged = set(bad["reported_statistics_below_alpha"])
    assert flagged & set(WIDTH_STATISTICS), {k: v["p_value"] for k, v in bad["statistics"].items()}
    assert "spearman_absdev_chi_eff_q" in flagged
    assert bad["statistics"]["spearman_absdev_chi_eff_q"]["below_alpha"]
    assert not bad["statistics"]["spearman_absdev_chi_eff_q"]["binding"]
    assert not bad["statistics"]["spearman_absdev_chi_eff_q"]["failed"]
    # ... but they are reported only (operator decision 2026-10-02): D6 is decided by the original
    # six, which do not see the width trend (the pilot (b) finding)
    assert not set(bad["failed_statistics"]) & set(WIDTH_STATISTICS)
    assert bad["failed_statistics"] == [] and bad["status"] != "fail"
    assert bad["binding_statistics"] == list(BINDING_STATISTICS) and len(BINDING_STATISTICS) == 6
    assert bad["reported_statistics"] == list(WIDTH_STATISTICS)
    mult = bad["multiplicity"]
    assert mult["n_statistics"] == 6 and mult["n_statistics_reported_only"] == 12
    assert mult["family_wise_false_fail_upper_bound"] == pytest.approx(1 - 0.99**6)
    fw = mult["family_wise_false_fail_empirical"]
    assert 0.0 <= fw["rate"] <= 0.2 and set(fw["per_family"]) == {"marginal", "correlation"}
    assert set(mult["family_wise_false_fail_empirical_all_statistics"]["per_family"]) == {
        "marginal", "correlation", "width"}
    # the D6 criterion on the payload: reported width p-values, binding status from the six
    payload = {"format_version": PPC_FORMAT, "config": {"alpha": 0.01, "n_draws": 1000},
               "identity_verified": True, "diagnostics": {"reliable": True, "alpha_resolvable": True},
               **bad}
    crit = ppc_criterion(payload)
    assert crit["status"] == "pass", crit
    assert set(crit["p_values"]) == set(BINDING_STATISTICS)
    assert "spearman_absdev_chi_eff_q" in crit["reported_statistics_below_alpha"]
    # a format-1.2 payload (width statistics then binding) is re-decided on the six
    old = {**payload, "format_version": "gwpop-search-ppc-1.2"}
    assert ppc_criterion(old)["status"] == "pass"
    assert ppc_criterion(old)["family_wise_false_fail_empirical"] is None
    # a binding statistic below alpha fails D6
    worse = json.loads(json.dumps(payload))
    worse["statistics"]["ks_q"]["p_value"] = 0.001
    assert ppc_criterion(worse)["status"] == "fail" and ppc_criterion(worse)["failed_statistics"] == ["ks_q"]
