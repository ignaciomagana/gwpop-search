"""Posterior predictive checks (v2 D6) on a selection-aware mock catalog.

The mock follows the mock-data-dag rules: sources from a known population,
a noise-realised detection statistic applied identically to the catalog and
to the injections, and PE samples that are exact posterior draws of a noisy
datum under a flat PE prior (``log_ref_density = 0``).
"""

import math

import numpy as np
import pytest
from scipy import stats

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gwpop_search.analysis._common import AnalysisInputError, posterior_from_equal_weight_draws  # noqa: E402
from gwpop_search.analysis.ppc import (  # noqa: E402
    PREDECLARED_STATISTICS,
    PPCConfig,
    ks_uniform_distance,
    posterior_predictive_check,
    ppc_criterion,
    ppc_report,
    spearman_rho,
    two_sided_ppp,
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
        config=PPCConfig(n_draws=500, seed=4, batch_size=50), verify_identity=False,
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
    # 100 draws cannot resolve alpha = 0.01 (needs n_draws * alpha >= 5)
    if not payload["failed_statistics"]:
        assert payload["status"] == "insufficient_draws"
    forged = dict(payload, identity_verified=True, status="pass")
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
