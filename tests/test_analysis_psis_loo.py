import math
import warnings

import numpy as np
import pytest
from scipy.stats import norm

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.analysis import toys  # noqa: E402
from gwpop_search.analysis._common import AnalysisInputError  # noqa: E402
from gwpop_search.analysis.psis_loo import (  # noqa: E402
    PSIS_LOO_FORMAT,
    PSISLOOConfig,
    evaluate_model_loo_terms,
    flagged_events,
    gpd_fit,
    loo_edge_influence,
    psis_loo_estimate,
    psis_loo_model,
    psis_loo_report,
    psis_smooth,
)
from gwpop_search.data import drop_posterior_events  # noqa: E402
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import DynestyConfig, PriorSpec, run_dynesty_population  # noqa: E402
from gwpop_search.validation import scenarios_from_psis_flags  # noqa: E402

HBI = HBIConfig(selection_chunk_size=None)


def test_psis_smoothing_matches_arviz_for_equal_weights():
    array_stats = pytest.importorskip("arviz_stats.base").array_stats
    rng = np.random.default_rng(0)
    for scale in (0.5, 1.5, 3.0):
        log_lik = rng.normal(0.0, scale, size=3000)
        ours = psis_smooth(-log_lik)
        ref_lw, ref_k = array_stats.psislw(log_lik)
        np.testing.assert_allclose(ours.khat, ref_k, rtol=1e-10)
        ours_norm = ours.log_weights - np.logaddexp.reduce(ours.log_weights)
        np.testing.assert_allclose(ours_norm, np.asarray(ref_lw), rtol=0, atol=1e-10)


def test_psis_keeps_the_scale_and_ignores_zero_weights():
    rng = np.random.default_rng(1)
    lw = rng.normal(size=500) + 7.0
    lw[:10] = -np.inf
    out = psis_smooth(lw)
    assert np.all(np.isneginf(out.log_weights[:10]))
    # smoothing only rescales the tail: the total stays close to the raw total
    assert abs(np.logaddexp.reduce(out.log_weights[10:]) - np.logaddexp.reduce(lw[10:])) < 0.1


@pytest.mark.parametrize("k_true", [0.2, 0.8])
def test_gpd_fit_recovers_the_shape(k_true):
    rng = np.random.default_rng(2)
    u = rng.uniform(size=4000)
    x = np.sort((np.power(1.0 - u, -k_true) - 1.0) / k_true)
    k, sigma = gpd_fit(x)
    assert abs(k - k_true) < 0.1 and sigma > 0


def _conjugate_setup(y, n_proposal=20000, seed=3, proposal_scale=2.0):
    """y_i ~ N(theta, 1), flat prior: posterior N(mean(y), 1/N); weighted proposal draws."""
    rng = np.random.default_rng(seed)
    n = y.size
    mean, sd = y.mean(), 1.0 / math.sqrt(n)
    theta = rng.normal(mean, proposal_scale * sd, size=n_proposal)
    log_w = norm.logpdf(theta, mean, sd) - norm.logpdf(theta, mean, proposal_scale * sd)
    return theta, log_w


def test_weighted_psis_loo_matches_the_analytic_conjugate_answer():
    y = np.random.default_rng(4).normal(0.3, 1.0, size=12)
    theta, log_w = _conjugate_setup(y)
    n = y.size
    for i in range(n):
        rest = np.delete(y, i)
        exact = norm.logpdf(y[i], rest.mean(), math.sqrt(1.0 + 1.0 / (n - 1)))
        est = psis_loo_estimate(log_w, norm.logpdf(y[i], theta, 1.0))
        assert est.khat < 0.5
        assert abs(est.elpd - exact) < 4 * est.se + 2e-3
        # unsmoothed identity: ln Z_-i - ln Z = log E_post[1/lambda_i] = -elpd_i
        assert est.raw_log_mean_inverse == pytest.approx(-est.elpd, abs=5 * est.se + 2e-3)


def test_khat_flags_an_outlier_that_moves_the_posterior():
    y = np.array([0.1, -0.3, 0.2, 0.0, 9.0])
    rng = np.random.default_rng(5)
    theta = rng.normal(y.mean(), 1.0 / math.sqrt(y.size), size=4000)  # equal-weight posterior draws
    est = psis_loo_estimate(np.zeros(theta.size), norm.logpdf(y[-1], theta, 1.0))
    assert est.khat > 0.7
    calm = psis_loo_estimate(np.zeros(theta.size), norm.logpdf(y[0], theta, 1.0))
    assert calm.khat < 0.5


# ---------------------------------------------------------------------------
# Toy HBI with dynesty: PSIS versus exact drop-one refits
# ---------------------------------------------------------------------------

PRIORS = {"mu": PriorSpec("uniform", low=-2.0, high=2.0), "sigma": PriorSpec("uniform", low=0.2, high=3.0)}
# Estimator toys: the sampler is irrelevant to what is validated here (production runs use
# rslice, decision D1); uniform sampling in ellipsoids is the cheapest correct choice in 2-D.
CONFIG = DynestyConfig(nlive=200, sample="unif", dlogz=0.1, batch_size=64, num_posterior_samples=400)


def _runs(pe, sel, model, priors, seeds, root, label, config=CONFIG):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [
            run_dynesty_population(pe, sel, model, priors, seed=s, config=config, hbi_config=HBI,
                                   run_dir=root / f"{label}_{s}")
            for s in seeds
        ]


@pytest.fixture(scope="module")
def gaussian_toy(tmp_path_factory):
    obs = toys.ToyObservation()
    rng = np.random.default_rng(3)
    d = toys.toy_observed_data(rng, lambda r, n: r.normal(0.0, 1.0, n), 8, obs)
    d[0] = 3.3
    pe = toys.toy_posterior_catalog(rng, d, 150, obs)
    sel = toys.toy_selection_catalog(rng, 15000, obs)
    model = toys.GaussianToy()
    root = tmp_path_factory.mktemp("psis-toy")
    full = _runs(pe, sel, model, PRIORS, (1, 2), root, "full")
    return pe, sel, model, full, root


def test_psis_loo_matches_exact_drop_one_refits(gaussian_toy):
    pe, sel, model, full, root = gaussian_toy
    terms = evaluate_model_loo_terms(full, pe, sel, model, label="gauss")
    result = psis_loo_model(terms)
    assert result.n_runs == 2 and not any(result.flags)
    lnz = np.mean([r.log_evidence for r in full])
    sig_full = max(np.std([r.log_evidence for r in full], ddof=1), np.mean([r.log_evidence_error for r in full]))
    for i in (0, int(np.argmin(np.abs(np.arange(pe.n_events) - 3)))):
        dropped = drop_posterior_events(pe, [pe.event_names[i]])
        (refit,) = _runs(dropped, sel, model, PRIORS, (11,), root, f"drop{i}")
        lnz_drop = refit.log_evidence
        err = math.sqrt(sig_full**2 / 2 + refit.log_evidence_error**2 + result.total_uncertainty[i] ** 2)
        assert abs((lnz - lnz_drop) - result.elpd[i]) < 3 * err


def test_loo_terms_refuse_other_data_and_report_structure(gaussian_toy):
    pe, sel, model, full, _ = gaussian_toy
    other = toys.toy_posterior_catalog(np.random.default_rng(99), np.linspace(0.6, 2.0, 8), 150,
                                       toys.ToyObservation())
    with pytest.raises(AnalysisInputError):
        evaluate_model_loo_terms(full, other, sel, model)
    terms = evaluate_model_loo_terms(full, pe, sel, model, label="gauss")
    result = psis_loo_model(terms, forced_events=[pe.event_names[2]])
    assert result.flags[2] == ("forced",)
    report = psis_loo_report(
        {"p": result, "c": result},
        edges=[{"parent_hash": "p", "child_hash": "c", "mutation_id": "m"}],
        log_model_priors={"p": math.log(0.5), "c": math.log(0.5)},
        structure_membership={"m=on": ["c"]},
    )
    assert report["format_version"] == PSIS_LOO_FORMAT
    edge = report["edges"][0]
    assert edge["max_abs_delta_log_bayes_factor"] == pytest.approx(0.0, abs=1e-12)
    assert flagged_events(report) == [pe.event_names[2]]
    probs = report["model_probabilities"]["per_event"][pe.event_names[0]]
    assert probs["p"] == pytest.approx(0.5)
    scenarios = scenarios_from_psis_flags(report)
    assert [s.drop_events for s in scenarios] == [(pe.event_names[2],)]
    assert scenarios[0].category == "leave_one_out"
    rows = loo_edge_influence(result, result, log_bayes_factor=2.0)
    assert all(not row["sign_reversal"] for row in rows)


def test_support_complement_flags_the_event_that_defines_an_edge(tmp_path):
    obs = toys.ToyObservation(sigma_obs=0.15)
    rng = np.random.default_rng(7)
    d = np.array([-0.3, 0.1, 0.4, 0.6, 0.8, 0.9, 2.8])  # the last event alone pins xmax >= ~2.4
    pe = toys.toy_posterior_catalog(rng, d, 60, obs)
    sel = toys.toy_selection_catalog(rng, 15000, obs)
    model = toys.TruncatedGaussianToy()
    priors = {"mu": PriorSpec("uniform", low=-2.0, high=2.0), "xmax": PriorSpec("uniform", low=0.5, high=4.0)}
    light = DynestyConfig(nlive=120, sample="unif", dlogz=0.5, batch_size=64, num_posterior_samples=300)
    full = _runs(pe, sel, model, priors, (1, 2), tmp_path, "edge", light)
    assert any(r.diagnostics["n_zero_likelihood_points"] > 0 for r in full)
    result = psis_loo_model(evaluate_model_loo_terms(full, pe, sel, model, label="edge"),
                            config=PSISLOOConfig())
    # ~35-40% of the leave-one-out integrand lies where the full-data likelihood is zero,
    # while k-hat alone does not flag the event
    assert result.complement_mass_max[-1] > 0.05
    assert "support_complement" in result.flags[-1]
    assert np.all(result.complement_mass_max[:3] < 1e-3)
