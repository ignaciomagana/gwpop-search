import math
import warnings

import numpy as np
import pytest
from scipy.stats import norm

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from gwpop_search.analysis import toys  # noqa: E402
from gwpop_search.analysis._common import (  # noqa: E402
    AnalysisInputError,
    pool_dynesty_results,
    posterior_from_equal_weight_draws,
)
from gwpop_search.analysis.sddr import (  # noqa: E402
    EdgeNesting,
    NullEmbedding,
    boundary_density,
    classify_edge,
    classify_graph_edges,
    density_at_null,
    edge_classification_report,
    sddr_edge_check,
    silverman_bandwidth,
    verify_nesting,
    vw_factor,
)
from gwpop_search.analysis.terms import BatchedCatalogTerms  # noqa: E402
from gwpop_search.grammar import (  # noqa: E402
    DEFAULT_MUTATIONS,
    apply_mutation,
    baseline_model_spec,
    enumerate_model_graph,
)
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import DynestyConfig, PriorSpec, run_dynesty_population  # noqa: E402

HBI = HBIConfig(selection_chunk_size=None)
MUT = {m.mutation_id: m for m in DEFAULT_MUTATIONS}


def test_depth1_edges_are_classified_from_the_grammar():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=40)
    rows = {row.mutation_id: row for row in classify_graph_edges(graph)}
    assert len(rows) == 14
    exact = sorted(k for k, v in rows.items() if v.classification == "exact")
    assert exact == sorted(
        [f"chieff.{w}.linear_{x}" for w in ("mean", "width") for x in ("m1", "q", "z")]
        + ["pairing.beta.linear_m1", "pairing.beta.logistic_m1", "mass.family.powerlaw"]
    )
    assert sorted(k for k, v in rows.items() if v.classification == "approximate") == [
        "chieff.family.gaussian_mixture",
        "mass.family.pl_two_peak",
    ]
    assert sorted(k for k, v in rows.items() if v.classification == "evidence_only") == [
        "mass.family.broken_powerlaw",
        "pairing.family.truncated_gaussian_q",
        "redshift.family.madau_dickinson",
    ]
    peak = rows["mass.family.powerlaw"]
    assert peak.larger_model == "parent" and peak.embeddings[0].location == "lower"
    assert peak.embeddings[0].tested_parameter == "peak_fraction"
    assert set(peak.embeddings[0].unidentified) == {"peak_mu", "peak_sigma"}
    logistic = rows["pairing.beta.logistic_m1"]
    assert logistic.embeddings[0].tested_parameter == "delta_beta_q"
    assert logistic.embeddings[0].location == "interior"
    mixture = rows["chieff.family.gaussian_mixture"]
    assert mixture.vw_required and mixture.vw_first_form_valid and mixture.symmetric_embeddings
    assert mixture.prior_mismatches == ("chi_mu->chi_mu_1", "chi_mu->chi_mu_2", "chi_sigma->chi_sigma_1",
                                        "chi_sigma->chi_sigma_2")
    two_peak = rows["mass.family.pl_two_peak"]
    assert not two_peak.vw_first_form_valid
    report = edge_classification_report(graph)
    assert report["counts"] == {"exact": 9, "approximate": 2, "evidence_only": 3}


def test_replacement_and_slope_parent_edges_are_not_nested():
    root = baseline_model_spec()
    m1 = apply_mutation(root, MUT["chieff.mean.linear_m1"])
    q_from_m1 = apply_mutation(m1, MUT["chieff.mean.linear_q"])
    assert classify_edge(m1, q_from_m1, "chieff.mean.linear_q").classification == "evidence_only"
    mixture = apply_mutation(m1, MUT["chieff.family.gaussian_mixture"])
    assert classify_edge(m1, mixture, "chieff.family.gaussian_mixture").classification == "evidence_only"
    additive = apply_mutation(m1, MUT["pairing.beta.linear_m1"])
    assert classify_edge(m1, additive, "pairing.beta.linear_m1").classification == "exact"


def test_exact_embeddings_are_verified_numerically():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=40)
    by_hash = graph.by_hash
    for row in classify_graph_edges(graph):
        if row.classification == "evidence_only":
            continue
        check = verify_nesting(row, by_hash[row.parent_hash], by_hash[row.child_hash], n_checks=3, n_samples=256)
        if row.classification == "exact" or row.mutation_id == "chieff.family.gaussian_mixture":
            assert check["all_nested"], (row.mutation_id, check)


def test_boundary_density_removes_the_factor_two_bias():
    rng = np.random.default_rng(0)
    b = 4.0
    x = rng.beta(1.0, b, size=20000)  # density b at the boundary 0
    w = np.full(x.size, 1.0 / x.size)
    est = density_at_null(x, w, 0.0, location="lower", support=(0.0, 1.0), n_bootstrap=50)
    assert est.estimable
    assert math.exp(est.log_density) == pytest.approx(b, rel=0.05)
    naive = est.alternatives["log_density_naive"]
    assert math.exp(naive) == pytest.approx(b / 2, rel=0.1)  # the ln 2 bias of a naive kernel
    assert abs(est.log_density - math.log(b)) < 3 * est.sigma + 0.02
    upper = boundary_density(1.0 - x, w, 1.0, 0.02, side="upper", method="linear")
    assert upper == pytest.approx(b, rel=0.06)


@pytest.mark.parametrize(
    "truth,sampler,zero_gradient",
    [
        # p'(0) = 0: reflection is exact here, so it must be unbiased too
        (2.0 / math.sqrt(2.0 * math.pi), lambda r, n: np.abs(r.normal(0.0, 1.0, n)), True),
        # p'(0) != 0: only the local-linear boundary kernel stays unbiased
        (2.0, lambda r, n: r.exponential(0.5, n), False),
        (4.0, lambda r, n: r.beta(1.0, 4.0, size=n), False),
    ],
)
def test_boundary_estimators_bias_ranking(truth, sampler, zero_gradient):
    """The default 'linear' kernel is unbiased at a bound; the alternatives are not.

    Averaged over independent samples, so this is a bias statement rather than
    one realization. The full 6-target, 200-trial study is recorded under
    gwpop-search-data/validation/analysis_estimators/boundary_density_validation.*.
    A boundary null (peak_fraction = 0) puts this bias straight into ln BF, so a
    naive symmetric kernel would shift ln BF by about ln 2.
    """
    rng = np.random.default_rng(3)
    values: dict[str, list[float]] = {m: [] for m in ("linear", "reflection", "naive")}
    for _ in range(25):
        x = sampler(rng, 4000)
        w = np.full(x.size, 1.0 / x.size)
        h = silverman_bandwidth(x, w)
        for method in values:
            values[method].append(boundary_density(x, w, 0.0, h, side="lower", method=method))
    bias = {m: float(np.mean(v)) / truth - 1.0 for m, v in values.items()}
    assert abs(bias["linear"]) < 0.03, bias
    # a symmetric kernel at a bound keeps only half its mass: ~ -50%, i.e. ~ -ln 2 in ln BF
    assert -0.60 < bias["naive"] < -0.45, bias
    if zero_gradient:
        assert abs(bias["reflection"]) < 0.03, bias
    else:
        # reflection assumes p'(0) = 0 and is visibly biased when that fails
        assert bias["reflection"] < -0.05, bias
        assert abs(bias["linear"]) < abs(bias["reflection"]), bias


def test_interior_density_and_tail_bound():
    rng = np.random.default_rng(1)
    x = rng.normal(0.4, 1.0, size=20000)
    w = np.full(x.size, 1.0 / x.size)
    est = density_at_null(x, w, 0.0, support=(-5.0, 5.0), n_bootstrap=50)
    assert abs(est.log_density - norm.logpdf(0.0, 0.4, 1.0)) < 3 * est.sigma + 0.01
    far = density_at_null(x, w, 6.5, support=(-10.0, 10.0), n_bootstrap=20)
    assert not far.estimable
    assert far.log_density_upper_limit > norm.logpdf(6.5, 0.4, 1.0)


def test_vw_factor_matches_the_prior_ratio_expectation():
    rng = np.random.default_rng(2)
    n = 40000
    omega = rng.uniform(0.0, 1.0, n)
    psi = rng.normal(0.1, 0.2, 3 * n)
    psi = psi[np.abs(psi) <= 0.5][:n]  # the larger model's posterior lives inside its prior
    sample = posterior_from_equal_weight_draws(("omega", "psi_large"), np.column_stack([omega, psi]))
    emb = NullEmbedding("omega", 0.0, "lower", {"psi": "psi_large"}, ())
    small = {"psi": PriorSpec("uniform", low=-0.3, high=0.3)}
    large = {"psi_large": PriorSpec("uniform", low=-0.5, high=0.5)}
    out = vw_factor(sample, emb, small, large, eps=0.05)
    inside = (norm.cdf(0.3, 0.1, 0.2) - norm.cdf(-0.3, 0.1, 0.2)) / (
        norm.cdf(0.5, 0.1, 0.2) - norm.cdf(-0.5, 0.1, 0.2)
    )
    assert out["log_vw"] == pytest.approx(math.log((1.0 / 0.6) * inside), abs=0.03)
    outside = posterior_from_equal_weight_draws(("omega", "psi_large"), np.array([[0.0, 0.7], [0.01, 0.1]]))
    with pytest.raises(AnalysisInputError, match="outside"):
        vw_factor(outside, emb, small, large, eps=0.05)


# ---------------------------------------------------------------------------
# SDDR vs nested sampling on toy HBIs
# ---------------------------------------------------------------------------

# Estimator toys: the sampler is irrelevant to what is validated here (production runs use
# rslice, decision D1); uniform sampling in ellipsoids is the cheapest correct choice in 2-D.
CONFIG = DynestyConfig(nlive=200, sample="unif", dlogz=0.1, batch_size=64, num_posterior_samples=500)


def _run(pe, sel, model, priors, seed, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_dynesty_population(pe, sel, model, priors, seed=seed, config=CONFIG, hbi_config=HBI,
                                      run_dir=path)


def test_interior_sddr_agrees_with_nested_sampling(tmp_path):
    obs = toys.ToyObservation()
    rng = np.random.default_rng(11)
    d = toys.toy_observed_data(rng, lambda r, n: r.normal(0.15, 1.0, n), 15, obs)
    pe = toys.toy_posterior_catalog(rng, d, 100, obs)
    sel = toys.toy_selection_catalog(rng, 10000, obs)
    larger = toys.GaussianToy()
    smaller = toys.GaussianToy(fixed={"mu": 0.0})
    large_priors = {"mu": PriorSpec("uniform", low=-2.0, high=2.0), "sigma": PriorSpec("uniform", low=0.3, high=3.0)}
    small_priors = {"sigma": PriorSpec("uniform", low=0.3, high=3.0)}
    runs_l = [_run(pe, sel, larger, large_priors, s, tmp_path / f"l{s}") for s in (1, 2)]
    runs_s = [_run(pe, sel, smaller, small_priors, 3, tmp_path / "s3")]
    lnbf = np.mean([r.log_evidence for r in runs_l]) - runs_s[0].log_evidence
    sig_ns = math.sqrt(sum(r.log_evidence_error**2 for r in runs_l) / 4 + runs_s[0].log_evidence_error ** 2)
    nesting = EdgeNesting("p", "c", "toy.mean", "exact", "child",
                          (NullEmbedding("mu", 0.0, "interior", {}, ()),))
    check = sddr_edge_check(nesting, pool_dynesty_results(runs_l), large_priors, small_priors,
                            log_bf_ns=float(lnbf), sigma_ns=sig_ns, n_bootstrap=100)
    assert check.status in {"agree", "disagree"}
    assert abs(check.log_bf_child_over_parent_sddr - lnbf) < 1.5 * check.tolerance


def test_boundary_sddr_agrees_with_nested_sampling_and_the_naive_kernel_does_not(tmp_path):
    obs = toys.ToyObservation()
    rng = np.random.default_rng(12)
    d = toys.toy_observed_data(rng, lambda r, n: r.normal(0.0, 1.0, n), 20, obs)
    pe = toys.toy_posterior_catalog(rng, d, 100, obs)
    sel = toys.toy_selection_catalog(rng, 10000, obs)
    larger = toys.PeakMixtureToy(peak_sigma=0.3)
    priors = {"f": PriorSpec("uniform", low=0.0, high=0.5), "m": PriorSpec("uniform", low=-3.0, high=3.0)}
    runs = [_run(pe, sel, larger, priors, s, tmp_path / f"b{s}") for s in (1, 2)]
    nested = toys.PeakMixtureToy(peak_sigma=0.3, include_peak=False)
    log_z_small = float(BatchedCatalogTerms(pe, sel, nested, ("dummy",), hbi_config=HBI).log_likelihood([[0.0]])[0])
    lnbf = float(np.mean([r.log_evidence for r in runs]) - log_z_small)
    sig_ns = math.sqrt(sum(r.log_evidence_error**2 for r in runs)) / 2
    nesting = EdgeNesting("p", "c", "toy.peak", "exact", "child",
                          (NullEmbedding("f", 0.0, "lower", {}, ("m",)),))
    check = sddr_edge_check(nesting, pool_dynesty_results(runs), priors, {}, log_bf_ns=lnbf,
                            sigma_ns=sig_ns, n_bootstrap=100)
    assert abs(check.log_bf_child_over_parent_sddr - lnbf) < 1.5 * check.tolerance
    density = check.details["densities"][0]
    naive_bf = check.details["log_prior_at_null"] - density["alternatives"]["log_density_naive"]
    # the naive boundary kernel overstates the evidence for the extra component by ~ln 2
    assert naive_bf - check.log_bf_child_over_parent_sddr > 0.4
