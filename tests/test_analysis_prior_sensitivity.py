import math

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from gwpop_search.analysis._common import AnalysisInputError, WeightedPosterior, posterior_from_equal_weight_draws
from gwpop_search.analysis.prior_sensitivity import (
    edge_prior_sensitivity,
    model_prior_variants,
    occam_widening,
    prior_reweighting,
    rescaled_prior,
)
from gwpop_search.grammar import DEFAULT_MUTATIONS, apply_mutation, baseline_model_spec, enumerate_model_graph
from gwpop_search.inference import PriorSpec

MUT = {m.mutation_id: m for m in DEFAULT_MUTATIONS}


def test_rescaled_priors_keep_the_centre():
    u = rescaled_prior(PriorSpec("uniform", low=-0.6, high=0.6), 0.5)
    assert (u.low, u.high) == pytest.approx((-0.3, 0.3))
    lu = rescaled_prior(PriorSpec("log_uniform", low=0.03, high=0.5), 2.0)
    assert math.log(lu.low) + math.log(lu.high) == pytest.approx(math.log(0.03) + math.log(0.5))
    assert math.log(lu.high / lu.low) == pytest.approx(2 * math.log(0.5 / 0.03))


def _truncated_normal_posterior(mu, sd, low, high, n, seed):
    a, b = (low - mu) / sd, (high - mu) / sd
    draws = truncnorm.rvs(a, b, loc=mu, scale=sd, size=n, random_state=np.random.default_rng(seed))
    return posterior_from_equal_weight_draws(("x",), draws[:, None])


def _log_z(mu, sd, low, high):
    """ln of int N(x; mu, sd) U(x; low, high) dx."""
    return math.log(norm.cdf(high, mu, sd) - norm.cdf(low, mu, sd)) - math.log(high - low)


@pytest.mark.parametrize("new_bounds", [(-1.0, 1.0), (0.2, 1.0), (0.35, 2.0)])
def test_narrowed_prior_reweighting_is_exact(new_bounds):
    mu, sd, low, high = 0.3, 0.2, -2.0, 2.0
    sample = _truncated_normal_posterior(mu, sd, low, high, 60000, seed=1)
    old = PriorSpec("uniform", low=low, high=high)
    new = PriorSpec("uniform", low=new_bounds[0], high=new_bounds[1])
    out = prior_reweighting(sample, "x", old, new)
    exact = _log_z(mu, sd, *new_bounds) - _log_z(mu, sd, low, high)
    assert out["delta_log_evidence"] == pytest.approx(exact, abs=4 * out["importance_se"] + 1e-3)
    assert out["reweighting_ess"] <= sample.kish_ess + 1e-6
    with pytest.raises(AnalysisInputError, match="support"):
        prior_reweighting(sample, "x", old, PriorSpec("uniform", low=-3.0, high=1.0))


def test_occam_widening_requires_a_contained_posterior():
    contained = _truncated_normal_posterior(0.3, 0.2, -2.0, 2.0, 20000, seed=2)
    prior = PriorSpec("uniform", low=-2.0, high=2.0)
    out = occam_widening(contained, "x", prior, factor=2.0)
    assert out["contained"] and out["delta_log_evidence"] == pytest.approx(-math.log(2.0))
    # exact check: widening a contained posterior's prior costs ln 2
    exact = _log_z(0.3, 0.2, -4.0, 4.0) - _log_z(0.3, 0.2, -2.0, 2.0)
    assert out["delta_log_evidence"] == pytest.approx(exact, abs=1e-6)
    edge = _truncated_normal_posterior(1.9, 0.2, -2.0, 2.0, 20000, seed=3)
    flagged = occam_widening(edge, "x", prior, factor=2.0)
    assert not flagged["contained"] and flagged["delta_log_evidence"] is None
    assert flagged["flag"] == "posterior_not_contained"


def _fake_posterior(spec, overrides, n=20000, seed=0):
    """Draws inside every prior; ``overrides`` gives N(mu, sd) columns (truncated to the prior)."""
    rng = np.random.default_rng(seed)
    names = tuple(sorted(spec.priors))
    columns = []
    for name in names:
        prior = spec.priors[name]
        lo, hi = prior.parameters["low"], prior.parameters["high"]
        if name in overrides:
            mu, sd = overrides[name]
            a, b = (lo - mu) / sd, (hi - mu) / sd
            columns.append(truncnorm.rvs(a, b, loc=mu, scale=sd, size=n, random_state=rng))
        else:
            columns.append(rng.uniform(lo, hi, n))
    return WeightedPosterior(names, np.column_stack(columns), np.full(n, 1.0 / n), np.zeros(n, int), 1, "fake")


def test_edge_width_sensitivity_follows_lindley_for_a_contained_slope():
    parent = baseline_model_spec()
    child = apply_mutation(parent, MUT["chieff.mean.linear_m1"])
    child_sample = _fake_posterior(child, {"chi_mu_m1_slope": (0.004, 0.002)})
    parent_sample = _fake_posterior(parent, {})
    out = edge_prior_sensitivity(parent, child, parent_sample, child_sample, log_bayes_factor=2.5)
    (row,) = out["parameters"]
    assert row["parameter"] == "chi_mu_m1_slope" and row["model"] == "child"
    # halving a contained prior gains ln 2; doubling loses ln 2: d lnBF / d lnW = -1
    assert row["delta_log_bayes_factor_narrowed"] == pytest.approx(math.log(2.0), abs=0.02)
    assert row["delta_log_bayes_factor_widened"] == pytest.approx(-math.log(2.0))
    assert row["dlogbf_dlogwidth"] == pytest.approx(-1.0, abs=0.03)
    assert {v["variant"] for v in out["variants"]} == {"narrowed", "widened"}
    # removal edge: the parent's extra parameters are varied, with the opposite sign
    removed = apply_mutation(parent, MUT["mass.family.powerlaw"])
    peak = _fake_posterior(parent, {"peak_fraction": (0.1, 0.03), "peak_mu": (34.0, 2.0)})
    out = edge_prior_sensitivity(parent, removed, peak, _fake_posterior(removed, {}), log_bayes_factor=-3.0)
    rows = {r["parameter"]: r for r in out["parameters"]}
    assert set(rows) == {"peak_fraction", "peak_mu", "peak_sigma"}
    assert all(r["model"] == "parent" for r in rows.values())
    assert rows["peak_mu"]["delta_log_bayes_factor_widened"] == pytest.approx(math.log(2.0))


def test_model_prior_variants_normalize_and_report_structural_masses():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=40)
    specs = {m.model_hash: m for m in graph.nodes}
    rng = np.random.default_rng(4)
    log_z = {h: float(rng.normal(-100.0, 1.0)) for h in specs}
    variants = model_prior_variants(graph.by_hash[graph.root_hash], specs, log_z)
    assert [v.penalty for v in variants] == pytest.approx([0.0, math.log(2), math.log(4)])
    for variant in variants:
        assert sum(variant.posterior.values()) == pytest.approx(1.0)
        assert sum(variant.prior.values()) == pytest.approx(1.0)
    uniform = variants[0]
    assert set(np.round(list(uniform.prior.values()), 12)) == {round(1 / 15, 12)}
    atom = "chieff.options.mean_dependence=\"linear_m1\""
    child = [h for h, s in specs.items() if s.chieff.options.get("mean_dependence") == "linear_m1"][0]
    assert uniform.atoms[atom]["posterior_mass"] == pytest.approx(uniform.posterior[child])
    penalized = variants[2]
    assert penalized.prior[child] == pytest.approx(0.25 / (1 + 14 * 0.25))


# ---------------------------------------------------------------------------
# Review findings: a narrowed prior must not hand a claim an unchecked number
# ---------------------------------------------------------------------------


def test_narrowed_reweighting_refuses_below_a_declared_ess():
    """A shift from a handful of effective draws is not a prior-robustness result.

    With ``peak_fraction ~ U(0, 0.5)`` halved about its centre (the old
    behaviour) and a posterior piled up near 0, a single surviving draw of 4096
    returns a confident-looking multi-nat shift, and neighbouring seeds return
    nothing at all. Criterion 3 then fails an edge on it.
    """
    old = PriorSpec("uniform", low=0.0, high=0.5)
    new = rescaled_prior(old, 0.5)  # U(0.125, 0.375): the centre-anchored variant
    assert (new.low, new.high) == pytest.approx((0.125, 0.375))
    rng = np.random.default_rng(2)
    values = np.clip(rng.lognormal(mean=math.log(0.038), sigma=0.35, size=4096), 1e-9, 0.5)
    sample = posterior_from_equal_weight_draws(("peak_fraction",), values[:, None])

    refused = prior_reweighting(sample, "peak_fraction", old, new)
    assert refused["reweighting_ess"] < 50.0
    assert refused["delta_log_evidence"] is None
    assert refused["flag"] == "low_reweighting_ess"
    # the raw value is kept so the refusal can be inspected, and it is large
    assert abs(refused["delta_log_evidence_unreliable"]) > 3.0

    # the same call with the threshold lowered reports the (untrustworthy) number
    allowed = prior_reweighting(sample, "peak_fraction", old, new, min_ess=0.0)
    assert allowed["delta_log_evidence"] == pytest.approx(refused["delta_log_evidence_unreliable"])
    assert allowed["flag"] is None

    # a well-supported narrowing is unaffected
    good = _truncated_normal_posterior(0.3, 0.2, -2.0, 2.0, 20000, seed=1)
    out = prior_reweighting(good, "x", PriorSpec("uniform", low=-2.0, high=2.0),
                            PriorSpec("uniform", low=-1.0, high=1.0))
    assert out["flag"] is None and out["delta_log_evidence"] is not None
    assert out["reweighting_ess"] >= 50.0


def test_a_null_at_a_prior_bound_is_narrowed_about_that_bound():
    """Halving a mixture-fraction prior about its centre deletes the tested null."""
    from gwpop_search.analysis.prior_sensitivity import null_anchors

    prior = PriorSpec("uniform", low=0.0, high=0.5)
    anchored = rescaled_prior(prior, 0.5, anchor="low")
    assert (anchored.low, anchored.high) == pytest.approx((0.0, 0.25))
    upper = rescaled_prior(prior, 0.5, anchor="high")
    assert (upper.low, upper.high) == pytest.approx((0.25, 0.5))
    lu = rescaled_prior(PriorSpec("log_uniform", low=0.03, high=0.5), 0.5, anchor="low")
    assert lu.low == pytest.approx(0.03)
    assert math.log(lu.high / lu.low) == pytest.approx(0.5 * math.log(0.5 / 0.03))
    with pytest.raises(ValueError, match="anchor"):
        rescaled_prior(prior, 0.5, anchor="centre")

    parent = baseline_model_spec("gwtc5-v1")
    removed = apply_mutation(parent, MUT["mass.family.powerlaw"])
    # the edge's null is peak_fraction = 0, at the prior's lower bound
    assert null_anchors(parent, removed) == {"peak_fraction": "lower"}

    peak = _fake_posterior(parent, {"peak_fraction": (0.02, 0.02), "peak_mu": (34.0, 2.0)}, seed=4)
    out = edge_prior_sensitivity(parent, removed, peak, _fake_posterior(removed, {}, seed=5),
                                 log_bayes_factor=-3.0)
    rows = {r["parameter"]: r for r in out["parameters"]}
    fraction = rows["peak_fraction"]
    assert fraction["null_at_prior_bound"] == "lower" and fraction["narrow_anchor"] == "low"
    narrowed = fraction["narrowed"]["new_prior"]
    # the null f = 0 is still in the narrowed prior, and the estimate is usable
    assert narrowed["low"] == pytest.approx(0.0) and narrowed["high"] == pytest.approx(0.25)
    assert fraction["narrowed"]["flag"] is None
    assert fraction["delta_log_bayes_factor_narrowed"] is not None
    # a parameter with no null at a bound keeps the centred rescaling
    assert rows["peak_mu"]["narrow_anchor"] is None
    centre = parent.priors["peak_mu"].parameters
    wide = rows["peak_mu"]["narrowed"]["new_prior"]
    assert wide["low"] + wide["high"] == pytest.approx(centre["low"] + centre["high"])


def test_low_ess_variants_are_reported_as_absent_not_as_a_number():
    """``edge_prior_sensitivity`` passes the refusal through to the variants."""
    parent = baseline_model_spec("gwtc5-v1")
    removed = apply_mutation(parent, MUT["mass.family.powerlaw"])
    peak = _fake_posterior(parent, {"peak_fraction": (0.02, 0.02)}, seed=6)
    out = edge_prior_sensitivity(parent, removed, peak, _fake_posterior(removed, {}, seed=7),
                                 log_bayes_factor=-3.0, min_reweighting_ess=1e9)
    narrowed = [v for v in out["variants"] if v["variant"] == "narrowed"]
    assert narrowed and all(v["log_bayes_factor"] is None for v in narrowed)
    assert all(v["flag"] == "low_reweighting_ess" for v in narrowed)
    assert all(v["reweighting_ess"] is not None for v in narrowed)
