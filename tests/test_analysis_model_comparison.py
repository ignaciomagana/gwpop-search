import math

import numpy as np
import pytest

pytest.importorskip("jax")

from gwpop_search.analysis.edge_mc_error import ModelMCWeights  # noqa: E402
from dataclasses import replace  # noqa: E402

from gwpop_search.analysis.model_comparison import (  # noqa: E402
    ClaimCriteria,
    ComparisonConfig,
    EvidenceSummary,
    NullCalibration,
    build_model_comparison,
    kass_raftery,
    render_markdown,
)
from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph  # noqa: E402
from gwpop_search.search import ComplexityModelPrior, UniformModelPrior  # noqa: E402

IDENTITY = {"data": {"pe_sha256": "x", "selection_sha256": "y"}, "hbi_config": {"rate_treatment": "shape"}}


def _mc(label, tilt=0.0, n_sel=100):
    """Small synthetic posterior-averaged weights (2 events, one campaign)."""
    sel = np.full(n_sel, 1.0 / n_sel) * (1.0 + tilt * np.linspace(-1, 1, n_sel))
    sel /= sel.sum()
    return ModelMCWeights(
        label=label,
        names=("x",),
        likelihood_identity=IDENTITY,
        event_names=("E1", "E2"),
        pe_counts=np.array([4.0, 4.0]),
        campaign_ids=("C",),
        campaign_n_draw=np.array([1000.0]),
        omega_event_bar=np.full((2, 4), 0.25),
        omega_selection_bar=sel,
        fraction_bar=np.array([1.0]),
        diag_event=np.full(2, 0.25e-3),
        diag_selection=1e-3 * float(np.sum(sel**2)),
        diag_fraction=1e-3 / 1000.0,
        sum_w2=1e-3,
        expected_event_sigma2=np.zeros(2),
        expected_selection_sigma2=1e-4,
        expected_variance=0.4,
        variance_quantiles={"q0.9": 0.5},
        selection_neff_ok_fraction=1.0,
        min_selection_ess=500.0,
        min_event_ess=3.0,
        n_points=1000,
        kish_ess=1000.0,
        source="synthetic",
    )


def _graph_and_evidence(strong_mutation="chieff.mean.linear_m1", strong_lnbf=8.0):
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=1, max_models=40)
    root = graph.root_hash
    child_of = {edge.mutation_id: edge.child_hash for edge in graph.edges}
    evidences = {}
    for model in graph.nodes:
        h = model.model_hash
        base = -100.0 if h == root else (-100.0 + strong_lnbf if h == child_of[strong_mutation] else -101.0)
        evidences[h] = EvidenceSummary(
            model_hash=h, log_evidence=base, estimates=(base - 0.05, base + 0.05),
            reported_errors=(0.1, 0.1), information=12.0, nlive=1000, gates_passed=True,
        )
    return graph, evidences, root, child_of[strong_mutation]


def _full_inputs(graph, root, child, mutation="chieff.mean.linear_m1"):
    mc = {m.model_hash: _mc(m.model_hash[:6], tilt=0.01 if m.model_hash == child else 0.0) for m in graph.nodes}
    return dict(
        mc_weights=mc,
        sddr={mutation: {"status": "agree", "classification": "exact"}},
        prior_sensitivity={(root, child): {"variants": [
            {"parameter": "chi_mu_m1_slope", "variant": "narrowed", "log_bayes_factor": 8.6},
            {"parameter": "chi_mu_m1_slope", "variant": "widened", "log_bayes_factor": 7.3},
        ]}},
        null_calibrations=[
            NullCalibration("root-median", "max_edge_log_bayes_factor", 8.0, 0, 200),
            NullCalibration("root-q90", "max_edge_log_bayes_factor", 8.0, 0, 200),
        ],
        stress={(root, child): [7.5, 8.2]},
        nearby={mutation: [6.0]},
        loo={"edges": [{"parent_hash": root, "child_hash": child, "any_sign_reversal": False,
                        "n_flagged_events": 0, "max_abs_delta_log_bayes_factor": 0.4}]},
    )


def _claim(report, mutation):
    return next(c for c in report["claims"] if c["mutation_id"] == mutation)


def test_budget_sigma_mc_falls_back_to_the_plug_in_when_the_u_statistic_is_negative():
    """The unbiased U-statistic can go slightly negative; the budget must not report 0.

    Under common random numbers the true edge variance is tiny, so the
    U-statistic lands below zero on finite samples (measured in the low-variance
    toy regime: -3.9e-5 against a realized 1.25e-3). The plug-in is non-negative
    and tracked the realized variance to 2% there, so the budget uses it.
    """
    from gwpop_search.analysis.model_comparison import _budget_sigma_mc

    # U-statistic negative -> plug-in wins
    assert _budget_sigma_mc({"variance": -3.9e-5, "variance_plug_in": 1.27e-3, "sigma": 0.0}) == pytest.approx(
        math.sqrt(1.27e-3)
    )
    # U-statistic positive -> it is the (tighter, unbiased) estimate, plug-in ignored
    assert _budget_sigma_mc({"variance": 0.0704, "variance_plug_in": 0.107, "sigma": math.sqrt(0.0704)}) == (
        pytest.approx(math.sqrt(0.0704))
    )
    # both negative -> 0, never NaN
    assert _budget_sigma_mc({"variance": -1e-6, "variance_plug_in": -2e-6, "sigma": 0.0}) == 0.0
    # a supplied sigma carries no variance decomposition
    assert _budget_sigma_mc({"sigma": 0.25, "bias": 0.0, "source": "supplied"}) == pytest.approx(0.25)


def test_report_error_budget_probabilities_and_a_supported_claim():
    graph, evidences, root, child = _graph_and_evidence()
    prior = ComplexityModelPrior(math.log(2.0))
    report = build_model_comparison(graph, evidences, prior, **_full_inputs(graph, root, child))
    assert report["evidence_coverage"]["complete"]
    assert report["deduplication"]["n_effective_models"] == 15
    edge = next(e for e in report["edges"] if e["mutation_id"] == "chieff.mean.linear_m1")
    assert edge["log_bayes_factor"] == pytest.approx(8.0)
    # formula (6): max(repeat std, mean logzerr) / sqrt(R), in quadrature with the joint MC error
    sigma_ns = max(np.std([-0.05, 0.05], ddof=1), 0.1) / math.sqrt(2)
    assert edge["sigma_ns_parent"] == pytest.approx(sigma_ns)
    assert edge["sigma_total"] == pytest.approx(math.sqrt(2 * sigma_ns**2 + edge["sigma_mc"] ** 2))
    assert edge["log_posterior_odds"] == pytest.approx(8.0 - math.log(2.0))
    probs = {h: row["posterior_probability"] for h, row in report["models"].items()}
    assert sum(probs.values()) == pytest.approx(1.0)
    weights = {h: evidences[h].log_evidence + prior.log_prior(graph.by_hash[h], root=graph.by_hash[root])
               for h in probs}
    top = max(weights.values())
    norm = sum(math.exp(w - top) for w in weights.values())
    assert probs[child] == pytest.approx(math.exp(weights[child] - top) / norm)
    propagated = report["models"][child]["posterior_probability_propagated"]
    assert propagated["q16"] <= probs[child] + 1e-9 <= propagated["q84"] + 1e-9
    atom = report["structural"]["atoms"]['chieff.options.mean_dependence="linear_m1"']
    assert atom["posterior_mass"] == pytest.approx(probs[child])
    assert atom["log_bayes_factor"] > 3
    claim = _claim(report, "chieff.mean.linear_m1")
    assert claim["status"] == "claimed", claim
    weak = _claim(report, "chieff.mean.linear_q")
    assert weak["status"] == "not_claimed"
    assert weak["criteria"]["strength"]["status"] == "fail"
    assert report["search_statistics"]["max_edge_log_bayes_factor"] == pytest.approx(8.0)
    text = render_markdown(report)
    assert "| chieff.mean.linear_m1 | **claimed**" in text


@pytest.mark.parametrize(
    "change, expected, criterion",
    [
        ({"null_calibrations": None}, "incomplete", "null_calibration"),
        ({"null_calibrations": [NullCalibration("n", "max_edge_log_bayes_factor", 8.0, 5, 200)]},
         "not_claimed", "null_calibration"),
        ({"null_calibrations": [NullCalibration("n", "max_edge_log_bayes_factor", 8.0, 0, 50)]},
         "not_claimed", "null_calibration"),
        ({"stress": "reversal"}, "not_claimed", "stress"),
        ({"sddr": {"chieff.mean.linear_m1": {"status": "disagree", "classification": "exact"}}},
         "not_claimed", "sddr"),
        ({"prior_sensitivity": "weak"}, "not_claimed", "prior_robustness"),
        ({"mc_weights": None}, "incomplete", "numerics"),
    ],
)
def test_claim_criteria_fail_or_stay_incomplete(change, expected, criterion):
    graph, evidences, root, child = _graph_and_evidence()
    inputs = _full_inputs(graph, root, child)
    for key, value in change.items():
        if value == "reversal":
            value = {(root, child): [7.5, -0.5]}
        elif value == "weak":
            value = {(root, child): {"variants": [{"variant": "widened", "log_bayes_factor": 0.8}]}}
        inputs[key] = value
    report = build_model_comparison(graph, evidences, ComplexityModelPrior(math.log(2.0)), **inputs)
    claim = _claim(report, "chieff.mean.linear_m1")
    assert claim["status"] == expected
    assert claim["criteria"][criterion]["status"] in {"fail", "missing", "incomplete"}


def test_resolution_limited_nulls_cannot_pass():
    graph, evidences, root, child = _graph_and_evidence()
    inputs = _full_inputs(graph, root, child)
    inputs["null_calibrations"] = [NullCalibration("n", "max_edge_log_bayes_factor", 8.0, 0, 50)]
    report = build_model_comparison(graph, evidences, UniformModelPrior(), **inputs)
    null = _claim(report, "chieff.mean.linear_m1")["criteria"]["null_calibration"]
    assert null["nulls"][0]["resolution_limited"] and null["status"] == "fail"


def test_aliases_are_merged_before_normalizing():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=2, max_models=40)
    rng = np.random.default_rng(0)
    evidences = {}
    for model in graph.nodes:
        value = float(rng.normal(-100.0, 0.5))
        evidences[model.model_hash] = EvidenceSummary(model.model_hash, value, (value - 0.1, value + 0.1), (0.1, 0.1))
    report = build_model_comparison(graph, evidences, ComplexityModelPrior(math.log(2.0)),
                                    config=ComparisonConfig(n_propagation=200))
    dedup = report["deduplication"]
    assert dedup["n_nodes"] == 40 and dedup["n_effective_models"] == 36
    assert len(report["models"]) == 36
    assert sum(row["posterior_probability"] for row in report["models"].values()) == pytest.approx(1.0)
    merged = dedup["merged_evidence"]
    assert sorted(len(v["merged"]) for v in merged.values()) == [2, 2, 3]
    for head, info in merged.items():
        estimates = [x for h in info["merged"] for x in evidences[h].estimates]
        assert report["models"][head]["evidence"]["log_evidence"] == pytest.approx(np.mean(estimates))
    # aliases whose evidences disagree beyond their errors are flagged
    assert set(dedup["inconsistent_alias_evidence"]) <= set(merged)


def test_incomplete_coverage_leaves_probabilities_undefined():
    graph, evidences, root, child = _graph_and_evidence()
    evidences.pop(child)
    report = build_model_comparison(graph, evidences, UniformModelPrior())
    assert not report["evidence_coverage"]["complete"]
    assert report["evidence_coverage"]["missing"] == [child]
    assert report["structural"] is None and report["propagation"] is None
    assert all(row["posterior_probability"] is None for row in report["models"].values())


def test_evidence_summary_errors_and_scale():
    s = EvidenceSummary("h", -10.0, (-10.3, -9.7), (0.1, 0.2), information=40.0, nlive=1000)
    repeat = np.std([-10.3, -9.7], ddof=1)
    assert s.ns_error("formula6") == pytest.approx(repeat / math.sqrt(2))
    assert s.ns_error("formula6", kappa_hat=3.0) == pytest.approx(3.0 * math.sqrt(0.04) / math.sqrt(2))
    assert s.ns_error("conservative") == pytest.approx(repeat)
    with pytest.raises(ValueError, match="mean"):
        EvidenceSummary("h", -5.0, (-10.3, -9.7), (0.1, 0.2))
    assert kass_raftery(0.5).startswith("not worth")
    assert kass_raftery(-4.0) == "strong (for parent)"
    assert kass_raftery(6.0) == "very strong (for child)"


# ---------------------------------------------------------------------------
# Review findings: the claim pipeline must not pass on inputs it never checked
# ---------------------------------------------------------------------------


def test_an_sddr_that_could_not_be_computed_is_not_a_satisfied_criterion():
    """``not_estimable`` on an SDDR-eligible edge blocks the claim.

    ``sddr_edge_check`` returns ``not_estimable`` for three *method* failures
    (the VW support condition fails, too few draws near the null, the density
    is not estimable with no NS bound). Scoring those ``not_applicable`` made
    them count as a pass, so an edge could be claimed with no cross-check at
    all -- and ``mass.family.pl_two_peak`` (vw_required, vw_first_form_valid
    false) returns exactly that, deterministically, on the production graph.
    """
    graph, evidences, root, child = _graph_and_evidence()
    prior = ComplexityModelPrior(math.log(2.0))
    inputs = _full_inputs(graph, root, child)

    for reason in ("VW factor needs supp pi_S within supp pi_L",
                   "too few posterior samples near the null for VW"):
        inputs["sddr"] = {"chieff.mean.linear_m1": {
            "status": "not_estimable", "classification": "approximate", "details": {"reason": reason}}}
        claim = _claim(build_model_comparison(graph, evidences, prior, **inputs), "chieff.mean.linear_m1")
        assert claim["criteria"]["sddr"]["status"] == "missing", claim["criteria"]["sddr"]
        assert claim["criteria"]["sddr"]["reason"] == reason
        assert claim["status"] == "incomplete"

    # an eligible edge declared not_applicable is equally uncheckable
    inputs["sddr"] = {"chieff.mean.linear_m1": {"status": "not_applicable", "classification": "exact"}}
    claim = _claim(build_model_comparison(graph, evidences, prior, **inputs), "chieff.mean.linear_m1")
    assert claim["criteria"]["sddr"]["status"] == "missing"
    assert claim["status"] == "incomplete"

    # only an evidence_only edge is genuinely not applicable, and it still claims
    inputs["sddr"] = {"chieff.mean.linear_m1": {"status": "not_applicable", "classification": "evidence_only"}}
    claim = _claim(build_model_comparison(graph, evidences, prior, **inputs), "chieff.mean.linear_m1")
    assert claim["criteria"]["sddr"]["status"] == "not_applicable"
    assert claim["status"] == "claimed"


def test_the_production_graph_has_an_edge_whose_sddr_is_never_estimable():
    """The finding above is reachable on the real gwtc5-v1 depth-1 graph."""
    from gwpop_search.analysis.sddr import classify_graph_edges

    graph = enumerate_model_graph(baseline_model_spec("gwtc5-v1"), max_depth=1, max_models=40)
    rows = {row.mutation_id: row for row in classify_graph_edges(graph)}
    two_peak = rows["mass.family.pl_two_peak"]
    assert two_peak.sddr_eligible and two_peak.vw_required and not two_peak.vw_first_form_valid


def test_a_null_calibration_must_describe_this_report():
    """A stale or mis-targeted null campaign cannot satisfy criterion 5."""
    graph, evidences, root, child = _graph_and_evidence()
    prior = ComplexityModelPrior(math.log(2.0))
    inputs = _full_inputs(graph, root, child)

    # (a) the right statistic with a value from somewhere else
    inputs["null_calibrations"] = [NullCalibration("stale", "max_edge_log_posterior_odds", 0.3, 0, 200)]
    report = build_model_comparison(graph, evidences, prior, **inputs)
    claim = _claim(report, "chieff.mean.linear_m1")
    null = claim["criteria"]["null_calibration"]
    assert null["status"] == "fail" and claim["status"] == "not_claimed"
    row = null["nulls"][0]
    assert row["matches_report_statistic"] is False
    assert row["report_statistic"] == pytest.approx(report["search_statistics"]["max_edge_log_posterior_odds"])
    assert row["p_value"] <= 0.01  # it would have passed on the p-value alone

    # (b) the report's own value passes
    inputs["null_calibrations"] = [NullCalibration(
        "matched", "max_edge_log_posterior_odds",
        report["search_statistics"]["max_edge_log_posterior_odds"], 0, 200)]
    claim = _claim(build_model_comparison(graph, evidences, prior, **inputs), "chieff.mean.linear_m1")
    assert claim["criteria"]["null_calibration"]["status"] == "pass"
    assert claim["status"] == "claimed"

    # (c) a statistic this report does not compute cannot be tied to it
    inputs["null_calibrations"] = [NullCalibration("other", "some_other_statistic", 8.0, 0, 200)]
    claim = _claim(build_model_comparison(graph, evidences, prior, **inputs), "chieff.mean.linear_m1")
    assert claim["criteria"]["null_calibration"]["status"] == "missing"
    assert claim["criteria"]["null_calibration"]["nulls"][0]["report_statistic"] is None
    assert claim["status"] == "incomplete"


def test_a_null_calibration_cannot_pass_on_a_partially_scored_graph():
    """With evidence missing, the observed statistic is a max over a subset."""
    graph, evidences, root, child = _graph_and_evidence()
    inputs = _full_inputs(graph, root, child)
    dropped = next(m.model_hash for m in graph.nodes if m.model_hash not in {root, child})
    evidences.pop(dropped)
    inputs["null_calibrations"] = [NullCalibration("n", "max_edge_log_bayes_factor", 8.0, 0, 200)]
    report = build_model_comparison(graph, evidences, ComplexityModelPrior(math.log(2.0)), **inputs)
    assert not report["evidence_coverage"]["complete"]
    claim = _claim(report, "chieff.mean.linear_m1")
    null = claim["criteria"]["null_calibration"]
    assert null["status"] == "incomplete" and null["evidence_coverage_complete"] is False


def test_mixing_fidelity_rungs_across_models_is_detected():
    """D4's calibrated statistic must come from one rung; a mixture is not it."""
    graph, evidences, root, child = _graph_and_evidence()
    inputs = _full_inputs(graph, root, child)
    f3 = "bound=multi,dlogz=0.1,nlive=1000"
    f4 = "bound=multi,dlogz=0.05,nlive=2000"
    evidences = {h: replace(e, rung=f3) for h, e in evidences.items()}
    homogeneous = build_model_comparison(graph, evidences, ComplexityModelPrior(math.log(2.0)), **inputs)
    assert homogeneous["evidence_coverage"]["rung_homogeneous"]
    assert homogeneous["evidence_coverage"]["distinct_rungs"] == [f3]
    assert _claim(homogeneous, "chieff.mean.linear_m1")["status"] == "claimed"

    evidences[child] = replace(evidences[child], rung=f4)
    mixed = build_model_comparison(graph, evidences, ComplexityModelPrior(math.log(2.0)), **inputs)
    coverage = mixed["evidence_coverage"]
    assert not coverage["rung_homogeneous"] and sorted(coverage["distinct_rungs"]) == sorted([f3, f4])
    claim = _claim(mixed, "chieff.mean.linear_m1")
    assert claim["criteria"]["numerics"]["status"] == "fail"
    assert claim["criteria"]["numerics"]["fidelity_rungs"] == {"parent": f3, "child": f4}
    assert claim["status"] == "not_claimed"


def test_gmc4_gates_the_same_sigma_mc_the_budget_uses():
    """The gate and the error budget must not disagree about what sigma_MC is.

    ``EdgeMCError.variance`` is the unbiased U-statistic; under common random
    numbers it lands below zero on finite samples and ``sigma`` then reports 0.
    The budget already falls back to the non-negative plug-in, so gating G-MC4
    on ``sigma`` passed on a value known to be wrong.
    """
    from gwpop_search.analysis.edge_mc_error import edge_mc_error
    from gwpop_search.analysis.model_comparison import _budget_sigma_mc

    graph, evidences, root, child = _graph_and_evidence()
    inputs = _full_inputs(graph, root, child)
    # a diagonal large enough that the U-statistic correction overshoots the
    # (tiny, common-random-number) edge variance, exactly as it does in the
    # low-variance regime of validation/analysis_estimators/mc_error_lowV.json
    mc_weights = {h: replace(w, diag_event=np.full(2, 5e-4))
                  for h, w in inputs["mc_weights"].items()}
    inputs["mc_weights"] = mc_weights
    raw = edge_mc_error(mc_weights[child], mc_weights[root]).to_dict()
    assert raw["variance"] < 0.0 and raw["variance_negative"] and raw["sigma"] == 0.0
    plug_in_sigma = math.sqrt(raw["variance_plug_in"])
    assert plug_in_sigma > 0.0
    assert _budget_sigma_mc(raw) == pytest.approx(plug_in_sigma)

    criteria = ClaimCriteria(gmc_max_edge_sigma=0.5 * plug_in_sigma)
    report = build_model_comparison(
        graph, evidences, ComplexityModelPrior(math.log(2.0)),
        config=ComparisonConfig(n_propagation=200, criteria=criteria), **inputs)
    edge = next(e for e in report["edges"] if e["mutation_id"] == "chieff.mean.linear_m1")
    claim = _claim(report, "chieff.mean.linear_m1")
    gate = claim["criteria"]["numerics"]["gmc_edge"]
    # the gate reads the budget's sigma_MC, not the floored U-statistic
    assert gate["G-MC4_sigma"] == pytest.approx(edge["sigma_mc"]) == pytest.approx(plug_in_sigma)
    assert gate["G-MC4_sigma_u_statistic"] == 0.0
    assert gate["G-MC4"] is False
    assert claim["criteria"]["numerics"]["status"] == "fail"


def test_an_atom_absent_from_the_variants_is_missing_not_failed():
    """A missing input must not be reported as a scientific failure."""
    graph, evidences, root, child = _graph_and_evidence()
    inputs = _full_inputs(graph, root, child)
    report = build_model_comparison(graph, evidences, ComplexityModelPrior(math.log(2.0)), **inputs)
    claim = _claim(report, "chieff.mean.linear_m1")
    assert claim["criteria"]["prior_robustness"]["status"] == "pass"

    # the same report with the edge's atom removed from every model-prior variant
    import gwpop_search.analysis.model_comparison as mc_mod

    row = next(e for e in report["edges"] if e["mutation_id"] == "chieff.mean.linear_m1")
    variants = [{"penalty_per_axis": 0.0, "atoms": {}}]
    out = mc_mod._evaluate_claim(
        row, merged={h: EvidenceSummary(h, -100.0, gates_passed=True) for h in (root, child)},
        mc_weights=None, structural=report["structural"], variants=variants, sddr=None,
        prior_sensitivity=None, null_calibrations=None, stress=None, nearby=None, loo=None,
        criteria=ClaimCriteria(),
    )
    assert out["criteria"]["prior_robustness"]["status"] == "missing"
    assert out["criteria"]["prior_robustness"]["model_prior_variants"][0]["posterior_mass"] is None
    assert out["status"] == "incomplete"

    # a present mass below the threshold is still a failure
    variants = [{"penalty_per_axis": 0.0, "atoms": {row["atom"]: {"posterior_mass": 0.1}}}]
    out = mc_mod._evaluate_claim(
        row, merged={h: EvidenceSummary(h, -100.0, gates_passed=True) for h in (root, child)},
        mc_weights=None, structural=report["structural"], variants=variants, sddr=None,
        prior_sensitivity=None, null_calibrations=None, stress=None, nearby=None, loo=None,
        criteria=ClaimCriteria(),
    )
    assert out["criteria"]["prior_robustness"]["status"] == "fail"


def test_edge_inputs_keyed_by_graph_hashes_reach_an_aliased_edge():
    """Prior-sensitivity and stress inputs are canonicalized like mc_weights."""
    from gwpop_search.analysis.structure import find_alias_groups

    graph = enumerate_model_graph(baseline_model_spec(), max_depth=2, max_models=40)
    aliases = find_alias_groups(graph.nodes)
    # an edge whose child is an alias member rather than its group's head
    edge = next(e for e in graph.edges
                if aliases.canonical[e.child_hash] != e.child_hash
                and aliases.canonical[e.parent_hash] != aliases.canonical[e.child_hash])
    rng = np.random.default_rng(0)
    evidences = {}
    for model in graph.nodes:
        value = float(rng.normal(-100.0, 0.5))
        evidences[model.model_hash] = EvidenceSummary(
            model.model_hash, value, (value - 0.1, value + 0.1), (0.1, 0.1), gates_passed=True)
    graph_key = (edge.parent_hash, edge.child_hash)
    report = build_model_comparison(
        graph, evidences, ComplexityModelPrior(math.log(2.0)),
        config=ComparisonConfig(n_propagation=200),
        prior_sensitivity={graph_key: {"variants": [{"parameter": "x", "variant": "narrowed",
                                                     "log_bayes_factor": 5.0}]}},
        stress={graph_key: [5.0, 5.1]},
    )
    claim = next(c for c in report["claims"]
                 if (c["parent_hash"], c["child_hash"])
                 == (aliases.canonical[edge.parent_hash], aliases.canonical[edge.child_hash])
                 and c["mutation_id"] == edge.mutation_id)
    assert claim["criteria"]["prior_robustness"]["width_variants"] is not None
    assert claim["criteria"]["stress"]["event_drop"]["values"] == [5.0, 5.1]
