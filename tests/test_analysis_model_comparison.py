import math

import numpy as np
import pytest

pytest.importorskip("jax")

from gwpop_search.analysis.edge_mc_error import ModelMCWeights  # noqa: E402
from gwpop_search.analysis.model_comparison import (  # noqa: E402
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
