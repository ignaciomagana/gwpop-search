"""The v2 claim table (criteria D1-D6) on mock analysis outputs."""

import copy
import json

import pytest

from gwpop_search.analysis.claims_v2 import (
    DISFAVOURED,
    INCONCLUSIVE,
    SUPPORTED,
    alt_root_edge_values,
    build_claim_table,
    render_claims_markdown,
)
from gwpop_search.analysis.ppc import PPC_FORMAT, PREDECLARED_STATISTICS
from gwpop_search.grammar import baseline_model_spec, enumerate_model_graph
from gwpop_search.grammar.paths import graph_edge_keys, mutation_paths

ROOT = "r" * 64


def _edge(child, lnbf, *, sigma=0.5, bias=0.1, mutation=None, parent=ROOT):
    return {
        "parent_hash": parent, "child_hash": child, "graph_parent_hash": parent, "graph_child_hash": child,
        "mutation_id": mutation or f"mut.{child[:4]}", "atom": f"atom.{child[:4]}",
        "log_bayes_factor": lnbf, "sigma_total": sigma, "mc_error_included": True, "mc": {"bias": bias},
    }


def _width(lnbf, params=("slope",)):
    return [
        {"parameter": p, "variant": side, "log_bayes_factor": lnbf + delta}
        for p in params for side, delta in (("narrowed", 0.4), ("widened", -0.7))
    ]


def _claim(edge, *, gates=(True, True), width=None):
    return {
        "parent_hash": edge["parent_hash"], "child_hash": edge["child_hash"], "mutation_id": edge["mutation_id"],
        "atom": edge["atom"],
        "criteria": {
            "numerics": {"production_gates": list(gates), "rung_homogeneous": True,
                         "gmc_models": {edge["child_hash"]: {"G-MC1": False}}},
            "prior_robustness": {"width_variants": _width(edge["log_bayes_factor"]) if width is None else width},
            "sddr": {"status": "pass"},
        },
    }


def _report(edges, claims=None, child_probs=None):
    claims = claims if claims is not None else [_claim(e) for e in edges]
    probs = {ROOT: 0.1}
    for e in edges:
        probs[e["child_hash"]] = (child_probs or {}).get(e["child_hash"], 0.9)
    variants = [{"penalty_per_axis": lam, "posterior_model_probabilities": dict(probs)} for lam in (0.0, 0.69, 1.39)]
    return {"graph_root_hash": ROOT, "edges": edges, "claims": claims, "model_prior_variants": variants}


def _sddr(edges, diff=0.1):
    return {"edges": [
        {"parent_hash": e["parent_hash"], "child_hash": e["child_hash"], "mutation_id": e["mutation_id"],
         "classification": "exact", "status": "agree", "log_bf_child_over_parent_ns": e["log_bayes_factor"],
         "log_bf_child_over_parent_sddr": e["log_bayes_factor"] - diff, "sigma_sddr": 0.05}
        for e in edges
    ]}


def _ppc(model_hash, p=0.5, **over):
    payload = {
        "format_version": PPC_FORMAT, "model_hash": model_hash, "identity_verified": True,
        "config": {"alpha": 0.01, "n_draws": 2000},
        "statistics": {name: {"p_value": p} for name in PREDECLARED_STATISTICS},
        "diagnostics": {"reliable": True, "alpha_resolvable": True}, "borderline_statistics": [],
    }
    payload.update(over)
    return payload


def _alt(edges, sign=1.0, skip=(), inapplicable=()):
    values = {e["mutation_id"]: sign * e["log_bayes_factor"] for e in edges if e["mutation_id"] not in skip}
    return {"edges": values, "inapplicable": list(inapplicable)}


def _taper2(edges, sign=1.0):
    return [{"parent_hash": e["parent_hash"], "child_hash": e["child_hash"],
             "log_bayes_factor": sign * e["log_bayes_factor"], "valid": True} for e in edges]


def _inputs(edges, **over):
    base = dict(
        sddr=_sddr(edges),
        taper2=_taper2(edges),
        alt_roots={"A1": _alt(edges), "A2": _alt(edges)},
        ppc={e["child_hash"]: _ppc(e["child_hash"]) for e in edges},
        taper_mass={h: 0.01 for h in [ROOT] + [e["child_hash"] for e in edges]},
    )
    base.update(over)
    return base


def _row(table, child):
    return next(r for r in table["edges"] if r["child_hash"] == child)


def test_supported_requires_all_six_and_discloses_trials():
    edges = [_edge("a" * 64, 9.0), _edge("b" * 64, 0.4)]
    table = build_claim_table(_report(edges), **_inputs(edges))
    row = _row(table, "a" * 64)
    assert row["label"] == SUPPORTED
    assert row["n_atoms_tried"] == 2 == table["n_atoms_tried"]
    assert row["D2_strength"]["lower"] == pytest.approx(9.0 - 1.0 - 0.1)
    assert row["D1_numerics"]["gmc_failed_reported_not_binding"] == [f"G-MC1({'a' * 8})"]
    assert row["flags"] == []
    assert _row(table, "b" * 64)["label"] == INCONCLUSIVE
    md = render_claims_markdown(table)
    assert "SUPPORTED** (2 atoms tried)" in md
    json.dumps(table)  # JSON-ready


def test_bias_is_subtracted_in_d2():
    # ln BF - 2 sigma = 3.05 >= 3, but minus |bias| = 2.95 < 3
    edges = [_edge("a" * 64, 4.05, sigma=0.5, bias=-0.1)]
    row = build_claim_table(_report(edges), **_inputs(edges))["edges"][0]
    assert row["D2_strength"]["status"] == "fail"
    assert row["label"] == INCONCLUSIVE


@pytest.mark.parametrize("case", ["d1_gate", "d3_width", "d3_prior", "d3_taper2", "d4", "d5_reverse", "d5_missing", "d6"])
def test_every_criterion_blocks_supported(case):
    edge = _edge("a" * 64, 9.0)
    edges = [edge]
    report = _report(edges)
    inputs = _inputs(edges)
    if case == "d1_gate":
        report["claims"] = [_claim(edge, gates=(True, False))]
    elif case == "d3_width":
        report["claims"] = [_claim(edge, width=[{"parameter": "slope", "variant": "narrowed", "log_bayes_factor": 5.0},
                                               {"parameter": "slope", "variant": "widened", "log_bayes_factor": 0.8}])]
    elif case == "d3_prior":
        report = _report(edges, child_probs={edge["child_hash"]: 0.2})  # child/(child+parent) = 2/3 < 0.75
    elif case == "d3_taper2":
        inputs["taper2"] = _taper2(edges, sign=-1.0)
    elif case == "d4":
        inputs["sddr"] = _sddr(edges, diff=2.0)
    elif case == "d5_reverse":
        inputs["alt_roots"] = {"A1": _alt(edges), "A2": _alt(edges, sign=-1.0)}
    elif case == "d5_missing":
        inputs["alt_roots"] = {"A1": _alt(edges)}
    elif case == "d6":
        inputs["ppc"] = {edge["child_hash"]: _ppc(edge["child_hash"], p=0.004)}
    row = build_claim_table(report, **inputs)["edges"][0]
    assert row["label"] == INCONCLUSIVE
    expected = {
        "d1_gate": ("D1_numerics", "fail"), "d3_width": ("D3_prior_sensitivity", "fail"),
        "d3_prior": ("D3_prior_sensitivity", "fail"), "d3_taper2": ("D3_prior_sensitivity", "fail"),
        "d4": ("D4_sddr", "flag_investigate"), "d5_reverse": ("D5_alternative_roots", "fail"),
        "d5_missing": ("D5_alternative_roots", "incomplete"), "d6": ("D6_ppc", "fail"),
    }[case]
    assert row[expected[0]]["status"] == expected[1]
    if case == "d4":
        assert "sddr_disagreement_requires_human_review" in row["flags"]


def test_d3_rerun_path_factor_check_and_missing_taper2():
    edge = _edge("a" * 64, 9.0)
    width = [{"parameter": "slope", "variant": "narrowed", "log_bayes_factor": 9.2},
             {"parameter": "slope", "variant": "widened", "log_bayes_factor": None, "flag": "posterior_not_contained"}]
    report = _report([edge], claims=[_claim(edge, width=width)])
    inputs = _inputs([edge])
    row = build_claim_table(report, **inputs)["edges"][0]
    assert row["D3_prior_sensitivity"]["width_status"] == "incomplete"
    rerun = [{"parent_hash": ROOT, "child_hash": "a" * 64, "parameter": "slope", "variant": "widened",
              "log_bayes_factor": 8.1, "valid": True, "rerun_job": "j1"}]
    row = build_claim_table(report, d3_reruns=rerun, **inputs)["edges"][0]
    assert row["D3_prior_sensitivity"]["status"] == "pass"
    assert row["D3_prior_sensitivity"]["reruns_used"][0]["rerun_job"] == "j1"
    assert row["label"] == SUPPORTED
    bad = [dict(rerun[0], valid=False)]
    row = build_claim_table(copy.deepcopy(report), d3_reruns=bad, **inputs)["edges"][0]
    assert row["D3_prior_sensitivity"]["width_status"] == "incomplete"
    # only one side of a parameter: incomplete
    one_side = _report([edge], claims=[_claim(edge, width=width[:1])])
    assert build_claim_table(one_side, **inputs)["edges"][0]["D3_prior_sensitivity"]["width_status"] == "incomplete"
    # a prior-sensitivity run at a factor other than 2 is not the pre-declared halving/doubling
    sens = {"edges": [{"parent_hash": ROOT, "child_hash": "a" * 64, "factor": 3.0}]}
    row = build_claim_table(_report([edge]), prior_sensitivity=sens, **inputs)["edges"][0]
    assert row["D3_prior_sensitivity"]["width_status"] == "incomplete"
    # a claimed edge needs its taper-2 rerun
    row = build_claim_table(_report([edge]), **dict(inputs, taper2=[]))["edges"][0]
    assert row["D3_prior_sensitivity"]["taper2_status"] == "missing" and row["label"] == INCONCLUSIVE


def test_disfavoured_needs_d1_and_sign_stability_only():
    edge = _edge("d" * 64, -10.0, sigma=1.0, bias=0.5)
    edges = [edge]
    inputs = _inputs(edges, ppc={}, taper2=[], alt_roots={})
    row = build_claim_table(_report(edges), **inputs)["edges"][0]
    assert row["D2_strength"]["upper"] == pytest.approx(-7.5)
    assert row["D3_prior_sensitivity"]["status"] == "sign_stable"
    assert row["label"] == DISFAVOURED
    assert "negative_edge_without_taper2_rerun" in row["flags"]
    # a sign flip of a width variant, or of the taper-2 rerun, blocks it
    flipped = [{"parameter": "slope", "variant": "narrowed", "log_bayes_factor": -9.0},
               {"parameter": "slope", "variant": "widened", "log_bayes_factor": 0.3}]
    row = build_claim_table(_report(edges, claims=[_claim(edge, width=flipped)]), **inputs)["edges"][0]
    assert row["label"] == INCONCLUSIVE
    row = build_claim_table(_report(edges), **dict(inputs, taper2=_taper2(edges, sign=-1.0)))["edges"][0]
    assert row["label"] == INCONCLUSIVE
    # D1 failing blocks it
    row = build_claim_table(_report(edges, claims=[_claim(edge, gates=(False, True))]), **inputs)["edges"][0]
    assert row["label"] == INCONCLUSIVE
    # not strong enough: upper = -2.5
    weak = _edge("d" * 64, -5.0, sigma=1.0, bias=0.5)
    assert build_claim_table(_report([weak]), **inputs)["edges"][0]["label"] == INCONCLUSIVE


def test_d4_floor_and_bound_consistent():
    edge = _edge("a" * 64, 9.0, sigma=0.1)
    inputs = _inputs([edge], sddr=_sddr([edge], diff=0.45))
    row = build_claim_table(_report([edge]), **inputs)["edges"][0]
    assert row["D4_sddr"]["band"] == pytest.approx(0.5)  # max(0.5, 2 * hypot(0.1, 0.05))
    assert row["D4_sddr"]["status"] == "agree"
    bound = {"edges": [{"parent_hash": ROOT, "child_hash": "a" * 64, "mutation_id": edge["mutation_id"],
                        "classification": "exact", "status": "bound_consistent"}]}
    row = build_claim_table(_report([edge]), **dict(inputs, sddr=bound))["edges"][0]
    assert row["D4_sddr"]["status"] == "bound_consistent" and row["label"] == SUPPORTED
    family = {"edges": [{"parent_hash": ROOT, "child_hash": "a" * 64, "mutation_id": edge["mutation_id"],
                         "classification": "evidence_only", "status": "not_applicable"}]}
    row = build_claim_table(_report([edge]), **dict(inputs, sddr=family))["edges"][0]
    assert row["D4_sddr"]["status"] == "not_applicable" and row["label"] == SUPPORTED
    not_est = {"edges": [dict(bound["edges"][0], status="not_estimable")]}
    row = build_claim_table(_report([edge]), **dict(inputs, sddr=not_est))["edges"][0]
    assert row["D4_sddr"]["status"] == "missing" and row["label"] == INCONCLUSIVE


def test_d5_not_applicable_on_one_root_and_loo_reported():
    edge = _edge("a" * 64, 9.0, mutation="pairing.beta_per_component")
    alt = {"A1": _alt([edge], skip=(edge["mutation_id"],), inapplicable=(edge["mutation_id"],)), "A2": _alt([edge])}
    loo = {"edges": [{"parent_hash": ROOT, "child_hash": "a" * 64, "any_sign_reversal": False, "n_flagged_events": 1}]}
    row = build_claim_table(_report([edge]), **_inputs([edge], alt_roots=alt), loo=loo)["edges"][0]
    d5 = row["D5_alternative_roots"]
    assert d5["alt_roots"]["A1"]["status"] == "not_applicable"
    assert d5["status"] == "pass" and row["label"] == SUPPORTED
    assert "d5_alt_root_not_applicable_for_this_atom" in row["flags"]
    assert d5["psis_loo_reported_not_binding"]["n_flagged_events"] == 1
    both = {"A1": alt["A1"], "A2": alt["A1"]}
    assert build_claim_table(_report([edge]), **_inputs([edge], alt_roots=both))["edges"][0][
        "D5_alternative_roots"]["status"] == "incomplete"
    with pytest.raises(Exception):
        build_claim_table(_report([edge]), **_inputs([edge], alt_roots={"A3": alt["A2"]}))


def test_d5_reads_nearby_summaries_and_keys_depth2_edges_by_path():
    graph = enumerate_model_graph(baseline_model_spec(), max_depth=2, max_models=12)
    keys = graph_edge_keys(graph)
    paths = mutation_paths(graph)
    deep = next(e for e in graph.edges if e.depth == 2)
    key = keys[(deep.parent_hash, deep.child_hash, deep.mutation_id)]
    assert key == "+".join(paths[deep.parent_hash]) + "|" + deep.mutation_id
    edge = _edge(deep.child_hash, 9.0, mutation=deep.mutation_id, parent=deep.parent_hash)
    report = _report([edge])
    report["graph_root_hash"] = graph.root_hash
    report["model_prior_variants"] = [
        {"penalty_per_axis": 0.0, "posterior_model_probabilities": {deep.parent_hash: 0.1, deep.child_hash: 0.9}}
    ]
    summary = {"scenarios": [{"graph_root_hash": "x", "root_edge_log_bayes_factors": {key: 4.0},
                              "inapplicable_paths": []}]}
    inputs = _inputs([edge], alt_roots={"A1": summary, "A2": {"root_edge_log_bayes_factors": {key: 2.0}}},
                     taper_mass=None)
    row = build_claim_table(report, graph=graph, **inputs)["edges"][0]
    assert row["edge_key"] == key
    assert row["D5_alternative_roots"]["status"] == "pass"
    assert "taper_mass_not_reported" in row["flags"]
    # without the graph a depth-2 edge cannot be keyed on the alternative roots
    row = build_claim_table(report, **inputs)["edges"][0]
    assert row["edge_key"] is None and row["D5_alternative_roots"]["status"] == "incomplete"
    with pytest.raises(Exception, match="root_edge_log_bayes_factors"):
        alt_root_edge_values({"graph_root_hash": "x"})
