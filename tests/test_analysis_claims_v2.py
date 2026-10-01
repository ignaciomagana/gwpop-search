"""The v2 claim table (criteria D1-D6) on mock analysis outputs."""

import copy
import json
import math

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


def _below(fraction=0.6, error=0.0, *, cut=0.9, threshold=1.0, kind="sharp", n_eff=5000.0):
    """A mass_below_entry block: P(sigma^2 <= cut) = fraction (+- error)."""
    return {"threshold": threshold, "kind": kind, "kish_ess": n_eff, "near_cut_band_mass": 0.3,
            "cuts": {repr(float(cut)): {"cut": cut, "fraction": fraction, "error": error, "n_eff": n_eff},
                     repr(float(threshold)): {"cut": threshold, "fraction": 1.0, "error": 0.0, "n_eff": n_eff}}}


def _inputs(edges, **over):
    hashes = {ROOT} | {e["child_hash"] for e in edges} | {e["parent_hash"] for e in edges}
    base = dict(
        sddr=_sddr(edges),
        taper2=_taper2(edges),
        alt_roots={"A1": _alt(edges), "A2": _alt(edges)},
        ppc={e["child_hash"]: _ppc(e["child_hash"]) for e in edges},
        taper_mass={h: 0.01 for h in [ROOT] + [e["child_hash"] for e in edges]},
        mass_below={h: _below() for h in hashes},
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


def test_d1_requires_the_taper_mass_of_both_endpoints():
    edges = [_edge("a" * 64, 9.0)]
    table = build_claim_table(_report(edges), **_inputs(edges, taper_mass={"a" * 64: 0.01}))
    row = table["edges"][0]
    assert row["D1_numerics"]["status"] == "incomplete"
    assert row["D1_numerics"]["taper_mass_reported"] is False
    assert row["label"] == INCONCLUSIVE and "taper_mass_not_reported" in row["flags"]
    # nor is DISFAVOURED available without it
    neg = [_edge("c" * 64, -9.0)]
    table = build_claim_table(_report(neg), **_inputs(neg, taper_mass=None))
    assert table["edges"][0]["label"] == INCONCLUSIVE


def test_near_cut_band_mass_is_reported_not_gating():
    """The DRAFT 0.10 band-mass limit is withdrawn: a large band mass no longer blocks D2."""
    import gwpop_search.analysis.claims_v2 as claims

    assert not hasattr(claims, "TAPER_MASS_D2_LIMIT")
    edges = [_edge("a" * 64, 9.0), _edge("c" * 64, -9.0)]
    masses = {ROOT: 0.67, "a" * 64: 0.67, "c" * 64: 0.67}
    table = build_claim_table(_report(edges), **_inputs(edges, taper_mass=masses))
    row = _row(table, "a" * 64)
    assert row["D2_strength"]["status"] == "pass" and row["label"] == SUPPORTED
    assert row["D2_strength"]["near_cut_band_mass_reported_not_gating"] == {"parent": 0.67, "child": 0.67}
    assert _row(table, "c" * 64)["label"] == DISFAVOURED
    assert table["constants"]["d2_primary_cut"] == 1.0 and table["constants"]["d2_tighter_cuts"] == [0.9]
    assert "taper_mass_d2_limit" not in table["constants"]


def test_d2_at_the_tighter_cut_uses_the_exact_sharp_cut_identity():
    from gwpop_search.analysis.claims_v2 import d2_at_cut

    a = "a" * 64
    # parent keeps 80% of its posterior below 0.9, child only 20%: ln BF(0.9) = 9 + ln(0.2/0.8)
    edges = [_edge(a, 9.0, sigma=0.5, bias=0.1)]
    below = {ROOT: _below(0.8), a: _below(0.2)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=below))["edges"][0]
    (cut,) = row["D2_strength"]["tighter_cuts"]
    assert cut["cut"] == 0.9
    assert cut["log_bayes_factor"] == pytest.approx(9.0 + math.log(0.2 / 0.8))
    assert cut["lower"] == pytest.approx(9.0 + math.log(0.25) - 1.0 - 0.1)
    assert cut["status"] == "pass" and row["D2_strength"]["status"] == "pass"
    assert row["D2_strength"]["min_lower"] == pytest.approx(cut["lower"])
    # the rule needs a tighter cut below the evaluations' sharp threshold
    assert d2_at_cut(9.0, 0.5, 0.1, 1.0, parent=below[ROOT], child=below[a])["status"] == "incomplete"
    smooth = {ROOT: _below(0.8, kind="smooth"), a: _below(0.2)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=smooth))["edges"][0]
    assert row["D2_strength"]["status"] == "incomplete" and row["label"] == INCONCLUSIVE
    other = {ROOT: _below(0.8, threshold=4.0), a: _below(0.2, threshold=4.0)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=other))["edges"][0]
    assert row["D2_strength"]["status"] == "incomplete"
    # a zero fraction cannot be resolved by the samples
    zero = {ROOT: _below(0.8), a: _below(0.0)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=zero))["edges"][0]
    assert row["D2_strength"]["tighter_cuts"][0]["status"] == "incomplete"
    assert row["D2_strength"]["status"] == "incomplete"


def test_d2_passing_at_the_primary_cut_but_failing_at_the_tighter_cut_fails():
    a = "a" * 64
    # primary: lower = 5 - 1 - 0.1 = 3.9 >= 3; at 0.9: + ln(0.3/0.9) = -1.10 -> lower 2.80 < 3
    edges = [_edge(a, 5.0, sigma=0.5, bias=0.1)]
    below = {ROOT: _below(0.9), a: _below(0.3)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=below))["edges"][0]
    d2 = row["D2_strength"]
    assert d2["primary_status"] == "pass" and d2["lower"] == pytest.approx(3.9)
    assert d2["tighter_cuts"][0]["status"] == "fail"
    assert d2["tighter_cuts"][0]["lower"] == pytest.approx(3.9 + math.log(1.0 / 3.0))
    assert d2["status"] == "fail" and "0.9" in d2["reason"]
    assert row["label"] == INCONCLUSIVE
    # both cuts pass -> pass (the child keeps as much mass below the cut as the parent)
    below = {ROOT: _below(0.5), a: _below(0.5)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=below))["edges"][0]
    assert row["D2_strength"]["status"] == "pass" and row["label"] == SUPPORTED
    # the tighter cut can also rescue nothing: a primary failure is a failure
    weak = [_edge(a, 3.5, sigma=0.5, bias=0.1)]
    row = build_claim_table(_report(weak), **_inputs(weak, mass_below={ROOT: _below(0.2), a: _below(0.9)}))["edges"][0]
    assert row["D2_strength"]["tighter_cuts"][0]["status"] == "pass"
    assert row["D2_strength"]["status"] == "fail"


def test_d2_is_incomplete_without_the_fractions_below_the_cut():
    a = "a" * 64
    edges = [_edge(a, 9.0), _edge("c" * 64, -9.0)]
    table = build_claim_table(_report(edges), **_inputs(edges, mass_below={ROOT: _below()}))
    row = _row(table, a)
    assert row["D2_strength"]["status"] == "incomplete"
    assert row["D2_strength"]["tighter_cuts"][0]["status"] == "missing"
    assert row["label"] == INCONCLUSIVE and "posterior_mass_below_cut_not_reported" in row["flags"]
    neg = _row(table, "c" * 64)
    assert neg["D2_strength"]["disfavoured"] is False
    assert neg["D2_strength"]["disfavoured_status"] == "incomplete"
    assert neg["label"] == INCONCLUSIVE


def test_fraction_uncertainty_is_added_in_quadrature():
    from gwpop_search.analysis.claims_v2 import d2_at_cut

    p_p, e_p, p_c, e_c = 0.5, 0.02, 0.4, 0.03
    out = d2_at_cut(6.0, 0.5, 0.1, 0.9, parent=_below(p_p, e_p), child=_below(p_c, e_c))
    sigma_fraction = math.hypot(e_c / p_c, e_p / p_p)
    assert out["sigma_fraction"] == pytest.approx(sigma_fraction)
    assert out["sigma_total"] == pytest.approx(math.hypot(0.5, sigma_fraction))
    lnbf = 6.0 + math.log(p_c / p_p)
    assert out["lower"] == pytest.approx(lnbf - 2.0 * math.hypot(0.5, sigma_fraction) - 0.1)
    assert out["upper"] == pytest.approx(lnbf + 2.0 * math.hypot(0.5, sigma_fraction) + 0.1)
    # the fraction error alone can decide: lower(c') just above 3 with exact fractions ...
    a = "a" * 64
    edges = [_edge(a, 4.2, sigma=0.5, bias=0.1)]
    exact = {ROOT: _below(0.5, 0.0), a: _below(0.5, 0.0)}
    assert build_claim_table(_report(edges), **_inputs(edges, mass_below=exact))["edges"][0][
        "D2_strength"]["status"] == "pass"
    # ... and below 3 once their binomial errors (ln-space 0.2 each) are included
    noisy = {ROOT: _below(0.5, 0.1), a: _below(0.5, 0.1)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=noisy))["edges"][0]
    assert row["D2_strength"]["tighter_cuts"][0]["status"] == "fail"
    assert row["D2_strength"]["status"] == "fail"


def test_disfavoured_needs_the_symmetric_condition_at_both_cuts():
    c = "c" * 64
    # primary: upper = -5 + 1 + 0.1 = -3.9 <= -3
    edges = [_edge(c, -5.0, sigma=0.5, bias=0.1)]
    inputs = dict(ppc={}, taper2=[], alt_roots={})
    same = {ROOT: _below(0.5), c: _below(0.5)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=same, **inputs))["edges"][0]
    assert row["D2_strength"]["disfavoured"] is True and row["label"] == DISFAVOURED
    # the child keeps more mass below 0.9 than the parent: ln BF(0.9) = -5 + ln 3 -> upper -2.80
    lifted = {ROOT: _below(0.3), c: _below(0.9)}
    row = build_claim_table(_report(edges), **_inputs(edges, mass_below=lifted, **inputs))["edges"][0]
    cut = row["D2_strength"]["tighter_cuts"][0]
    assert cut["upper"] == pytest.approx(-3.9 + math.log(3.0)) and cut["disfavoured"] is False
    assert row["D2_strength"]["disfavoured"] is False and row["D2_strength"]["disfavoured_status"] == "fail"
    assert row["label"] == INCONCLUSIVE
    # and the other way round: a primary-cut upper bound above -3 is never DISFAVOURED
    weak = [_edge(c, -3.5, sigma=0.5, bias=0.1)]
    row = build_claim_table(_report(weak), **_inputs(weak, mass_below={ROOT: _below(0.9), c: _below(0.1)},
                                                    **inputs))["edges"][0]
    assert row["D2_strength"]["tighter_cuts"][0]["disfavoured"] is True
    assert row["label"] == INCONCLUSIVE


def test_v2_numerics_and_claims_declare_the_same_tighter_cut():
    from gwpop_search.analysis.claims_v2 import D2_PRIMARY_CUT, D2_TIGHTER_CUTS
    from gwpop_search.inference.v2_numerics import V2_POSTERIOR_MASS_BELOW_CUTS, V2_TAPER_THRESHOLD

    assert tuple(D2_TIGHTER_CUTS) == tuple(V2_POSTERIOR_MASS_BELOW_CUTS) == (0.9,)
    assert D2_PRIMARY_CUT == V2_TAPER_THRESHOLD == 1.0


def test_collect_reads_recorded_fractions_and_the_claim_cli_merges_them(tmp_path):
    from gwpop_search.analysis.claims_v2 import collect_v2_evaluations, mass_below_entry
    from gwpop_search.analysis.posthoc_cut import POSTHOC_MASS_BELOW_FORMAT, posthoc_report, read_mass_below

    taper = {"kind": "sharp", "threshold": 1.0}
    pooled = {"taper": taper, "kish_ess": 900.0, "posterior_mass_in_taper_region": 0.4,
              "posterior_mass_below": {"0.9": {"cut": 0.9, "fraction": 0.55, "error": 0.016, "n_eff": 900.0},
                                       "1.0": {"cut": 1.0, "fraction": 1.0, "error": 0.0, "n_eff": 900.0}}}
    for name, block in (("new", pooled), ("old", {k: v for k, v in pooled.items() if k != "posterior_mass_below"})):
        d = tmp_path / name
        d.mkdir()
        (d / "evaluation.json").write_text(json.dumps({
            "model_hash": name * 2, "fidelity": "F3",
            "diagnostics": {"passed": True, "taper": {"pooled": block}}}))
    collected = collect_v2_evaluations([tmp_path])
    assert set(collected["taper_mass"]) == {"newnew", "oldold"}
    assert set(collected["mass_below"]) == {"newnew"}
    entry = collected["mass_below"]["newnew"]
    assert entry["threshold"] == 1.0 and entry["kind"] == "sharp"
    assert entry["cuts"]["0.9"]["fraction"] == 0.55 and entry["near_cut_band_mass"] == 0.4
    assert mass_below_entry({"taper": taper}) is None
    report = posthoc_report([{"model_hash": "oldold", "entry": entry}], cuts=(0.9,))
    assert report["format_version"] == POSTHOC_MASS_BELOW_FORMAT
    assert read_mass_below(json.loads(json.dumps(report))) == {"oldold": entry}
    with pytest.raises(Exception, match="format"):
        read_mass_below({"models": {}})


def test_trials_count_distinct_models_not_edges():
    """A depth-2 model reached from both depth-1 parents is one hypothesis tried."""
    a, b, ab = "a" * 64, "b" * 64, "d" * 64
    edges = [_edge(a, 9.0), _edge(b, 0.4), _edge(ab, 1.0, parent=a, mutation="mut.x"),
             _edge(ab, 0.8, parent=b, mutation="mut.y")]
    table = build_claim_table(_report(edges), **_inputs(edges))
    assert table["n_atoms_tried"] == 3
    assert table["n_edges_evaluated"] == 4
    assert "3** (4 edges evaluated)" in render_claims_markdown(table)


def test_d6_needs_enough_draws_for_the_two_sided_tail():
    edges = [_edge("a" * 64, 9.0)]
    few = {"a" * 64: _ppc("a" * 64, config={"alpha": 0.01, "n_draws": 500})}
    row = build_claim_table(_report(edges), **_inputs(edges, ppc=few))["edges"][0]
    assert row["D6_ppc"]["status"] == "incomplete"
    assert row["label"] == INCONCLUSIVE
