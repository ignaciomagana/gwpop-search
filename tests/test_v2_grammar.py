"""v2 grammar: root R0, the 19 atoms, depth-2 enumeration rule, alt roots, G12 checks."""

from __future__ import annotations

import json

import pytest

from gwpop_search.cli import build_parser
from gwpop_search.grammar import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MUTATIONS,
    HYPERPRIOR_PROFILES,
    V2_ATOM_IDS,
    V2_MUTATION_TABLE,
    V2_MUTATIONS,
    apply_mutation,
    baseline_hyperprior_profile,
    baseline_model_spec,
    compose_atoms,
    enumerate_model_graph,
    enumerate_v2_depth1,
    load_model_graph,
    mutations_for_profile,
    plan_depth2,
    save_model_graph,
    structural_diff_axes,
    v2_root_model_spec,
)
from gwpop_search.grammar.schema import PriorConfig
from gwpop_search.grammar.v2 import (
    CHIEFF_ATOMS,
    V2_DRAFT_PRIORS,
    V2_MUTATION_ATOM,
    NotComposable,
    depth2_priority,
    enumerate_alt_root_graph,
    extend_graph_with_depth2,
    v2_alternative_roots,
    v2_graph_payload,
    v2_model_graph_checks,
)


def _prior(spec, name):
    p = spec.priors[name]
    return p.family, p.parameters["low"], p.parameters["high"]


def test_root_is_the_lvk_default_with_gwtc5_table5_priors():
    root = v2_root_model_spec()
    assert (root.mass.family, root.pairing.family, root.chieff.family, root.redshift.family) == (
        "bp2p", "tapered_powerlaw_q", "linear_gaussian", "powerlaw_1pz")
    expect = {
        "alpha_1": (-4, 12), "alpha_2": (-4, 12), "m_break": (20, 50), "mu_p10": (5, 20),
        "sigma_p10": (0, 10), "mu_p35": (25, 60), "sigma_p35": (0, 10), "mlow_1": (3, 10),
        "delta_m_1": (0, 10), "delta_m_2": (0, 10), "beta": (-2, 7), "kappa": (-10, 10),
        "chi_mu": (-1, 1), "chi_log_sigma": (-5, 0), "mlow_2_frac": (0, 1),
        "lam_u_pl": (0, 1), "lam_u_p10": (0, 1), "lam_u_p35": (0, 1),
    }
    assert set(root.priors) == set(expect)
    for name, (lo, hi) in expect.items():
        assert _prior(root, name) == ("uniform", lo, hi), name
    s = root.support
    assert (s["zmax"], s["mmin"], s["mmax"], s["q_floor"]) == (1.9, 3.0, 300.0, 0.05)
    assert (s["m1_grid"], s["n_m1"], s["n_q"]) == ("geomspace", 1000, 500)
    assert root.to_dict()["support"] == s


def test_v2_profile_is_registered_and_v1_hashes_are_unchanged():
    assert "gwtc5-v2" in HYPERPRIOR_PROFILES
    assert baseline_model_spec("gwtc5-v2") == v2_root_model_spec()
    assert baseline_hyperprior_profile(v2_root_model_spec()) == "gwtc5-v2"
    assert mutations_for_profile("gwtc5-v2") == V2_MUTATIONS
    assert mutations_for_profile("gwtc5-v1") == DEFAULT_MUTATIONS
    # pinned v1 roots
    assert baseline_model_spec("phase3").model_hash == (
        "6d0c93c5d606f64a0d86ca4979d570f55e7dac985322c74fd3de0da3569808d4")
    assert baseline_model_spec("gwtc5-v1").model_hash == (
        "666c4ffc99f000e8c3783589866f0ab6bc120bf161fca5f47b5130983263ff67")


def test_depth1_graph_has_exactly_20_nodes_and_19_single_axis_edges():
    root = v2_root_model_spec()
    graph = enumerate_v2_depth1(root)
    assert len(graph.nodes) == 20 and len(graph.edges) == 19
    assert len({n.model_hash for n in graph.nodes}) == 20
    assert sorted(V2_MUTATION_ATOM[e.mutation_id] for e in graph.edges) == sorted(V2_ATOM_IDS)
    by_hash = graph.by_hash
    for edge in graph.edges:
        assert edge.parent_hash == root.model_hash and edge.depth == 1
        child = by_hash[edge.child_hash]
        mutation = V2_MUTATION_TABLE[edge.mutation_id]
        assert structural_diff_axes(root, child) == (mutation.axis,)
        assert child.support == root.support
        DEFAULT_COMPONENT_REGISTRY.validate_model(child)
    # the depth limit is respected: no atom re-applies to its own child at depth 1
    assert all(graph.depths[n.model_hash] <= 1 for n in graph.nodes)


def test_atom_priors_follow_the_spec():
    root = v2_root_model_spec()
    child = {aid: apply_mutation(root, V2_MUTATION_TABLE[mid]) for aid, mid in V2_ATOM_IDS.items()}
    assert set(root.priors) - set(child["M1"].priors) == {"mu_p35", "sigma_p35", "lam_u_p35"}
    assert set(root.priors) - set(child["M2"].priors) == {"mu_p10", "sigma_p10", "lam_u_p10"}
    assert _prior(child["M3"], "mu_p3") == ("uniform", 14, 24)
    assert _prior(child["M3"], "mu_p10") == ("uniform", 5, 14)
    assert _prior(child["M4"], "mu_p3") == ("uniform", 60, 90)
    assert _prior(child["M4"], "mu_p10") == ("uniform", 5, 20)
    assert set(child["M5"].priors) - set(root.priors) == {"alpha"}
    assert _prior(child["C1"], "chi_mu_q_slope") == ("uniform", -2, 2)
    assert _prior(child["C2"], "chi_log_sigma_q_slope") == ("uniform", -12, 4)
    assert _prior(child["C3"], "chi_mu_z_slope") == ("uniform", -1, 1)
    assert _prior(child["C4"], "chi_log_sigma_z_slope") == ("uniform", -3, 5)
    assert _prior(child["S3"], "chi_eps") == ("uniform", -1, 1)
    assert _prior(child["S4"], "chi_nu") == ("log_uniform", 1, 100)
    assert _prior(child["P1"], "beta_low") == _prior(child["P1"], "beta_high") == ("uniform", -2, 7)
    for c in ("pl", "p10", "p35"):
        assert _prior(child["P2"], f"beta_{c}") == ("uniform", -10, 13)
    assert child["C2"].chieff.options["q_pivot"] == 1.0 and child["C4"].chieff.options["z_pivot"] == 0.0
    # every draft choice is disclosed
    for key in ("C5", "C6", "S1", "S2", "P1", "Z1", "Z2"):
        assert any(k.startswith(key) for k in V2_DRAFT_PRIORS), key


def test_mass_atoms_require_the_v2_root_family():
    v1 = baseline_model_spec("gwtc5-v1")
    for mutation in V2_MUTATIONS:
        with pytest.raises(Exception):
            apply_mutation(v1, mutation)
    for mid in ("v2.mass.drop_p35", "v2.mass.no_break"):
        assert V2_MUTATION_TABLE[mid].requires_family == "bp2p"


def test_enumerate_models_cli_v2_profile(tmp_path, capsys):
    out = tmp_path / "graph.json"
    args = build_parser().parse_args(["enumerate-models", "--output", str(out), "--max-depth", "1",
                                      "--max-models", "100", "--hyperprior-profile", "gwtc5-v2"])
    args.func(args)
    payload = json.loads(out.read_text())
    assert payload["root_hash"] == v2_root_model_spec().model_hash
    assert len(payload["nodes"]) == 20 and len(payload["edges"]) == 19


# ---------------------------------------------------------------------------
# depth 2
# ---------------------------------------------------------------------------


def test_depth2_priority_order_and_cap():
    ranked = depth2_priority(V2_ATOM_IDS)
    assert [p for _, p in ranked[:3]] == [("C2", "S2"), ("C2", "C4"), ("C2", "C6")]
    tiers = [t for t, _ in ranked]
    assert tiers == sorted(tiers)
    assert len(ranked) == 19 * 18 // 2
    assert ranked[3] == (1, ("M1", "C1"))
    plan = plan_depth2(V2_ATOM_IDS)
    assert len(plan.selected) == 12
    assert [s.atoms for s in plan.selected[:3]] == [("C2", "S2"), ("C2", "C4"), ("C2", "C6")]
    assert all(s.tier == 1 for s in plan.selected[3:])
    assert len(plan.selected) + len(plan.over_cap) + len(plan.not_composable) == len(ranked)


def test_depth2_only_uses_pairs_of_passing_edges():
    assert plan_depth2([]).selected == ()
    assert plan_depth2(["C2"]).selected == ()
    plan = plan_depth2(["C4", "C2", "S2"])
    assert [s.atoms for s in plan.selected] == [("C2", "S2"), ("C2", "C4"), ("C4", "S2")]
    plan = plan_depth2(["M4", "C6", "Z1"])
    assert [(s.tier, s.atoms) for s in plan.selected] == [(1, ("M4", "C6")), (3, ("M4", "Z1")), (3, ("C6", "Z1"))]
    assert plan_depth2(V2_ATOM_IDS, cap=0).selected == ()
    with pytest.raises(ValueError):
        plan_depth2(["X9"])


@pytest.mark.parametrize("pair", [("S1", "S2"), ("S3", "S4"), ("M1", "M3"), ("P1", "P2"), ("Z1", "Z2"),
                                  ("M5", "M2")])
def test_incompatible_pairs_are_not_composable(pair):
    root = v2_root_model_spec()
    with pytest.raises(NotComposable):
        compose_atoms(root, *pair)
    plan = plan_depth2(pair)
    assert plan.selected == () and len(plan.not_composable) == 1


def test_width_q_plus_mass_dependent_fraction_keeps_both_changes():
    root = v2_root_model_spec()
    model = compose_atoms(root, "C2", "S2")
    assert model == compose_atoms(root, "S2", "C2")
    assert model.chieff.family == "linear_gaussian_mixture"
    assert model.chieff.options["log_sigma_q"] == "linear"
    assert model.chieff.options["fraction_dependence"] == "logistic_log_m1"
    assert "chi_log_sigma_q_slope" in model.priors and "chi_fraction_m_t" in model.priors
    # both single-atom parents reach it with the other atom
    c2 = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS["C2"]])
    s2 = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS["S2"]])
    assert apply_mutation(c2, V2_MUTATION_TABLE[V2_ATOM_IDS["S2"]]) == model
    assert apply_mutation(s2, V2_MUTATION_TABLE[V2_ATOM_IDS["C2"]]) == model


def test_depth2_graph_extension_roundtrips(tmp_path):
    graph = enumerate_v2_depth1()
    plan = plan_depth2(V2_ATOM_IDS)
    full = extend_graph_with_depth2(graph, plan)
    assert len(full.nodes) == 32  # the spec cap: about 32 models including depth 2
    depth2 = [e for e in full.edges if e.depth == 2]
    assert len(depth2) == 2 * 12  # every depth-2 node is reached from both of its atoms' children
    path = tmp_path / "g.json"
    save_model_graph(path, full)
    again = load_model_graph(path)
    assert again.by_hash.keys() == full.by_hash.keys()
    assert len(again.edges) == 19 + 24


# ---------------------------------------------------------------------------
# alternative roots, G12 checks, payload
# ---------------------------------------------------------------------------


def test_alternative_roots_run_every_chi_eff_atom():
    root = v2_root_model_spec()
    alts = v2_alternative_roots(root)
    assert alts["A1"].pairing.options["beta_dependence"] == "per_mass_component"
    assert alts["A2"].redshift.options["kappa_dependence"] == "linear_log_m1"
    for alt in alts.values():
        assert len(structural_diff_axes(root, alt)) == 1
        graph = enumerate_alt_root_graph(alt, CHIEFF_ATOMS)
        assert len(graph.nodes) == 11 and len(graph.edges) == 10


def test_g12_model_graph_checks():
    graph = enumerate_v2_depth1()
    ok = v2_model_graph_checks(graph, export_zmax={"pe": 1.9, "selection": 1.9})
    assert ok["pass"] and ok["n_models"] == 20
    assert all(row["mlow_1_prior_low"] >= 3.0 and row["m1_ceiling"] <= 590.0 for row in ok["models"])
    assert not v2_model_graph_checks(graph, export_zmax={"selection": 2.5})["pass"]
    v1 = enumerate_model_graph(baseline_model_spec("gwtc5-v1"), max_depth=1, max_models=100)
    assert not v2_model_graph_checks(v1)["pass"]  # mmin U(2, 10), no support block


def test_graph_payload_is_loadable_and_marked_draft(tmp_path):
    graph = enumerate_v2_depth1()
    payload = v2_graph_payload(graph, depth2=plan_depth2([]))
    assert payload["metadata"]["status"] == "DRAFT"
    assert set(payload["metadata"]["atoms"]) == set(V2_ATOM_IDS)
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload))
    loaded = load_model_graph(path)
    assert len(loaded.nodes) == 20 and len(loaded.edges) == 19


@pytest.mark.parametrize("mass_atom, components", [
    ("M1", ("pl", "p10")), ("M2", ("pl", "p35")), ("M3", ("pl", "p10", "p35", "p3")),
    ("M4", ("pl", "p10", "p35", "p3")), ("M5", ("pl", "p10", "p35")),
])
def test_mass_x_per_component_pairing_composes_in_both_orders(mass_atom, components):
    """A mass atom and P2 commute: one pairing slope per component of the final mass family."""
    root = v2_root_model_spec()
    model = compose_atoms(root, mass_atom, "P2")
    assert model == compose_atoms(root, "P2", mass_atom)
    m = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS[mass_atom]])
    p = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS["P2"]])
    assert apply_mutation(m, V2_MUTATION_TABLE[V2_ATOM_IDS["P2"]]) == model
    assert apply_mutation(p, V2_MUTATION_TABLE[V2_ATOM_IDS[mass_atom]]) == model
    betas = sorted(k for k in model.priors if k.startswith("beta_"))
    assert betas == sorted(f"beta_{c}" for c in components)
    assert all(model.priors[b] == PriorConfig("uniform", {"low": -10.0, "high": 13.0}) for b in betas)


#: expected D5 status of every atom on each alternative root
_D5_NOT_APPLICABLE = {"A1": {"P1", "P2"}, "A2": {"Z1", "Z2"}}


@pytest.mark.parametrize("alt", ["A1", "A2"])
def test_every_atom_applies_or_is_declared_inapplicable_on_each_alternative_root(alt):
    """Every atom on A1/A2 yields a valid child or an InapplicableMutation (never a ValueError),
    its edge classifies, and a D5 suite over all atoms can be planned."""
    from gwpop_search.analysis.sddr import classify_edge
    from gwpop_search.grammar import InapplicableMutation
    from gwpop_search.grammar.v2 import v2_d5_atom_semantics
    from gwpop_search.validation.baselines import restricted_model_graph

    root = v2_alternative_roots()[alt]
    applicable = set()
    for atom in V2_ATOM_IDS:
        mutation = V2_MUTATION_TABLE[V2_ATOM_IDS[atom]]
        try:
            child = apply_mutation(root, mutation)
        except InapplicableMutation:
            continue
        applicable.add(atom)
        nesting = classify_edge(root, child, mutation.mutation_id)
        assert nesting.classification in ("exact", "approximate", "evidence_only")
    assert set(V2_ATOM_IDS) - applicable == _D5_NOT_APPLICABLE[alt]
    semantics = v2_d5_atom_semantics()[alt]
    assert {a for a, row in semantics.items() if row["status"] == "not_applicable"} == _D5_NOT_APPLICABLE[alt]
    graph, inapplicable = restricted_model_graph(
        root, [(mid,) for mid in V2_ATOM_IDS.values()], V2_MUTATION_TABLE,
    )
    assert len(graph.nodes) == 1 + len(applicable)
    assert sorted(inapplicable) == sorted(V2_ATOM_IDS[a] for a in _D5_NOT_APPLICABLE[alt])


def test_a1_mass_atoms_carry_the_component_pairing_slope():
    a1 = v2_alternative_roots()["A1"]
    m1 = apply_mutation(a1, V2_MUTATION_TABLE[V2_ATOM_IDS["M1"]])
    assert "beta_p35" not in m1.priors and {"beta_pl", "beta_p10"} <= set(m1.priors)
    m3 = apply_mutation(a1, V2_MUTATION_TABLE[V2_ATOM_IDS["M3"]])
    assert m3.priors["beta_p3"] == PriorConfig("uniform", {"low": -10.0, "high": 13.0})
    # on R0 (constant beta) the same atoms add no pairing slope
    r0 = v2_root_model_spec()
    assert not any(k.startswith("beta_p") for k in apply_mutation(r0, V2_MUTATION_TABLE[V2_ATOM_IDS["M3"]]).priors)


def test_z1_on_a2_and_p1_on_a1_are_inapplicable_not_different_hypotheses():
    from gwpop_search.grammar import InapplicableMutation

    alts = v2_alternative_roots()
    with pytest.raises(InapplicableMutation, match="kappa_dependence"):
        apply_mutation(alts["A2"], V2_MUTATION_TABLE[V2_ATOM_IDS["Z1"]])
    with pytest.raises(InapplicableMutation, match="beta_dependence"):
        apply_mutation(alts["A1"], V2_MUTATION_TABLE[V2_ATOM_IDS["P1"]])
