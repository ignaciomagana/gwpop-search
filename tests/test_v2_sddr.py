"""Savage-Dickey null embeddings of the v2 edges."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from gwpop_search.analysis._common import posterior_from_equal_weight_draws  # noqa: E402
from gwpop_search.analysis.sddr import (  # noqa: E402
    UniformDifferencePrior,
    classify_edge,
    classify_graph_edges,
    prior_log_density,
    prior_support,
    sddr_edge_check,
    verify_nesting,
)
from gwpop_search.grammar import (  # noqa: E402
    V2_ATOM_IDS,
    V2_MUTATION_TABLE,
    apply_mutation,
    compose_atoms,
    enumerate_v2_depth1,
    v2_root_model_spec,
)
from gwpop_search.grammar.v2 import V2_MUTATION_ATOM  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz

#: atom -> (classification, larger model, tested coordinate)
EXPECTED = {
    "M1": ("exact", "parent", "lam_u_p35"),
    "M2": ("exact", "parent", "lam_u_p10"),
    "M3": ("approximate", "child", "lam_u_p3"),  # mu_p10 narrowed: shared prior differs
    "M4": ("exact", "child", "lam_u_p3"),
    "M5": ("exact", "parent", "alpha_2-alpha_1"),
    "C1": ("exact", "child", "chi_mu_q_slope"),
    "C2": ("exact", "child", "chi_log_sigma_q_slope"),
    "C3": ("exact", "child", "chi_mu_z_slope"),
    "C4": ("exact", "child", "chi_log_sigma_z_slope"),
    "C5": ("exact", "child", "chi_mu_log_m1_slope"),
    "C6": ("exact", "child", "chi_log_sigma_log_m1_slope"),
    "S1": ("exact", "child", "chi_fraction"),
    "S2": ("evidence_only", None, None),
    "S3": ("exact", "child", "chi_eps"),
    "S4": ("evidence_only", None, None),
    "P1": ("exact", "child", "beta_high-beta_low"),
    "P2": ("evidence_only", None, None),
    "Z1": ("exact", "child", "md_kappa"),
    "Z2": ("exact", "child", "kappa_log_m1_slope"),
}


def test_every_depth1_edge_has_the_expected_nesting_class():
    graph = enumerate_v2_depth1()
    rows = {V2_MUTATION_ATOM[r.mutation_id]: r for r in classify_graph_edges(graph)}
    assert set(rows) == set(EXPECTED)
    for aid, (cls, larger, tested) in EXPECTED.items():
        row = rows[aid]
        assert row.classification == cls, (aid, row.reason)
        assert row.larger_model == larger, aid
        if tested is not None:
            assert row.embeddings[0].tested_parameter == tested
        else:
            assert row.reason and not row.embeddings
    assert "partially overlap" in rows["M3"].reason
    assert "two-dimensional" in rows["P2"].reason and "two-dimensional" in rows["S2"].reason
    assert "nu -> infinity" in rows["S4"].reason


def test_exact_embeddings_reproduce_the_smaller_model_numerically():
    graph = enumerate_v2_depth1()
    by_hash = graph.by_hash
    checked = 0
    for edge, row in zip(graph.edges, classify_graph_edges(graph)):
        if row.classification != "exact":
            continue
        result = verify_nesting(row, by_hash[edge.parent_hash], by_hash[edge.child_hash],
                                n_checks=3, n_samples=400, seed=7)
        assert result["all_nested"], (V2_MUTATION_ATOM[edge.mutation_id], result)
        checked += 1
    assert checked == 15


def test_decided_conventions_keep_the_null_embeddings_exact():
    """C3/C4 (pivot z = 0.5) and Z2 (local mass function): slope 0 is the parent, to rounding."""
    graph = enumerate_v2_depth1()
    by_hash = graph.by_hash
    rows = {V2_MUTATION_ATOM[r.mutation_id]: (e, r) for e, r in zip(graph.edges, classify_graph_edges(graph))}
    for aid in ("C3", "C4", "Z2"):
        edge, row = rows[aid]
        assert row.classification == "exact" and row.larger_model == "child"
        assert row.embeddings[0].null_value == 0.0 and not row.vw_required
        result = verify_nesting(row, by_hash[edge.parent_hash], by_hash[edge.child_hash],
                                n_checks=4, n_samples=400, seed=11, rtol=1e-13)
        assert result["all_nested"], (aid, result)
        assert result["embeddings"][0]["support_pattern_mismatches"] == 0


def test_a2_suite_edges_stay_exactly_nested():
    """On the alternative root A2 (kappa(m1), local mass function) every atom's edge classifies
    as on R0, and the exact ones reproduce A2 at the null."""
    from gwpop_search.grammar import InapplicableMutation
    from gwpop_search.grammar.v2 import v2_alternative_roots

    a2 = v2_alternative_roots()["A2"]
    checked = 0
    for aid, (cls, larger, tested) in EXPECTED.items():
        mutation = V2_MUTATION_TABLE[V2_ATOM_IDS[aid]]
        try:
            child = apply_mutation(a2, mutation)
        except InapplicableMutation:
            assert aid in ("Z1", "Z2")
            continue
        row = classify_edge(a2, child, mutation.mutation_id)
        assert (row.classification, row.larger_model) == (cls, larger), (aid, row.reason)
        if cls != "exact" or aid not in ("M1", "M5", "C2", "C3", "C4", "S1", "S3", "P1"):
            continue
        assert row.embeddings[0].tested_parameter == tested
        result = verify_nesting(row, a2, child, n_checks=2, n_samples=300, seed=5)
        assert result["all_nested"], (aid, result)
        checked += 1
    assert checked == 8
    # the A2 root edge itself (R0 -> A2 is the Z2 edge)
    root = v2_root_model_spec()
    row = classify_edge(root, a2, V2_ATOM_IDS["Z2"])
    assert row.classification == "exact" and row.embeddings[0].tested_parameter == "kappa_log_m1_slope"
    assert verify_nesting(row, root, a2, n_checks=3, n_samples=300, seed=6, rtol=1e-13)["all_nested"]


def test_depth2_edges_are_classified_by_their_single_axis():
    root = v2_root_model_spec()
    c2 = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS["C2"]])
    c4 = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS["C4"]])
    s2 = apply_mutation(root, V2_MUTATION_TABLE[V2_ATOM_IDS["S2"]])
    both = compose_atoms(root, "C2", "C4")
    assert classify_edge(c2, both, V2_ATOM_IDS["C4"]).embeddings[0].tested_parameter == "chi_log_sigma_z_slope"
    assert classify_edge(c4, both, V2_ATOM_IDS["C2"]).classification == "exact"
    mixed = compose_atoms(root, "C2", "S2")
    row = classify_edge(s2, mixed, V2_ATOM_IDS["C2"])
    assert row.classification == "exact" and row.embeddings[0].tested_parameter == "chi_log_sigma_q_slope"
    assert verify_nesting(row, s2, mixed, n_checks=2, n_samples=300)["all_nested"]
    assert classify_edge(c2, mixed, V2_ATOM_IDS["S2"]).classification == "evidence_only"
    # a correlated Gaussian nests in the correlated constant-fraction mixture
    s1c2 = compose_atoms(root, "C2", "S1")
    row = classify_edge(c2, s1c2, V2_ATOM_IDS["S1"])
    assert row.classification == "exact"
    assert verify_nesting(row, c2, s1c2, n_checks=2, n_samples=300)["all_nested"]


def test_uniform_difference_prior():
    prior = UniformDifferencePrior(-4.0, 12.0, -4.0, 12.0)
    assert prior_support(prior) == (-16.0, 16.0)
    assert prior_log_density(prior, 0.0) == pytest.approx(-math.log(16.0))
    assert prior_log_density(prior, 8.0) == pytest.approx(math.log(8.0 / 256.0))
    assert prior_log_density(prior, 16.5) == -math.inf
    x = np.linspace(-16, 16, 20001)
    from gwpop_search.analysis.sddr import prior_log_density_array

    dens = np.exp(prior_log_density_array(prior, x))
    assert abs(_trapz(dens, x) - 1.0) < 1e-6


def _toy_posterior(names, draws, log_like, rng, n_keep=60000):
    w = np.exp(log_like - log_like.max())
    idx = rng.choice(len(w), size=n_keep, replace=True, p=w / w.sum())
    return posterior_from_equal_weight_draws(names, draws[idx], source="toy")


def test_difference_null_sddr_matches_the_analytic_bayes_factor():
    """M5 (no break): the root is the larger model, tested d = alpha_2 - alpha_1."""
    graph = enumerate_v2_depth1()
    edge = next(e for e in graph.edges if V2_MUTATION_ATOM[e.mutation_id] == "M5")
    parent, child = graph.by_hash[edge.parent_hash], graph.by_hash[edge.child_hash]
    nesting = classify_edge(parent, child, edge.mutation_id)
    rng = np.random.default_rng(1)
    n = 2_000_000
    draws = rng.uniform(-4.0, 12.0, size=(n, 2))  # (alpha_1, alpha_2) from the prior
    mu, s = 0.8, 0.6
    d = draws[:, 1] - draws[:, 0]
    sample = _toy_posterior(("alpha_1", "alpha_2"), draws, norm.logpdf(d, mu, s), rng)
    # analytic: Z_S / Z_L = L(d = 0) / int pi(d) L(d) dd, pi(d) = (16 - |d|) / 256
    grid = np.linspace(-16, 16, 400001)
    evidence_l = _trapz((16 - np.abs(grid)) / 256.0 * norm.pdf(grid, mu, s), grid)
    expect_child_over_parent = math.log(norm.pdf(0.0, mu, s) / evidence_l)
    check = sddr_edge_check(nesting, sample, parent.priors, child.priors, n_bootstrap=50,
                            log_bf_ns=expect_child_over_parent, sigma_ns=0.0)
    assert check.status == "agree", check.to_dict()
    assert abs(check.log_bf_child_over_parent_sddr - expect_child_over_parent) < 0.05


def test_dirichlet_boundary_null_sddr_matches_the_analytic_bayes_factor():
    """M1 (drop the 35 Msun peak): lam_u_p35 = 0 at the lower prior bound."""
    graph = enumerate_v2_depth1()
    edge = next(e for e in graph.edges if V2_MUTATION_ATOM[e.mutation_id] == "M1")
    parent, child = graph.by_hash[edge.parent_hash], graph.by_hash[edge.child_hash]
    nesting = classify_edge(parent, child, edge.mutation_id)
    rng = np.random.default_rng(2)
    u = rng.uniform(0.0, 1.0, size=(2_000_000, 1))
    scale = 0.15  # likelihood exp(-u / scale): the data prefer a small third weight
    sample = _toy_posterior(("lam_u_p35",), u, -u[:, 0] / scale, rng, n_keep=80000)
    expect = -math.log(scale * (1.0 - math.exp(-1.0 / scale)))  # ln Z_S / Z_L = ln L(0) / int L
    check = sddr_edge_check(nesting, sample, parent.priors, child.priors, n_bootstrap=50,
                            log_bf_ns=expect, sigma_ns=0.0)
    assert check.status in ("agree", "computed"), check.to_dict()
    assert abs(check.log_bf_child_over_parent_sddr - expect) < 0.05
