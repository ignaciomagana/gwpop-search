"""v2 D5: alternative-root scenarios restricted to root + candidates + all 9 chi_eff atoms."""

import json

import pytest

from gwpop_search.cli import build_parser
from gwpop_search.grammar import DEFAULT_MUTATIONS, apply_mutation, baseline_model_spec
from gwpop_search.grammar.io import save_model_spec
from gwpop_search.search import ModelEvidence
from gwpop_search.validation import (
    NearbyBaselineConfig,
    NearbyBaselineScenario,
    NearbyBaselineSuiteSpec,
    V2_CHI_EFF_ATOMS,
    load_nearby_baseline_suite_spec,
    nearby_scenario_graph,
    nearby_scenario_inapplicable_paths,
    nearby_scenario_models,
    register_mutation_catalogue,
    restricted_model_graph,
    root_edge_log_bayes_factors,
    save_nearby_baseline_suite_spec,
    v2_alt_root_scenario,
)
from gwpop_search.validation.baselines import mutation_catalogue

#: stand-ins on the v1 grammar for the nine v2 chi_eff atoms (S4 dropped 2026-10-02) (the v2 grammar
#: registers its own catalogue); S2 is a two-step path, as the follow-up
#: mixture-fraction model is built
CHI_EFF = {
    "C1": "chieff.mean.linear_q",
    "C2": "chieff.width.linear_q",
    "C3": "chieff.mean.linear_z",
    "C4": "chieff.width.linear_z",
    "C5": "chieff.mean.linear_m1",
    "C6": "chieff.width.linear_m1",
    "S1": "chieff.family.gaussian_mixture",
    "S2": ("chieff.family.gaussian_mixture", "chieff.fraction.logistic_q"),
    "S3": "chieff.width.logistic_q",
}


def _mut(mutation_id):
    return next(item for item in DEFAULT_MUTATIONS if item.mutation_id == mutation_id)


def _alt_root():
    # stand-in for "BP2P + beta per component": a root that already carries a pairing change
    return apply_mutation(baseline_model_spec(), _mut("pairing.beta.logistic_m1"))


def _scenario(**over):
    kwargs = dict(
        candidate_paths=[
            "pairing.beta.logistic_m1",                          # inapplicable on this root
            "mass.family.broken_powerlaw",                       # a mass candidate
            ("chieff.width.linear_q", "chieff.mean.linear_z"),   # a depth-2 candidate
        ],
        chi_eff_mutation_ids=CHI_EFF,
        mutation_catalogue_name="default+followup",
    )
    kwargs.update(over)
    return v2_alt_root_scenario("A1_beta_per_component", _alt_root(), **kwargs)


def test_alt_root_scenario_runs_exactly_root_candidates_and_chi_eff_atoms():
    scenario = _scenario()
    assert scenario.restricted and scenario.max_depth == 2
    graph = nearby_scenario_graph(scenario)
    table = mutation_catalogue("default+followup")
    root = _alt_root()
    expected = {root.model_hash}
    for atom, path in CHI_EFF.items():
        model = root
        for mutation_id in ((path,) if isinstance(path, str) else path):
            model = apply_mutation(model, table[mutation_id])
            expected.add(model.model_hash)
    for path in (("mass.family.broken_powerlaw",), ("chieff.width.linear_q", "chieff.mean.linear_z")):
        model = root
        for mutation_id in path:
            model = apply_mutation(model, table[mutation_id])
            expected.add(model.model_hash)
    assert {m.model_hash for m in graph.nodes} == expected
    # 1 root + 9 chi_eff nodes (S1 is S2's prefix) + mass + depth-2 = 12
    assert len(graph.nodes) == 12
    assert len(graph.nodes) <= scenario.max_models
    assert nearby_scenario_inapplicable_paths(scenario) == ("pairing.beta.logistic_m1",)
    listing = nearby_scenario_models(scenario, NearbyBaselineConfig(max_f3_models=20))
    assert listing["n_graph_models"] == 12 and listing["f3_budget_fits_all_models"] is True
    # an unrestricted scenario on the same root enumerates the whole neighbourhood instead
    full = nearby_scenario_graph(NearbyBaselineScenario("full", root, max_depth=1, max_models=40))
    assert len(full.nodes) > len([n for n in graph.nodes if graph.depths[n.model_hash] <= 1])


def test_all_nine_chi_eff_atoms_are_required():
    assert V2_CHI_EFF_ATOMS == ("C1", "C2", "C3", "C4", "C5", "C6", "S1", "S2", "S3")
    with pytest.raises(ValueError, match="chi_eff atoms"):
        _scenario(chi_eff_mutation_ids=dict(CHI_EFF, S4="chieff.family.student_t"))
    partial = {k: v for k, v in CHI_EFF.items() if k != "S3"}
    with pytest.raises(ValueError, match="chi_eff atoms"):
        _scenario(chi_eff_mutation_ids=partial)
    dup = dict(CHI_EFF, S3=CHI_EFF["C1"])
    with pytest.raises(ValueError, match="distinct"):
        _scenario(chi_eff_mutation_ids=dup)
    with pytest.raises(ValueError, match="unknown mutation"):
        nearby_scenario_graph(_scenario(candidate_paths=["no.such.mutation"]))
    with pytest.raises(ValueError, match="unknown mutation catalogue"):
        _scenario(mutation_catalogue_name="nope")


def test_restricted_suite_roundtrip_writes_format_1_2_and_unrestricted_stays_1_1(tmp_path):
    suite = NearbyBaselineSuiteSpec(scenarios=(_scenario(),), config=NearbyBaselineConfig())
    path = tmp_path / "d5.json"
    save_nearby_baseline_suite_spec(path, suite)
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-nearby-baseline-suite-1.2"
    assert payload["scenarios"][0]["mutation_catalogue"] == "default+followup"
    restored = load_nearby_baseline_suite_spec(path)
    assert restored == suite
    assert nearby_scenario_graph(restored.scenarios[0]).root_hash == _alt_root().model_hash
    plain = NearbyBaselineSuiteSpec(
        scenarios=(NearbyBaselineScenario("plain", baseline_model_spec()),), config=NearbyBaselineConfig()
    )
    save_nearby_baseline_suite_spec(path, plain)
    payload = json.loads(path.read_text())
    assert payload["format_version"] == "gwpop-search-nearby-baseline-suite-1.1"
    assert "mutation_paths" not in payload["scenarios"][0]


def test_root_edge_log_bayes_factors_are_signed_and_keyed_by_path():
    scenario = _scenario()
    graph = nearby_scenario_graph(scenario)
    evidences = {m.model_hash: ModelEvidence(m.model_hash, -100.0 + 0.5 * i, 0.1) for i, m in enumerate(graph.nodes)}
    values = root_edge_log_bayes_factors(graph, evidences)
    by_hash = {m.model_hash: m for m in graph.nodes}
    assert "chieff.width.linear_q|chieff.mean.linear_z" in values
    assert "chieff.family.gaussian_mixture|chieff.fraction.logistic_q" in values
    assert "mass.family.broken_powerlaw" in values
    for edge in graph.edges:
        assert any(
            v == pytest.approx(evidences[edge.child_hash].log_evidence - evidences[edge.parent_hash].log_evidence)
            for v in values.values()
        )
    assert set(by_hash) >= {e.child_hash for e in graph.edges}
    # models without evidence are skipped
    assert root_edge_log_bayes_factors(graph, {}) == {}


def test_restricted_graph_limits_and_catalogue_registration():
    table = mutation_catalogue("default")
    with pytest.raises(ValueError, match="max_models"):
        restricted_model_graph(baseline_model_spec(), [("chieff.mean.linear_q",), ("chieff.width.linear_q",)],
                               table, max_models=2)
    register_mutation_catalogue("test-catalogue", DEFAULT_MUTATIONS[:3])
    register_mutation_catalogue("test-catalogue", DEFAULT_MUTATIONS[:3])  # identical: allowed
    with pytest.raises(ValueError, match="already registered"):
        register_mutation_catalogue("test-catalogue", DEFAULT_MUTATIONS[:4])
    with pytest.raises(ValueError, match="deeper than max_depth"):
        NearbyBaselineScenario("x", baseline_model_spec(), max_depth=1,
                               mutation_paths=(("chieff.mean.linear_q", "chieff.width.linear_q"),))


def test_cli_writes_a_v2_alt_root_config(tmp_path):
    root_path = tmp_path / "root.json"
    save_model_spec(root_path, _alt_root())
    out = tmp_path / "a1.json"
    argv = ["write-v2-alt-root-config", "--scenario-id", "A1", "--root-model", str(root_path),
            "--candidate", "mass.family.broken_powerlaw", "--candidate", "chieff.width.linear_q,chieff.mean.linear_z",
            "--mutation-catalogue", "default+followup", "--output", str(out)]
    for label, path in CHI_EFF.items():
        argv += ["--chi-eff-atom", f"{label}={path if isinstance(path, str) else ','.join(path)}"]
    args = build_parser().parse_args(argv)
    args.func(args)
    suite = load_nearby_baseline_suite_spec(out)
    scenario = suite.scenarios[0]
    assert scenario.mutation_paths == _scenario(candidate_paths=[
        "mass.family.broken_powerlaw", ("chieff.width.linear_q", "chieff.mean.linear_z")]).mutation_paths
    args = build_parser().parse_args(argv[:-2] + ["--chi-eff-atom", "C1", "--output", str(out)])
    with pytest.raises(ValueError, match="LABEL=MUTATION_ID"):
        args.func(args)


def test_restricted_suite_run_writes_signed_edge_bayes_factors(tmp_path):
    """End to end on a tiny synthetic catalog: the suite evaluates exactly the
    restricted graph and its summary carries what the v2 claim table's D5 reads."""
    pytest.importorskip("jax").config.update("jax_enable_x64", True)
    pytest.importorskip("dynesty")
    from gwpop_search.analysis.claims_v2 import alt_root_edge_values
    from gwpop_search.inference.synthetic import generate_baseline_synthetic_dataset
    from gwpop_search.store import ResultStore
    from gwpop_search.validation import run_nearby_baseline_suite
    from test_parallel_nearby_and_spec_evaluation import _campaign, _graph, _survey

    dataset = generate_baseline_synthetic_dataset(seed=77, config=_survey())
    root = apply_mutation(baseline_model_spec(), _mut("mass.family.powerlaw"))
    scenario = NearbyBaselineScenario(
        "A2_restricted", root, max_depth=1, max_models=3,
        mutation_paths=(("chieff.mean.linear_q",), ("mass.family.powerlaw",)),
    )
    suite = NearbyBaselineSuiteSpec(
        scenarios=(scenario,),
        config=NearbyBaselineConfig(max_gpu_hours_per_scenario=25.0, max_f3_models=4, max_f4_models=2),
    )
    graph = _graph()
    summary = run_nearby_baseline_suite(
        tmp_path, dataset.posterior, dataset.selection, graph, _campaign(graph),
        base_dataset_identity="c" * 64, suite=suite,
    )
    row = summary["scenarios"][0]
    assert row["n_graph_models"] == 2
    assert row["inapplicable_paths"] == ["mass.family.powerlaw"]
    values = row["root_edge_log_bayes_factors"]
    assert list(values) == ["chieff.mean.linear_q"]
    store = ResultStore(tmp_path / "A2_restricted" / "state.sqlite")
    assert len({r["model_hash"] for r in store.evaluations(fidelity="F3")}) == 2
    parsed = alt_root_edge_values(summary)
    assert parsed["values"] == values and parsed["inapplicable"] == {"mass.family.powerlaw"}
