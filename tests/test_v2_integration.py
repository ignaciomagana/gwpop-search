"""Integration of the v2 workstreams (adapter, models, taper, PPC/claims).

The wiring these tests pin down:

* adapter <-> models: a canonical sky-marginal gwcat-v2 pair feeds a v2 model;
  the model's declared support (zmax, q_floor, mmin/mmax, sky) is checked
  against the dataset (G12 model + data), and the production evaluator refuses
  a mismatched dataset before sampling;
* taper <-> executor: a v2 campaign cannot be frozen without the taper, and a
  v2 graph file with its metadata block validates against its freeze;
* taper <-> analysis: the untapered analysis evaluators accept a tapered run
  only under a named treatment; the data-variant reweighting uses the sampled
  likelihood's own taper; the variance they see is the likelihood's sigma^2;
* claims: D1 inputs (gates, taper mass) and atom labels come from the
  evaluator's evaluation.json files and the v2 graph metadata; the D5 builder
  resolves the v2 atoms from the registered ``gwtc5-v2`` catalogue.
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwcat_v2_fixtures as fx  # noqa: E402

from gwpop_search.analysis._common import WeightedPosterior  # noqa: E402
from gwpop_search.analysis.terms import (  # noqa: E402
    TAPER_TREATMENTS,
    BatchedCatalogTerms,
    CatalogWeightEvaluator,
    pad_catalog,
)
from gwpop_search.grammar.v2 import (  # noqa: E402
    V2_ATOM_IDS,
    enumerate_v2_depth1,
    v2_graph_payload,
    v2_root_model_spec,
)
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference.v2_numerics import v2_fidelity_run_config, v2_variance_taper  # noqa: E402
from gwpop_search.models import compile_model_spec  # noqa: E402
from gwpop_search.models.data_support import (  # noqa: E402
    V2DataSupportError,
    require_v2_data_support,
    v2_data_support_report,
)

TAPERED = HBIConfig(selection_chunk_size=None, variance_taper=v2_variance_taper())
UNTAPERED = HBIConfig(selection_chunk_size=None)
# LVK-like root point (sampled coordinates)
HP = {
    "alpha_1": 1.5, "alpha_2": 5.4, "m_break": 37.5, "mu_p10": 9.9, "sigma_p10": 0.8,
    "mu_p35": 33.0, "sigma_p35": 4.0, "lam_u_pl": 0.33, "lam_u_p10": 0.39, "lam_u_p35": 0.095,
    "mlow_1": 5.0, "delta_m_1": 4.0, "mlow_2_frac": 0.5, "delta_m_2": 4.0,
    "beta": 1.1, "kappa": 3.0, "chi_mu": 0.06, "chi_log_sigma": -2.3,
}


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("gwcat_v2")
    fx.write_v2_pe(d / "pe.h5")
    fx.write_v2_selection(d / "sel.h5")
    return fx.loaded(d / "pe.h5", d / "sel.h5")


@pytest.fixture(scope="module")
def root():
    return v2_root_model_spec()


@pytest.fixture(scope="module")
def model(root):
    return compile_model_spec(root)


def _points(model, n=6, seed=0):
    rng = np.random.default_rng(seed)
    names = tuple(model.spec.priors)
    X = np.array([[HP[k] for k in names]] * n, dtype=float)
    X[:, names.index("chi_log_sigma")] += rng.uniform(-0.5, 0.5, n)
    X[:, names.index("kappa")] += rng.uniform(-1.0, 1.0, n)
    return names, X


# --------------------------------------------------------------------------
# adapter <-> models
# --------------------------------------------------------------------------


def test_adapter_pair_feeds_the_v2_model_and_matches_its_support(pair, model):
    pe, sel = pair
    assert pe.basis.name == "gwcat_v2_chieff_sky_marginal" and "ra" not in pe.basis.coordinates
    assert set(model.required_fields) == set(pe.basis.coordinates)
    report = v2_data_support_report(model, pe, sel)
    assert report["pass"], report["checks"]
    names = {c["name"] for c in report["checks"]}
    assert {"sky.pe", "sky.selection", "zmax.selection_declared",
            "support.every_event_has_supported_samples"} <= names
    from gwpop_search.hbi.jax_backend import build_shape_log_likelihood

    value = build_shape_log_likelihood(pe, sel, model, config=TAPERED)({k: jax.numpy.asarray(v) for k, v in HP.items()})
    assert np.isfinite(float(value))


def test_support_check_refuses_mismatched_datasets(pair, model):
    pe, sel = pair
    wrong_z = dataclasses.replace(sel, metadata={**sel.metadata, "z_max": 2.5})
    with pytest.raises(V2DataSupportError, match="zmax.selection_declared"):
        require_v2_data_support(model, pe, wrong_z)
    missing_z = dataclasses.replace(sel, metadata={k: v for k, v in sel.metadata.items() if k != "z_max"})
    assert not v2_data_support_report(model, pe, missing_z)["pass"]
    assert not v2_data_support_report(model, pe, sel, policy={"z_max": 2.5})["pass"]
    # an event whose every sample lies below q_floor has no population support
    q = pe.samples["q"].copy()
    q[pe.event_slice(0)] = 0.01
    unsupported = dataclasses.replace(pe, samples={**pe.samples, "q": q}, availability=None)
    report = v2_data_support_report(model, unsupported, sel)
    assert not report["pass"]
    assert pe.event_names[0] in next(
        c for c in report["checks"] if c["name"] == "support.every_event_has_supported_samples"
    )["unsupported_events"]


def test_support_check_refuses_a_sky_resolved_pair_for_a_marginalized_model(tmp_path, model):
    fx.write_v2_pe(tmp_path / "pe.h5")
    fx.write_v2_selection(
        tmp_path / "sel.h5", sky_marginalized=False, sky_position_available=np.asarray([True]),
        ds_ra=np.linspace(0.1, 6.0, 12), ds_dec=np.linspace(-1.0, 1.0, 12),
    )
    from gwpop_search.data.adapters.gwcat_v2 import load_pe, load_selection

    sel = load_selection(tmp_path / "sel.h5")
    pe = load_pe(tmp_path / "pe.h5", sky_marginal=False)
    assert "ra" in sel.basis.coordinates
    report = v2_data_support_report(model, pe, sel)
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert {"sky.pe", "sky.selection"} <= failed


def test_evaluator_refuses_a_mismatched_dataset_before_sampling(pair, root, tmp_path):
    from gwpop_search.inference.fidelity import DeterministicHBIEvaluator
    from gwpop_search.search import Fidelity

    pe, sel = pair
    wrong = dataclasses.replace(sel, metadata={**sel.metadata, "z_max": 2.5})
    evaluator = DeterministicHBIEvaluator(pe, wrong, config=v2_fidelity_run_config())
    with pytest.raises(V2DataSupportError):
        evaluator.evaluate(root, Fidelity.F0_SANITY, seed=1, run_dir=tmp_path / "run")
    assert not (tmp_path / "run" / "evaluation.json").exists()


# --------------------------------------------------------------------------
# taper <-> executor / freeze
# --------------------------------------------------------------------------


def test_v2_campaign_freeze_requires_the_taper_and_accepts_graph_metadata(tmp_path, pair):
    from gwpop_search.inference.fidelity import FidelityRunConfig
    from gwpop_search.production import (
        SearchBudget,
        SeedPolicy,
        build_production_campaign,
        model_graph_hash,
        verify_graph_file,
    )
    from gwpop_search.production import build_dataset_manifest_from_canonical_files
    from gwpop_search.search import SchedulerConfig

    graph = enumerate_v2_depth1()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(v2_graph_payload(graph), sort_keys=True))
    inspected = verify_graph_file(path)
    assert inspected["graph_hash"] == model_graph_hash(graph)
    assert inspected["extra_keys"] == ["metadata"]
    assert inspected["file_sha256"] != inspected["graph_hash"]

    pair[0].to_hdf5(tmp_path / "pe.h5")
    pair[1].to_hdf5(tmp_path / "sel.h5")
    manifest = build_dataset_manifest_from_canonical_files(
        tmp_path / "pe.h5", tmp_path / "sel.h5", dataset_id="t", event_selection={}, waveform_policy={},
    )
    kwargs = dict(
        campaign_id="t", git_commit="0" * 40, model_prior={"version": "uniform-v1"},
        scheduler=SchedulerConfig(beam_width=1, exploration_quota=0, seed=0),
        seed_policy=SeedPolicy(root_seed=1),
        budget=SearchBudget(max_gpu_hours=1.0, max_f3_models=1, max_f4_models=1, max_null_replays=1),
        artifact_root="a", state_database="s.sqlite",
    )
    with pytest.raises(ValueError, match="variance taper"):
        build_production_campaign(manifest, graph, fidelity=FidelityRunConfig(), **kwargs)
    campaign = build_production_campaign(
        manifest, graph, fidelity=v2_fidelity_run_config(), require_root_profile="gwtc5-v2", **kwargs
    )
    assert campaign.fidelity.hbi.variance_taper.threshold == 1.0
    assert campaign.model_graph_hash == inspected["graph_hash"]


# --------------------------------------------------------------------------
# taper <-> analysis evaluators
# --------------------------------------------------------------------------


def test_analysis_evaluators_need_a_named_taper_treatment(pair, model):
    pe, sel = pair
    names = tuple(model.spec.priors)
    with pytest.raises(NotImplementedError, match="taper_treatment"):
        pad_catalog(pe, sel, model, hbi_config=TAPERED)
    with pytest.raises(ValueError, match="unknown taper_treatment"):
        pad_catalog(pe, sel, model, hbi_config=TAPERED, taper_treatment="ignore")
    with pytest.raises(NotImplementedError):
        BatchedCatalogTerms(pe, sel, model, names, hbi_config=TAPERED)
    for treatment in TAPER_TREATMENTS:
        pad_catalog(pe, sel, model, hbi_config=TAPERED, taper_treatment=treatment)
    terms = BatchedCatalogTerms(pe, sel, model, names, hbi_config=TAPERED, taper_treatment="taper_as_prior")
    assert terms.taper_treatment == "taper_as_prior"
    assert BatchedCatalogTerms(pe, sel, model, names, hbi_config=UNTAPERED).taper_treatment is None


def test_analysis_variance_is_the_likelihood_sigma2(pair, model):
    """The data-variant reweighting adds ln T(sigma^2) from the analysis moments;
    that sigma^2 must be the one inside the sampled likelihood."""
    from gwpop_search.inference.dynesty_backend import build_batched_log_likelihood

    pe, sel = pair
    names, X = _points(model)
    ev = CatalogWeightEvaluator(
        pad_catalog(pe, sel, model, hbi_config=TAPERED, taper_treatment="weights_only"), model, names
    )
    moments = ev.moments(X, np.full(X.shape[0], 1.0 / X.shape[0]))
    comps = build_batched_log_likelihood(pe, sel, model, names, hbi_config=TAPERED, batch_size=8,
                                         with_variance=True).components(X)
    np.testing.assert_allclose(moments["variance"], comps["variance"], rtol=1e-10)
    np.testing.assert_allclose(moments["log_likelihood"], comps["log_likelihood_untapered"], rtol=1e-10)
    np.testing.assert_allclose(
        comps["log_likelihood"], comps["log_likelihood_untapered"] + TAPERED.variance_taper.log_taper(comps["variance"]),
        rtol=1e-12,
    )


def test_data_variant_reweighting_uses_the_sampled_taper(pair, model):
    from gwpop_search.analysis.data_variants import make_data_variant, reweight_to_variant

    pe, sel = pair
    names, X = _points(model, n=8)
    sample = WeightedPosterior(
        names=names, points=X, weights=np.full(X.shape[0], 1.0 / X.shape[0]),
        run_index=np.zeros(X.shape[0], dtype=int), n_runs=1, source="test",
    )
    variant = make_data_variant("drop1", pe, sel, drop_events=(pe.event_names[-1],))
    auto = reweight_to_variant(sample, pe, sel, model, variant, hbi_config=TAPERED, verify_identity=False)
    explicit = reweight_to_variant(sample, pe, sel, model, variant, hbi_config=TAPERED, verify_identity=False,
                                   log_taper=TAPERED.variance_taper.log_taper)
    untapered = reweight_to_variant(sample, pe, sel, model, variant, hbi_config=UNTAPERED, verify_identity=False)
    assert auto["taper_applied"] and auto["taper"] == TAPERED.variance_taper.to_dict()
    assert auto["delta_log_evidence"] == pytest.approx(explicit["delta_log_evidence"], abs=1e-12)
    assert untapered["taper_applied"] is False and untapered["taper"] is None


def test_taper_declared_reads_the_hbi_config_key():
    from gwpop_search.analysis.data_variants import _taper_declared

    assert _taper_declared({"hbi_config": TAPERED.to_dict(), "data": {}})
    assert not _taper_declared({"hbi_config": UNTAPERED.to_dict(), "data": {}})


def test_ppc_accepts_a_tapered_posterior_with_weights_only(pair, model):
    from gwpop_search.analysis.ppc import PPCConfig, posterior_predictive_check

    pe, sel = pair
    names, X = _points(model, n=4)
    sample = WeightedPosterior(
        names=names, points=X, weights=np.full(X.shape[0], 0.25),
        run_index=np.zeros(X.shape[0], dtype=int), n_runs=1, source="test",
    )
    result = posterior_predictive_check(
        sample, pe, sel, model, config=PPCConfig(n_draws=8, seed=0, batch_size=4),
        hbi_config=TAPERED, verify_identity=False,
    )
    assert result.taper_treatment == "weights_only"
    assert result.to_dict(include_draws=False)["taper_treatment"] == "weights_only"


# --------------------------------------------------------------------------
# claims / D5 wiring
# --------------------------------------------------------------------------


def test_collect_v2_evaluations_and_graph_atom_labels(tmp_path):
    from gwpop_search.analysis._common import AnalysisInputError
    from gwpop_search.analysis.claims_v2 import atom_labels_from_graph_payload, collect_v2_evaluations

    def write(path, model_hash, fidelity, passed, mass):
        path.parent.mkdir(parents=True, exist_ok=True)
        diag = {"passed": passed}
        if mass is not None:
            diag["taper"] = {"pooled": {"posterior_mass_in_taper_region": mass}}
        path.write_text(json.dumps({"model_hash": model_hash, "fidelity": fidelity, "diagnostics": diag}))

    write(tmp_path / "F3" / "a" / "evaluation.json", "a", "F3", True, 0.02)
    write(tmp_path / "F3" / "b" / "evaluation.json", "b", "F3", False, None)
    write(tmp_path / "F0" / "a" / "evaluation.json", "a", "F0", True, None)
    out = collect_v2_evaluations([tmp_path])
    assert out["gates"] == {"a": True, "b": False}
    assert out["taper_mass"] == {"a": 0.02}
    write(tmp_path / "F4" / "a" / "evaluation.json", "a", "F4", True, 0.03)
    with pytest.raises(AnalysisInputError, match="more than one evaluation"):
        collect_v2_evaluations([tmp_path])
    labels = atom_labels_from_graph_payload(v2_graph_payload(enumerate_v2_depth1()))
    assert labels == {mid: atom for atom, mid in V2_ATOM_IDS.items()}


def test_v2_catalogue_is_registered_for_d5():
    from gwpop_search.grammar.v2 import v2_alternative_roots
    from gwpop_search.validation import V2_CHI_EFF_ATOMS, v2_alt_root_scenario, v2_chi_eff_atom_mutations
    from gwpop_search.validation.baselines import mutation_catalogue

    table = mutation_catalogue("gwtc5-v2")
    atoms = v2_chi_eff_atom_mutations()
    assert tuple(atoms) == V2_CHI_EFF_ATOMS and set(atoms.values()) <= set(table)
    scenario = v2_alt_root_scenario(
        "A1", v2_alternative_roots()["A1"], candidate_paths=[V2_ATOM_IDS["C2"]],
        chi_eff_mutation_ids=atoms, mutation_catalogue_name="gwtc5-v2",
    )
    assert scenario.max_models == 11 and scenario.mutation_catalogue == "gwtc5-v2"


def _cli(argv):
    from gwpop_search.cli import build_parser

    args = build_parser().parse_args(argv)
    args.func(args)


def test_write_v2_alt_root_config_cli_defaults_to_the_v2_atoms(tmp_path):
    out = tmp_path / "d5.json"
    _cli(["write-v2-alt-root-config", "--scenario-id", "A2", "--alt-root", "A2",
          "--candidate", V2_ATOM_IDS["C2"], "--output", str(out)])
    payload = json.loads(out.read_text())
    scenario = payload["scenarios"][0]
    assert scenario["mutation_catalogue"] == "gwtc5-v2"
    assert len(scenario["mutation_paths"]) == 10
    with pytest.raises(ValueError, match="exactly one"):
        _cli(["write-v2-alt-root-config", "--scenario-id", "x", "--output", str(out)])


def test_taper2_rows_from_rerun_evaluations(tmp_path):
    from gwpop_search.analysis._common import AnalysisInputError
    from gwpop_search.analysis.claims_v2 import d3_prior, taper2_rows_from_evaluations

    graph = enumerate_v2_depth1()
    edge = next(e for e in graph.edges if e.mutation_id == V2_ATOM_IDS["C2"])

    def write(model_hash, ln_z, threshold, passed=True):
        path = tmp_path / str(threshold) / model_hash / "evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model_hash": model_hash, "fidelity": "F3", "diagnostics": {
            "passed": passed, "evidence": {"log_evidence_mean": ln_z, "conservative_error": 0.2},
            "taper": {"pooled": {"taper": {"kind": "smooth", "threshold": threshold}}}}}))

    write(edge.parent_hash, 10.0, 2.0)
    write(edge.child_hash, 14.5, 2.0)
    rows = taper2_rows_from_evaluations(graph, [tmp_path / "2.0"])
    assert len(rows) == 1 and rows[0]["log_bayes_factor"] == pytest.approx(4.5) and rows[0]["valid"]
    claim = {"parent_hash": edge.parent_hash, "child_hash": edge.child_hash, "mutation_id": edge.mutation_id}
    assert d3_prior(claim, [], 5.0, taper2=rows)["taper2_status"] == "pass"
    assert d3_prior(claim, [], -5.0, taper2=rows)["taper2_status"] == "fail"
    write(edge.parent_hash, 10.0, 1.0)
    with pytest.raises(AnalysisInputError, match="taper-at-2"):
        taper2_rows_from_evaluations(graph, [tmp_path / "1.0"])
