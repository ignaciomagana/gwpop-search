import json
import math
import sys
import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("dynesty")

from gwpop_search.cli import build_parser, main  # noqa: E402
from gwpop_search.grammar import (  # noqa: E402
    DEFAULT_MUTATIONS,
    ModelEdge,
    ModelGraph,
    apply_mutation,
    baseline_model_spec,
    save_model_graph,
)
from gwpop_search.hbi import HBIConfig  # noqa: E402
from gwpop_search.inference import DynestyConfig, prior_specs_from_model_spec, run_dynesty_population  # noqa: E402
from gwpop_search.inference.synthetic import SyntheticSurveyConfig, generate_baseline_synthetic_dataset  # noqa: E402
from gwpop_search.models import compile_model_spec  # noqa: E402

MUT = {m.mutation_id: m for m in DEFAULT_MUTATIONS}


def _run_cli(argv):
    old = sys.argv
    sys.argv = ["gwpop-search", *argv]
    try:
        main()
    finally:
        sys.argv = old


def test_analysis_subcommands_parse():
    parser = build_parser()
    args = parser.parse_args(["analyze-model-comparison", "--graph", "g.json", "--evidence", "e.json",
                              "--penalty", "0.693", "--output", "o.json"])
    assert args.ns_error_mode == "formula6" and args.alpha == 0.01 and args.lambdas == "0,ln2,ln4"
    args = parser.parse_args(["run-psis-loo-influence", "--graph", "g.json", "--results", "r", "--pe", "p.h5",
                              "--selection", "s.h5", "--output", "o.json"])
    assert args.k_threshold == 0.7 and args.refit_stop_fidelity == "F3"
    args = parser.parse_args(["run-holdout-validation", "--manifest", "m", "--graph", "g", "--campaign", "c",
                              "--model-hash", "h", "--root", "r"])
    assert (args.posterior_nlive, args.posterior_repeats, args.gate_profile) == (1000, 2, "f4")
    with pytest.raises(SystemExit):
        parser.parse_args(["write-loo-stress-config", "--manifest", "m", "--output", "o", "--stop-fidelity", "F2"])


@pytest.fixture(scope="module")
def tiny_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis-cli")
    config = SyntheticSurveyConfig(n_events=6, posterior_samples_per_event=32, n_injections=3_000,
                                   population_batch_size=512, redshift_sampling_grid=1024)
    dataset = generate_baseline_synthetic_dataset(seed=5, config=config)
    pe_path, sel_path = root / "pe.h5", root / "selection.h5"
    dataset.posterior.to_hdf5(pe_path)
    dataset.selection.to_hdf5(sel_path)
    parent = baseline_model_spec()
    child = apply_mutation(parent, MUT["chieff.mean.linear_m1"])
    graph = ModelGraph(
        root_hash=parent.model_hash,
        nodes=(parent, child),
        edges=(ModelEdge(parent.model_hash, child.model_hash, "chieff.mean.linear_m1", 1),),
        depths={parent.model_hash: 0, child.model_hash: 1},
    )
    graph_path = root / "graph.json"
    save_model_graph(graph_path, graph)
    cfg = DynestyConfig(nlive=40, sample="rslice", batch_size=16, maxiter=120, num_posterior_samples=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in (parent, child):
            for seed in (1, 2):
                run_dynesty_population(
                    dataset.posterior, dataset.selection, compile_model_spec(spec),
                    prior_specs_from_model_spec(spec), seed=seed, config=cfg,
                    hbi_config=HBIConfig(selection_chunk_size=None),
                    run_dir=root / "runs" / spec.model_hash / f"repeat_{seed:03d}",
                )
    return root, pe_path, sel_path, graph_path, parent, child


def test_analysis_commands_run_end_to_end_on_saved_results(tiny_campaign):
    root, pe_path, sel_path, graph_path, parent, child = tiny_campaign
    data = ["--pe", str(pe_path), "--selection", str(sel_path)]
    common = ["--graph", str(graph_path), "--results", str(root / "runs")]

    _run_cli(["classify-graph-edges", "--graph", str(graph_path), "--verify", "--output", str(root / "edges.json")])
    edges = json.loads((root / "edges.json").read_text())
    assert edges["edges"][0]["classification"] == "exact"
    assert edges["numerical_verification"][0]["all_nested"]

    _run_cli(["analyze-edge-mc-error", *data, *common, "--n-draws", "200", "--bootstrap-replicates", "4",
              "--bootstrap-draws", "100", "--min-ess", "1", "--output-dir", str(root / "mc")])
    mc = json.loads((root / "mc" / "edge_mc_error.json").read_text())
    assert set(mc["models"]) == {parent.model_hash, child.model_hash}
    edge = mc["edges"][0]
    assert edge["predicted"]["sigma"] >= 0 and edge["orientation"].startswith("ln BF = ln Z_child")
    assert (root / "mc" / "mc_weights" / f"{child.model_hash}.npz").exists()
    assert np.asarray(mc["sigma_mc"]["matrix"]).shape == (2, 2)

    _run_cli(["analyze-prior-sensitivity", *common, "--output", str(root / "prior.json")])
    prior = json.loads((root / "prior.json").read_text())
    assert prior["edges"][0]["parameters"][0]["parameter"] == "chi_mu_m1_slope"
    assert len(prior["model_prior_variants"]) == 3

    _run_cli(["analyze-sddr", *common, "--n-bootstrap", "20", "--output", str(root / "sddr.json")])
    sddr = json.loads((root / "sddr.json").read_text())
    row = sddr["edges"][0]
    assert row["mutation_id"] == "chieff.mean.linear_m1" and row["classification"] == "exact"
    assert row["status"] in {"agree", "disagree", "not_estimable", "bound_consistent", "bound_violated"}

    _run_cli(["run-psis-loo-influence", *data, *common, "--penalty", str(math.log(2.0)),
              "--write-exact-refit-config", str(root / "refit.json"), "--output", str(root / "loo.json")])
    loo = json.loads((root / "loo.json").read_text())
    assert set(loo["models"]) == {parent.model_hash, child.model_hash}
    assert len(loo["edges"]) == 1 and len(loo["edges"][0]["events"]) == 6
    assert "per_event" in loo["model_probabilities"]

    nulls = root / "nulls.json"
    nulls.write_text(json.dumps({"label": "root-median", "statistic": "max_edge_log_bayes_factor",
                                 "observed": 1.0, "exceedances": 3, "replays": 99}))
    _run_cli(["analyze-model-comparison", *common, "--penalty", str(math.log(2.0)),
              "--mc-dir", str(root / "mc" / "mc_weights"), "--sddr", str(root / "sddr.json"),
              "--prior-sensitivity", str(root / "prior.json"), "--loo", str(root / "loo.json"),
              "--null-calibration", str(nulls), "--n-propagation", "200",
              "--output", str(root / "report.json"), "--markdown", str(root / "report.md")])
    report = json.loads((root / "report.json").read_text())
    assert report["evidence_coverage"]["complete"]
    claim = report["claims"][0]
    assert claim["mutation_id"] == "chieff.mean.linear_m1"
    assert claim["status"] in {"not_claimed", "incomplete"}
    assert claim["criteria"]["null_calibration"]["status"] == "fail"  # p = 0.04 > 0.01
    assert "# Model comparison" in (root / "report.md").read_text()
