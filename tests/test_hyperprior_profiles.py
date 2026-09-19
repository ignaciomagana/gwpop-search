import json

import pytest

from gwpop_search.cli import build_parser
from gwpop_search.grammar import (
    DEFAULT_HYPERPRIOR_PROFILE,
    HYPERPRIOR_PROFILES,
    baseline_hyperprior_profile,
    baseline_model_spec,
    enumerate_model_graph,
)


def test_default_profile_reproduces_the_original_baseline_hash():
    # Pinned before profiles existed: the original baseline_model_spec() hash.
    assert DEFAULT_HYPERPRIOR_PROFILE == "phase3"
    assert baseline_model_spec().model_hash == (
        "6d0c93c5d606f64a0d86ca4979d570f55e7dac985322c74fd3de0da3569808d4"
    )
    assert baseline_model_spec("phase3").model_hash == baseline_model_spec().model_hash


def test_gwtc5_profile_changes_only_the_mmax_prior():
    phase3 = baseline_model_spec("phase3").to_dict()
    gwtc5 = baseline_model_spec("gwtc5-v1").to_dict()
    assert gwtc5["blocks"] == phase3["blocks"]
    changed = {
        name for name in phase3["priors"]
        if phase3["priors"][name] != gwtc5["priors"][name]
    }
    assert changed == {"mmax"}
    assert gwtc5["priors"]["mmax"] == {
        "family": "uniform", "parameters": {"high": 200.0, "low": 60.0}
    }
    assert set(gwtc5["priors"]) == set(phase3["priors"])


def test_profile_lookup_and_unknown_profile():
    for profile in HYPERPRIOR_PROFILES:
        assert baseline_hyperprior_profile(baseline_model_spec(profile)) == profile
    with pytest.raises(ValueError, match="unknown hyperprior profile"):
        baseline_model_spec("gwtc4")


def test_profile_graphs_have_same_structure_but_distinct_hashes():
    a = enumerate_model_graph(baseline_model_spec("phase3"), max_depth=1, max_models=100)
    b = enumerate_model_graph(baseline_model_spec("gwtc5-v1"), max_depth=1, max_models=100)
    assert len(a.nodes) == len(b.nodes) == 15
    assert sorted(e.mutation_id for e in a.edges) == sorted(e.mutation_id for e in b.edges)
    assert not ({m.model_hash for m in a.nodes} & {m.model_hash for m in b.nodes})
    for node in b.nodes:
        assert node.to_dict()["priors"]["mmax"]["parameters"]["high"] == 200.0


def test_enumerate_models_cli_profile(tmp_path, capsys):
    out = tmp_path / "graph.json"
    args = build_parser().parse_args(
        ["enumerate-models", "--output", str(out), "--max-depth", "1",
         "--max-models", "100", "--hyperprior-profile", "gwtc5-v1"]
    )
    args.func(args)
    graph = json.loads(out.read_text())
    assert graph["root_hash"] == baseline_model_spec("gwtc5-v1").model_hash
    assert len(graph["nodes"]) == 15
