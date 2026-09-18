import json

from gwpop_search.cli import build_parser
from gwpop_search.grammar import (
    baseline_model_spec,
    enumerate_model_graph,
    load_model_spec,
    save_model_graph,
)


def test_inspect_model_graph_cli_reports_structural_inventory(tmp_path, capsys):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=5,
    )
    path = tmp_path / "graph.json"
    save_model_graph(path, graph)

    args = build_parser().parse_args(
        ["inspect-model-graph", "--graph", str(path)]
    )
    args.func(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["root_hash"] == graph.root_hash
    assert payload["n_nodes"] == len(graph.nodes)
    assert payload["n_edges"] == len(graph.edges)
    assert {item["model_hash"] for item in payload["nodes"]} == {
        model.model_hash for model in graph.nodes
    }


def test_extract_model_cli_materializes_exact_frozen_spec(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=5,
    )
    graph_path = tmp_path / "graph.json"
    output = tmp_path / "model.json"
    save_model_graph(graph_path, graph)
    target = graph.nodes[-1]

    args = build_parser().parse_args(
        [
            "extract-model",
            "--graph",
            str(graph_path),
            "--model-hash",
            target.model_hash,
            "--output",
            str(output),
        ]
    )
    args.func(args)

    restored = load_model_spec(output)
    assert restored == target
    assert restored.model_hash == target.model_hash
