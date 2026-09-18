"""Serialization helpers for declarative models and model graphs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .enumerate import ModelGraph
from .schema import ModelSpec


def load_model_spec(path: str | Path) -> ModelSpec:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model specification must decode to a mapping")
    return ModelSpec.from_dict(payload)


def save_model_spec(path: str | Path, model: ModelSpec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    if path.suffix.lower() in {".yaml", ".yml"}:
        path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=True,
                default_flow_style=False,
            )
        )
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def save_model_graph(path: str | Path, graph: ModelGraph) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.to_dict(), sort_keys=True, indent=2))



def load_model_graph(path: str | Path) -> ModelGraph:
    """Load a serialized graph and verify all stored scientific identities."""
    payload = json.loads(Path(path).read_text())
    if payload.get("format_version") != "gwpop-search-model-graph-1.0":
        raise ValueError("unsupported model graph format")

    nodes = tuple(
        ModelSpec.from_dict(item["spec"])
        for item in payload["nodes"]
    )
    stored_hashes = [str(item["model_hash"]) for item in payload["nodes"]]
    actual_hashes = [model.model_hash for model in nodes]
    if stored_hashes != actual_hashes:
        raise ValueError("stored model graph contains a model hash mismatch")
    if len(set(actual_hashes)) != len(actual_hashes):
        raise ValueError("stored model graph contains duplicate model hashes")

    depths = {
        str(item["model_hash"]): int(item["depth"])
        for item in payload["nodes"]
    }
    edges = tuple(
        ModelEdge(
            parent_hash=str(item["parent_hash"]),
            child_hash=str(item["child_hash"]),
            mutation_id=str(item["mutation_id"]),
            depth=int(item["depth"]),
        )
        for item in payload["edges"]
    )
    node_hashes = set(actual_hashes)
    for edge in edges:
        if edge.parent_hash not in node_hashes or edge.child_hash not in node_hashes:
            raise ValueError("stored model graph edge references an unknown node")
        if depths[edge.child_hash] != edge.depth:
            raise ValueError("stored model graph edge depth disagrees with child node")

    root_hash = str(payload["root_hash"])
    if root_hash not in node_hashes:
        raise ValueError("stored model graph root is not a node")
    if depths[root_hash] != 0:
        raise ValueError("stored model graph root must have depth zero")

    return ModelGraph(
        root_hash=root_hash,
        nodes=nodes,
        edges=edges,
        depths=depths,
    )
