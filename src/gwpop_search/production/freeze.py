"""Helpers to hash/freeze graph and campaign inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from gwpop_search.grammar import ModelGraph


def canonical_graph_json(graph: ModelGraph) -> str:
    return json.dumps(
        graph.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def model_graph_hash(graph: ModelGraph) -> str:
    return hashlib.sha256(canonical_graph_json(graph).encode("utf-8")).hexdigest()


def verify_graph_file(path: str | Path) -> dict[str, object]:
    """Hash a model-graph file the way :func:`model_graph_hash` hashes the frozen graph.

    ``graph_hash`` is the hash of the loaded :class:`ModelGraph` (nodes, edges,
    root; every stored model hash re-verified on load), i.e. exactly what a
    campaign freeze records. A file may carry extra descriptive keys (the v2
    graph writer's ``metadata`` block: atom labels, DRAFT priors, G12 checks);
    they are not part of the frozen graph and are hashed separately as
    ``file_sha256`` (the canonical JSON of the whole file) for provenance.
    For a file without extra keys the two hashes coincide.
    """
    from gwpop_search.grammar import load_model_graph

    payload = json.loads(Path(path).read_text())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    file_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    graph = load_model_graph(Path(path))
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    extra = sorted(set(payload) - set(graph.to_dict()))
    return {
        "graph_hash": model_graph_hash(graph),
        "file_sha256": file_digest,
        "extra_keys": extra,
        "root_hash": payload.get("root_hash"),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }
