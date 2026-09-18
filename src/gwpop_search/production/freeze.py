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
    payload = json.loads(Path(path).read_text())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    return {
        "graph_hash": digest,
        "root_hash": payload.get("root_hash"),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }
