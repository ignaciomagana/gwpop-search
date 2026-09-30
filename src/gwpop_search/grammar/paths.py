"""Mutation paths: root-independent names for the nodes and edges of a model graph.

The same atom (or pair of atoms) applied to two different roots gives nodes
with different hashes; the v2 alternative-root check (D5) compares an edge of
the searched graph with "the same" edge on an alternative root. Both are named
by mutation ids:

* a node's path is the sorted tuple of mutation ids applied to the root to
  reach it (a shortest path; for the one-axis grammar every path to a node
  applies the same set);
* an edge's key is ``"<mutation id>"`` for an edge out of the root and
  ``"<a>[+<b>...]|<mutation id>"`` otherwise, i.e. its parent's path and the
  mutation it applies. Two edges into the same depth-2 node (``a`` then
  ``b``, or ``b`` then ``a``) therefore have different keys.
"""

from __future__ import annotations

from typing import Sequence


def mutation_paths(graph) -> dict[str, tuple[str, ...]]:
    """``{model_hash: sorted mutation ids of a shortest path from the root}``."""
    paths: dict[str, tuple[str, ...]] = {graph.root_hash: ()}
    children: dict[str, list] = {}
    for edge in graph.edges:
        children.setdefault(edge.parent_hash, []).append(edge)
    frontier = [graph.root_hash]
    while frontier:
        nxt = []
        for model_hash in frontier:
            for edge in children.get(model_hash, []):
                if edge.child_hash not in paths:
                    paths[edge.child_hash] = tuple(sorted(paths[model_hash] + (edge.mutation_id,)))
                    nxt.append(edge.child_hash)
        frontier = nxt
    return paths


def edge_path_key(parent_path: Sequence[str], mutation_id: str) -> str:
    """Root-independent key of the edge that applies ``mutation_id`` to ``parent_path``."""
    parent_path = tuple(sorted(str(x) for x in parent_path))
    if not parent_path:
        return str(mutation_id)
    return "+".join(parent_path) + "|" + str(mutation_id)


def path_edge_key(path: Sequence[str]) -> str:
    """Key of the last edge of an ordered mutation ``path`` (``(a, b)`` -> ``"a|b"``)."""
    path = tuple(str(x) for x in path)
    if not path:
        raise ValueError("a mutation path must contain at least one mutation")
    return edge_path_key(path[:-1], path[-1])


def graph_edge_keys(graph) -> dict[tuple[str, str, str], str]:
    """``{(parent_hash, child_hash, mutation_id): edge key}`` for every edge of ``graph``."""
    paths = mutation_paths(graph)
    return {
        (edge.parent_hash, edge.child_hash, edge.mutation_id): edge_path_key(paths[edge.parent_hash], edge.mutation_id)
        for edge in graph.edges
        if edge.parent_hash in paths
    }
