"""Deterministic finite enumeration of the declarative model graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .mutations import (
    DEFAULT_MUTATIONS,
    InapplicableMutation,
    MutationSpec,
    apply_mutation,
)
from .registry import DEFAULT_COMPONENT_REGISTRY, ComponentRegistry
from .schema import ModelSpec


@dataclass(frozen=True)
class ModelEdge:
    parent_hash: str
    child_hash: str
    mutation_id: str
    depth: int

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_hash": self.parent_hash,
            "child_hash": self.child_hash,
            "mutation_id": self.mutation_id,
            "depth": int(self.depth),
        }


@dataclass(frozen=True)
class ModelGraph:
    root_hash: str
    nodes: tuple[ModelSpec, ...]
    edges: tuple[ModelEdge, ...]
    depths: dict[str, int]

    @property
    def by_hash(self) -> dict[str, ModelSpec]:
        return {model.model_hash: model for model in self.nodes}

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": "gwpop-search-model-graph-1.0",
            "root_hash": self.root_hash,
            "nodes": [
                {
                    "model_hash": model.model_hash,
                    "depth": self.depths[model.model_hash],
                    "spec": model.to_dict(),
                }
                for model in self.nodes
            ],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def enumerate_model_graph(
    root: ModelSpec,
    *,
    mutations: Iterable[MutationSpec] = DEFAULT_MUTATIONS,
    max_depth: int = 2,
    max_models: int = 40,
    registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
) -> ModelGraph:
    """Breadth-first deterministic model enumeration.

    Mutation order is canonicalized by mutation_id, so enumeration is replayable
    independently of caller/container iteration order.
    """
    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    if max_models <= 0:
        raise ValueError("max_models must be positive")

    registry.validate_model(root)
    ordered_mutations = tuple(sorted(mutations, key=lambda item: item.mutation_id))

    nodes: list[ModelSpec] = [root]
    by_hash: dict[str, ModelSpec] = {root.model_hash: root}
    depths: dict[str, int] = {root.model_hash: 0}
    edges: list[ModelEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()

    cursor = 0
    while cursor < len(nodes):
        parent = nodes[cursor]
        parent_depth = depths[parent.model_hash]
        cursor += 1
        if parent_depth >= max_depth:
            continue

        for mutation in ordered_mutations:
            try:
                child = apply_mutation(parent, mutation, registry=registry)
            except InapplicableMutation:
                continue

            child_hash = child.model_hash
            if child_hash not in by_hash:
                if len(nodes) >= max_models:
                    continue
                by_hash[child_hash] = child
                depths[child_hash] = parent_depth + 1
                nodes.append(child)

            key = (parent.model_hash, child_hash, mutation.mutation_id)
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    ModelEdge(
                        parent_hash=parent.model_hash,
                        child_hash=child_hash,
                        mutation_id=mutation.mutation_id,
                        depth=parent_depth + 1,
                    )
                )

    return ModelGraph(
        root_hash=root.model_hash,
        nodes=tuple(nodes),
        edges=tuple(edges),
        depths=depths,
    )
