import json

import pytest

from gwpop_search.grammar import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MUTATIONS,
    InapplicableMutation,
    ModelSpec,
    PriorConfig,
    apply_mutation,
    baseline_model_spec,
    enumerate_model_graph,
    save_model_graph,
    save_model_spec,
    load_model_graph,
    load_model_spec,
    structural_diff_axes,
)


def test_baseline_model_spec_is_canonical_and_roundtrips():
    model = baseline_model_spec()
    restored = ModelSpec.from_json(model.canonical_json())

    assert restored == model
    assert restored.model_hash == model.model_hash
    assert len(model.model_hash) == 64
    assert model.short_hash == model.model_hash[:16]
    assert json.loads(model.canonical_json()) == model.to_dict()


def test_model_hash_is_independent_of_prior_mapping_insertion_order():
    model = baseline_model_spec()
    reversed_priors = dict(reversed(list(model.priors.items())))
    reordered = type(model)(
        mass=model.mass,
        pairing=model.pairing,
        chieff=model.chieff,
        redshift=model.redshift,
        mixture=model.mixture,
        priors=reversed_priors,
    )
    assert reordered.model_hash == model.model_hash


def test_hyperprior_change_changes_scientific_model_hash():
    model = baseline_model_spec()
    priors = dict(model.priors)
    priors["alpha"] = PriorConfig("uniform", {"low": 0.0, "high": 7.0})
    changed = type(model)(
        mass=model.mass,
        pairing=model.pairing,
        chieff=model.chieff,
        redshift=model.redshift,
        mixture=model.mixture,
        priors=priors,
    )
    assert changed.model_hash != model.model_hash
    assert structural_diff_axes(model, changed) == ()


def test_every_default_mutation_changes_exactly_its_declared_axis():
    root = baseline_model_spec()
    for mutation in DEFAULT_MUTATIONS:
        child = apply_mutation(root, mutation)
        assert child.model_hash != root.model_hash
        assert structural_diff_axes(root, child) == (mutation.axis,)
        DEFAULT_COMPONENT_REGISTRY.validate_model(child)


def test_mass_family_mutation_removes_irrelevant_peak_priors():
    root = baseline_model_spec()
    mutation = next(
        item for item in DEFAULT_MUTATIONS
        if item.mutation_id == "mass.family.powerlaw"
    )
    child = apply_mutation(root, mutation)

    assert child.mass.family == "powerlaw"
    assert "peak_fraction" not in child.priors
    assert "peak_mu" not in child.priors
    assert "peak_sigma" not in child.priors
    assert "alpha" in child.priors


def test_reapplying_same_option_mutation_is_inapplicable():
    root = baseline_model_spec()
    mutation = next(
        item for item in DEFAULT_MUTATIONS
        if item.mutation_id == "chieff.width.linear_q"
    )
    child = apply_mutation(root, mutation)
    with pytest.raises(InapplicableMutation):
        apply_mutation(child, mutation)


def test_registry_rejects_illegal_option_choice():
    root = baseline_model_spec()
    bad = type(root)(
        mass=root.mass,
        pairing=type(root.pairing)(
            family="powerlaw_q",
            options={"beta_dependence": "not-a-dependence"},
        ),
        chieff=root.chieff,
        redshift=root.redshift,
        mixture=root.mixture,
        priors=root.priors,
    )
    with pytest.raises(ValueError, match="illegal"):
        DEFAULT_COMPONENT_REGISTRY.validate_model(bad)


def test_deterministic_breadth_first_graph_contains_20_to_40_models():
    root = baseline_model_spec()
    a = enumerate_model_graph(root, max_depth=2, max_models=40)
    b = enumerate_model_graph(
        root,
        mutations=tuple(reversed(DEFAULT_MUTATIONS)),
        max_depth=2,
        max_models=40,
    )

    assert 20 <= len(a.nodes) <= 40
    assert [item.model_hash for item in a.nodes] == [
        item.model_hash for item in b.nodes
    ]
    assert [edge.to_dict() for edge in a.edges] == [
        edge.to_dict() for edge in b.edges
    ]
    assert a.root_hash == root.model_hash
    assert a.depths[root.model_hash] == 0
    assert max(a.depths.values()) <= 2


def test_every_graph_edge_is_a_single_structural_mutation():
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=2,
        max_models=40,
    )
    by_hash = graph.by_hash
    mutation_ids = {item.mutation_id for item in DEFAULT_MUTATIONS}

    for edge in graph.edges:
        assert edge.mutation_id in mutation_ids
        axes = structural_diff_axes(
            by_hash[edge.parent_hash],
            by_hash[edge.child_hash],
        )
        assert len(axes) == 1



def test_model_spec_json_and_yaml_roundtrip(tmp_path):
    model = baseline_model_spec()
    for suffix in (".json", ".yaml"):
        path = tmp_path / f"model{suffix}"
        save_model_spec(path, model)
        restored = load_model_spec(path)
        assert restored == model
        assert restored.model_hash == model.model_hash


def test_model_graph_serialization_contains_hashes_and_edges(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=1,
        max_models=20,
    )
    path = tmp_path / "graph.json"
    save_model_graph(path, graph)
    payload = json.loads(path.read_text())

    assert payload["root_hash"] == graph.root_hash
    assert len(payload["nodes"]) == len(graph.nodes)
    assert len(payload["edges"]) == len(graph.edges)
    assert payload["nodes"][0]["model_hash"] == graph.root_hash



def test_model_graph_roundtrip_revalidates_hashes_and_edges(tmp_path):
    graph = enumerate_model_graph(
        baseline_model_spec(),
        max_depth=2,
        max_models=30,
    )
    path = tmp_path / "graph.json"
    save_model_graph(path, graph)
    restored = load_model_graph(path)

    assert restored.root_hash == graph.root_hash
    assert [model.model_hash for model in restored.nodes] == [
        model.model_hash for model in graph.nodes
    ]
    assert [edge.to_dict() for edge in restored.edges] == [
        edge.to_dict() for edge in graph.edges
    ]

    payload = json.loads(path.read_text())
    payload["nodes"][0]["model_hash"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model hash mismatch"):
        load_model_graph(path)
