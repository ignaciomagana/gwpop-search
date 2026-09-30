"""Full model specs of the exploratory chi_eff follow-up models.

Each follow-up model is built from the frozen GWTC-5 root by the typed
mutations of :data:`~gwpop_search.grammar.mutations.FOLLOWUP_MUTATIONS`
(preceded, for the q-dependent mixture, by the existing
``chieff.family.gaussian_mixture`` atom), so its spec and hash are exactly what
``apply_mutation`` produces. None of these models is reachable from
``DEFAULT_MUTATIONS``; the frozen graph is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from .io import load_model_graph, save_model_spec
from .mutations import DEFAULT_MUTATIONS, FOLLOWUP_MUTATION_TABLE, apply_mutation
from .registry import DEFAULT_COMPONENT_REGISTRY
from .schema import ModelSpec

#: Root node of frozen/gwtc5-bbh-v1a/model_graph.json (gwtc5-v1 baseline).
FROZEN_GWTC5_ROOT_HASH = (
    "666c4ffc99f000e8c3783589866f0ab6bc120bf161fca5f47b5130983263ff67"
)

#: name -> ordered mutation ids applied to the root.
FOLLOWUP_MODEL_PATHS: dict[str, tuple[str, ...]] = {
    "chieff_student_t": ("chieff.family.student_t",),
    "chieff_mixture_logistic_q_fraction": (
        "chieff.family.gaussian_mixture",
        "chieff.fraction.logistic_q",
    ),
    "chieff_width_logistic_q": ("chieff.width.logistic_q",),
}


def _mutation_table():
    table = {item.mutation_id: item for item in DEFAULT_MUTATIONS}
    table.update(FOLLOWUP_MUTATION_TABLE)
    return table


def followup_model_specs(root: ModelSpec) -> dict[str, ModelSpec]:
    """Return ``{name: spec}`` for every follow-up model built on ``root``."""
    table = _mutation_table()
    result: dict[str, ModelSpec] = {}
    for name, path in FOLLOWUP_MODEL_PATHS.items():
        model = root
        for mutation_id in path:
            model = apply_mutation(model, table[mutation_id])
        DEFAULT_COMPONENT_REGISTRY.validate_model(model)
        result[name] = model
    return result


def load_frozen_root(model_graph_path: str | Path) -> ModelSpec:
    """Load the frozen graph (verifying every stored hash) and return its root."""
    graph = load_model_graph(model_graph_path)
    if graph.root_hash != FROZEN_GWTC5_ROOT_HASH:
        raise ValueError(
            f"unexpected root hash {graph.root_hash}; expected {FROZEN_GWTC5_ROOT_HASH}"
        )
    return graph.by_hash[graph.root_hash]


def write_followup_specs(root: ModelSpec, output_dir: str | Path) -> dict[str, object]:
    """Write ``<name>.json`` for each follow-up model plus ``manifest.json``.

    Every block option is written explicitly (the full canonical ModelSpec).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "root_hash": root.model_hash,
        "models": {},
    }
    for name, spec in followup_model_specs(root).items():
        path = output_dir / f"{name}.json"
        save_model_spec(path, spec)
        manifest["models"][name] = {
            "file": path.name,
            "model_hash": spec.model_hash,
            "mutation_path": list(FOLLOWUP_MODEL_PATHS[name]),
            "chieff": spec.chieff.to_dict(),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return manifest
