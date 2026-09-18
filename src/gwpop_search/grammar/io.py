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
