"""Canonical declarative population-model specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _canonicalize(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model specifications cannot contain NaN or infinity")
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    raise TypeError(f"unsupported model-spec value type: {type(value).__name__}")


@dataclass(frozen=True)
class PriorConfig:
    """Serializable hyperprior declaration independent of inference backend."""

    family: str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        if not family:
            raise ValueError("prior family cannot be empty")
        object.__setattr__(self, "family", family)

        parameters = {str(k): float(v) for k, v in self.parameters.items()}
        if not all(math.isfinite(value) for value in parameters.values()):
            raise ValueError("prior parameters must be finite")

        if family in {"uniform", "log_uniform"}:
            if set(parameters) != {"low", "high"}:
                raise ValueError(f"{family} requires exactly low/high")
            if parameters["high"] <= parameters["low"]:
                raise ValueError(f"{family} requires high > low")
            if family == "log_uniform" and parameters["low"] <= 0.0:
                raise ValueError("log_uniform low must be positive")
        elif family == "normal":
            if set(parameters) != {"loc", "scale"}:
                raise ValueError("normal requires exactly loc/scale")
            if parameters["scale"] <= 0.0:
                raise ValueError("normal scale must be positive")
        else:
            raise ValueError(f"unsupported prior family {family!r}")

        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "family": self.family,
            "parameters": _canonicalize(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PriorConfig":
        return cls(
            family=str(payload["family"]),
            parameters=dict(payload["parameters"]),
        )


@dataclass(frozen=True)
class BlockSpec:
    """One population-model block and its structural options."""

    family: str
    options: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = str(self.family).strip()
        if not family:
            raise ValueError("component family cannot be empty")
        object.__setattr__(self, "family", family)
        object.__setattr__(
            self,
            "options",
            _canonicalize(dict(self.options)),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "family": self.family,
            "options": _canonicalize(self.options),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BlockSpec":
        return cls(
            family=str(payload["family"]),
            options=dict(payload.get("options", {})),
        )


_BLOCK_NAMES = ("mass", "pairing", "chieff", "redshift", "mixture")


@dataclass(frozen=True)
class ModelSpec:
    """Canonical scientific model specification.

    Sampler settings, datasets, compute resources, and model priors do not belong
    here. Hyperpriors do: changing a parameter prior changes the evidence-defined
    scientific model and therefore changes the model hash.
    """

    mass: BlockSpec
    pairing: BlockSpec
    chieff: BlockSpec
    redshift: BlockSpec
    mixture: BlockSpec
    priors: Mapping[str, PriorConfig] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError(f"unsupported model schema version {self.schema_version!r}")
        priors = {
            str(name): (
                prior
                if isinstance(prior, PriorConfig)
                else PriorConfig.from_dict(prior)
            )
            for name, prior in self.priors.items()
        }
        object.__setattr__(self, "priors", priors)

    def structure_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "blocks": {
                name: getattr(self, name).to_dict()
                for name in _BLOCK_NAMES
            },
        }

    def canonical_dict(self) -> dict[str, JsonValue]:
        payload = self.structure_dict()
        payload["priors"] = {
            name: self.priors[name].to_dict()
            for name in sorted(self.priors)
        }
        return _canonicalize(payload)

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    @property
    def model_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.model_hash[:16]

    def to_dict(self) -> dict[str, JsonValue]:
        return self.canonical_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        blocks = dict(payload["blocks"])
        return cls(
            mass=BlockSpec.from_dict(blocks["mass"]),
            pairing=BlockSpec.from_dict(blocks["pairing"]),
            chieff=BlockSpec.from_dict(blocks["chieff"]),
            redshift=BlockSpec.from_dict(blocks["redshift"]),
            mixture=BlockSpec.from_dict(blocks["mixture"]),
            priors={
                name: PriorConfig.from_dict(prior)
                for name, prior in dict(payload.get("priors", {})).items()
            },
            schema_version=str(payload.get("schema_version", "1.0")),
        )

    @classmethod
    def from_json(cls, text: str) -> "ModelSpec":
        return cls.from_dict(json.loads(text))


def structural_diff_axes(parent: ModelSpec, child: ModelSpec) -> tuple[str, ...]:
    """Return semantic structure axes changed between two models.

    A family replacement is one atomic structural axis for that block; its new
    family-default options do not count as additional mutations.
    """
    axes: list[str] = []
    for block_name in _BLOCK_NAMES:
        a = getattr(parent, block_name)
        b = getattr(child, block_name)
        if a.family != b.family:
            axes.append(f"{block_name}.family")
            continue

        option_keys = set(a.options) | set(b.options)
        for key in sorted(option_keys):
            if a.options.get(key) != b.options.get(key):
                axes.append(f"{block_name}.options.{key}")
    return tuple(axes)
