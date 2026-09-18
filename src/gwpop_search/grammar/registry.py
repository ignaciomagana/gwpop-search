"""Legal component families and structural option validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .schema import BlockSpec, JsonValue, ModelSpec


@dataclass(frozen=True)
class FamilyDefinition:
    block: str
    family: str
    default_options: Mapping[str, JsonValue] = field(default_factory=dict)
    option_choices: Mapping[str, tuple[JsonValue, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block or not self.family:
            raise ValueError("family definitions require block and family names")

    def validate(self, block: BlockSpec) -> None:
        if block.family != self.family:
            raise ValueError(
                f"definition for {self.block}.{self.family} cannot validate "
                f"family {block.family!r}"
            )
        allowed = set(self.default_options) | set(self.option_choices)
        unknown = set(block.options) - allowed
        if unknown:
            raise ValueError(
                f"unknown option(s) for {self.block}.{self.family}: {sorted(unknown)}"
            )
        for key, choices in self.option_choices.items():
            if key in block.options and block.options[key] not in choices:
                raise ValueError(
                    f"illegal {self.block}.{self.family} option {key}="
                    f"{block.options[key]!r}; allowed={choices}"
                )


class ComponentRegistry:
    def __init__(self, definitions: tuple[FamilyDefinition, ...]):
        table: dict[tuple[str, str], FamilyDefinition] = {}
        for definition in definitions:
            key = (definition.block, definition.family)
            if key in table:
                raise ValueError(f"duplicate component definition {key}")
            table[key] = definition
        self._table = table

    def definition(self, block: str, family: str) -> FamilyDefinition:
        try:
            return self._table[(block, family)]
        except KeyError as exc:
            known = sorted(
                f"{b}.{f}" for b, f in self._table if b == block
            )
            raise ValueError(
                f"unknown component family {block}.{family}; known={known}"
            ) from exc

    def block(self, block: str, family: str) -> BlockSpec:
        definition = self.definition(block, family)
        return BlockSpec(family=family, options=definition.default_options)

    def validate_model(self, model: ModelSpec) -> None:
        for block_name in ("mass", "pairing", "chieff", "redshift", "mixture"):
            block = getattr(model, block_name)
            self.definition(block_name, block.family).validate(block)

        components = model.mixture.options.get("components", 1)
        if model.mixture.family == "single" and components != 1:
            raise ValueError("single mixture family requires components=1")
        if model.mixture.family == "finite" and components not in (2, 3):
            raise ValueError("finite mixture supports exactly 2 or 3 components")


DEFAULT_COMPONENT_REGISTRY = ComponentRegistry(
    (
        FamilyDefinition("mass", "powerlaw"),
        FamilyDefinition("mass", "pl_peak", {"peak_count": 1}),
        FamilyDefinition("mass", "broken_powerlaw"),
        FamilyDefinition("mass", "pl_two_peak", {"peak_count": 2}),
        FamilyDefinition(
            "pairing",
            "powerlaw_q",
            {"beta_dependence": "constant"},
            {
                "beta_dependence": (
                    "constant",
                    "linear_m1",
                    "logistic_m1",
                )
            },
        ),
        FamilyDefinition("pairing", "truncated_gaussian_q"),
        FamilyDefinition(
            "chieff",
            "truncated_gaussian",
            {
                "components": 1,
                "mean_dependence": "constant",
                "width_dependence": "constant",
            },
            {
                "components": (1,),
                "mean_dependence": (
                    "constant",
                    "linear_m1",
                    "linear_q",
                    "linear_z",
                    "logistic_m1",
                ),
                "width_dependence": (
                    "constant",
                    "linear_m1",
                    "linear_q",
                    "linear_z",
                ),
            },
        ),
        FamilyDefinition(
            "chieff",
            "gaussian_mixture",
            {
                "components": 2,
                "fraction_dependence": "constant",
            },
            {
                "components": (2, 3),
                "fraction_dependence": (
                    "constant",
                    "linear_m1",
                    "linear_q",
                    "linear_z",
                ),
            },
        ),
        FamilyDefinition(
            "redshift",
            "powerlaw",
            {"kappa_dependence": "constant"},
            {"kappa_dependence": ("constant", "linear_m1")},
        ),
        FamilyDefinition("redshift", "madau_dickinson"),
        FamilyDefinition("mixture", "single", {"components": 1}, {"components": (1,)}),
        FamilyDefinition(
            "mixture",
            "finite",
            {"components": 2, "fraction_dependence": "constant"},
            {
                "components": (2, 3),
                "fraction_dependence": (
                    "constant",
                    "linear_m1",
                    "linear_q",
                    "linear_z",
                ),
            },
        ),
    )
)
