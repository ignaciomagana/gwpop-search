"""Typed one-axis population-model mutations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .registry import ComponentRegistry, DEFAULT_COMPONENT_REGISTRY
from .schema import JsonValue, ModelSpec, PriorConfig, structural_diff_axes


class InapplicableMutation(ValueError):
    """Raised when a legal mutation does not apply to the current parent model."""


@dataclass(frozen=True)
class MutationSpec:
    mutation_id: str
    block: str
    operation: str
    value: JsonValue
    option: str | None = None
    requires_family: str | None = None
    prior_updates: Mapping[str, PriorConfig] = field(default_factory=dict)
    prior_removals: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not self.mutation_id:
            raise ValueError("mutation_id cannot be empty")
        if self.operation not in {"change_family", "set_option"}:
            raise ValueError(f"unsupported mutation operation {self.operation!r}")
        if self.operation == "set_option" and not self.option:
            raise ValueError("set_option mutations require an option name")
        if self.operation == "change_family" and self.option is not None:
            raise ValueError("change_family mutations cannot name an option")

    @property
    def axis(self) -> str:
        if self.operation == "change_family":
            return f"{self.block}.family"
        return f"{self.block}.options.{self.option}"


def apply_mutation(
    parent: ModelSpec,
    mutation: MutationSpec,
    *,
    registry: ComponentRegistry = DEFAULT_COMPONENT_REGISTRY,
) -> ModelSpec:
    block = getattr(parent, mutation.block)
    if mutation.requires_family and block.family != mutation.requires_family:
        raise InapplicableMutation(
            f"{mutation.mutation_id} requires {mutation.block} family "
            f"{mutation.requires_family!r}, found {block.family!r}"
        )

    if mutation.operation == "change_family":
        family = str(mutation.value)
        if block.family == family:
            raise InapplicableMutation(
                f"{mutation.mutation_id} would leave family unchanged"
            )
        child_block = registry.block(mutation.block, family)
    else:
        options = dict(block.options)
        if options.get(mutation.option) == mutation.value:
            raise InapplicableMutation(
                f"{mutation.mutation_id} would leave option unchanged"
            )
        options[str(mutation.option)] = mutation.value
        child_block = type(block)(family=block.family, options=options)

    priors = dict(parent.priors)
    for name in mutation.prior_removals:
        priors.pop(name, None)
    priors.update(mutation.prior_updates)

    child = replace(parent, **{mutation.block: child_block, "priors": priors})
    registry.validate_model(child)

    axes = structural_diff_axes(parent, child)
    if axes != (mutation.axis,):
        raise ValueError(
            f"mutation {mutation.mutation_id} declared axis {mutation.axis!r} "
            f"but changed structural axes {axes}"
        )
    return child


def _u(low, high):
    return PriorConfig("uniform", {"low": low, "high": high})


def _lu(low, high):
    return PriorConfig("log_uniform", {"low": low, "high": high})


DEFAULT_MUTATIONS: tuple[MutationSpec, ...] = (
    MutationSpec(
        "mass.family.powerlaw",
        "mass",
        "change_family",
        "powerlaw",
        requires_family="pl_peak",
        prior_removals=("peak_fraction", "peak_mu", "peak_sigma"),
        description="remove the Gaussian mass peak",
    ),
    MutationSpec(
        "mass.family.broken_powerlaw",
        "mass",
        "change_family",
        "broken_powerlaw",
        requires_family="pl_peak",
        prior_removals=("alpha", "peak_fraction", "peak_mu", "peak_sigma"),
        prior_updates={
            "alpha1": _u(0.0, 8.0),
            "alpha2": _u(0.0, 8.0),
            "break_fraction": _u(0.1, 0.9),
        },
        description="replace PL+peak with a continuous broken power law",
    ),
    MutationSpec(
        "mass.family.pl_two_peak",
        "mass",
        "change_family",
        "pl_two_peak",
        requires_family="pl_peak",
        prior_removals=("peak_fraction", "peak_mu", "peak_sigma"),
        prior_updates={
            "peak1_fraction": _u(0.0, 0.5),
            "peak1_mu": _u(15.0, 45.0),
            "peak1_sigma": _lu(1.0, 15.0),
            "peak2_fraction": _u(0.0, 0.5),
            "peak2_mu": _u(30.0, 80.0),
            "peak2_sigma": _lu(1.0, 20.0),
        },
        description="add a second primary-mass peak",
    ),
    MutationSpec(
        "pairing.family.truncated_gaussian_q",
        "pairing",
        "change_family",
        "truncated_gaussian_q",
        requires_family="powerlaw_q",
        prior_removals=(
            "beta_q",
            "beta_q_m1_slope",
            "delta_beta_q",
            "beta_q_m1_transition",
            "beta_q_m1_width",
        ),
        prior_updates={
            "q_mu": _u(0.2, 1.0),
            "q_sigma": _lu(0.02, 0.5),
        },
        description="replace power-law pairing with truncated-Gaussian q",
    ),
    MutationSpec(
        "pairing.beta.linear_m1",
        "pairing",
        "set_option",
        "linear_m1",
        option="beta_dependence",
        requires_family="powerlaw_q",
        prior_updates={"beta_q_m1_slope": _u(-0.3, 0.3)},
        description="allow the q power-law slope to vary linearly with m1",
    ),
    MutationSpec(
        "pairing.beta.logistic_m1",
        "pairing",
        "set_option",
        "logistic_m1",
        option="beta_dependence",
        requires_family="powerlaw_q",
        prior_updates={
            "delta_beta_q": _u(-10.0, 10.0),
            "beta_q_m1_transition": _u(10.0, 80.0),
            "beta_q_m1_width": _lu(1.0, 30.0),
        },
        description="allow a logistic transition in q pairing versus m1",
    ),
    MutationSpec(
        "chieff.mean.linear_m1",
        "chieff",
        "set_option",
        "linear_m1",
        option="mean_dependence",
        requires_family="truncated_gaussian",
        prior_updates={"chi_mu_m1_slope": _u(-0.02, 0.02)},
        description="allow chi_eff mean to vary linearly with m1",
    ),
    MutationSpec(
        "chieff.mean.linear_q",
        "chieff",
        "set_option",
        "linear_q",
        option="mean_dependence",
        requires_family="truncated_gaussian",
        prior_updates={"chi_mu_q_slope": _u(-0.6, 0.6)},
        description="allow chi_eff mean to vary linearly with q",
    ),
    MutationSpec(
        "chieff.mean.linear_z",
        "chieff",
        "set_option",
        "linear_z",
        option="mean_dependence",
        requires_family="truncated_gaussian",
        prior_updates={"chi_mu_z_slope": _u(-0.4, 0.4)},
        description="allow chi_eff mean to vary linearly with redshift",
    ),
    MutationSpec(
        "chieff.width.linear_m1",
        "chieff",
        "set_option",
        "linear_m1",
        option="width_dependence",
        requires_family="truncated_gaussian",
        prior_updates={"log_chi_sigma_m1_slope": _u(-0.05, 0.05)},
        description="allow log chi_eff width to vary linearly with m1",
    ),
    MutationSpec(
        "chieff.width.linear_q",
        "chieff",
        "set_option",
        "linear_q",
        option="width_dependence",
        requires_family="truncated_gaussian",
        prior_updates={"log_chi_sigma_q_slope": _u(-2.0, 2.0)},
        description="allow log chi_eff width to vary linearly with q",
    ),
    MutationSpec(
        "chieff.width.linear_z",
        "chieff",
        "set_option",
        "linear_z",
        option="width_dependence",
        requires_family="truncated_gaussian",
        prior_updates={"log_chi_sigma_z_slope": _u(-1.0, 1.0)},
        description="allow log chi_eff width to vary linearly with redshift",
    ),
    MutationSpec(
        "chieff.family.gaussian_mixture",
        "chieff",
        "change_family",
        "gaussian_mixture",
        requires_family="truncated_gaussian",
        prior_removals=(
            "chi_mu",
            "chi_sigma",
            "chi_mu_m1_slope",
            "chi_mu_q_slope",
            "chi_mu_z_slope",
            "log_chi_sigma_m1_slope",
            "log_chi_sigma_q_slope",
            "log_chi_sigma_z_slope",
        ),
        prior_updates={
            "chi_mu_1": _u(-0.5, 0.5),
            "chi_sigma_1": _lu(0.02, 0.5),
            "chi_mu_2": _u(-0.5, 0.5),
            "chi_sigma_2": _lu(0.02, 0.5),
            "chi_fraction": _u(0.0, 1.0),
        },
        description="split chi_eff into two truncated-Gaussian components",
    ),
    MutationSpec(
        "redshift.family.madau_dickinson",
        "redshift",
        "change_family",
        "madau_dickinson",
        requires_family="powerlaw",
        prior_removals=("kappa", "kappa_m1_slope"),
        prior_updates={
            "rate_a": _u(0.0, 10.0),
            "rate_b": _u(0.0, 10.0),
            "rate_z_turnover": _u(0.1, 4.0),
        },
        description="replace power-law rate evolution with Madau-Dickinson form",
    ),
)
