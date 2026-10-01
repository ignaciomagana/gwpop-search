"""Typed one-axis population-model mutations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .registry import ComponentRegistry, DEFAULT_COMPONENT_REGISTRY
from .schema import JsonValue, ModelSpec, PriorConfig, structural_diff_axes


class InapplicableMutation(ValueError):
    """Raised when a legal mutation does not apply to the current parent model."""


@dataclass(frozen=True)
class ConditionalPriors:
    """Extra prior changes applied only when the *parent* has ``block.options[option] == value``
    (or, with ``option=None``, ``block.family == value``).

    Used when an atom on one block owns hyperparameters of another block, e.g.
    the v2 mass add/drop-a-peak atoms on a root whose pairing slope is per mass
    component: the pairing slope ``beta_<c>`` of the added/dropped component
    must be added/removed together with the component (one structural axis);
    and the per-component pairing atom adds one slope per component of the
    parent's mass family.
    """

    block: str
    option: str | None
    value: JsonValue
    prior_updates: Mapping[str, PriorConfig] = field(default_factory=dict)
    prior_removals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior_removals", tuple(self.prior_removals))

    def applies(self, parent: ModelSpec) -> bool:
        block = getattr(parent, self.block)
        if self.option is None:
            return block.family == self.value
        return block.options.get(self.option) == self.value


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
    #: Auxiliary options (e.g. a pivot) written together with ``option`` when the
    #: parent block does not already carry them. Like the default options of a
    #: family change they are part of the one atomic axis, not extra mutations.
    #: Empty for every DEFAULT_MUTATIONS entry.
    companion_options: Mapping[str, JsonValue] = field(default_factory=dict)
    #: ``change_family`` only: options written over the new family's defaults
    #: (e.g. the v2 mass-dependent mixture fraction). Part of the one family axis.
    family_options: Mapping[str, JsonValue] = field(default_factory=dict)
    #: ``change_family`` only: parent-block options copied into the new block
    #: when the new family declares them (e.g. the v2 chi_eff correlation
    #: switches, so a spin-shape change keeps an already-added correlation and
    #: two-atom compositions commute). Their priors are inherited unchanged.
    carry_options: tuple[str, ...] = ()
    #: The mutation applies only if the block family is one of these (empty: no
    #: restriction beyond ``requires_family``). Used by the v2 correlation atoms,
    #: which apply to every v2 chi_eff family.
    requires_any_family: tuple[str, ...] = ()
    #: The mutation applies only if the parent block carries these option
    #: values (e.g. the v2 pairing-slope step P1 is defined relative to a
    #: constant slope; Madau-Dickinson Z1 relative to a constant kappa).
    requires_options: Mapping[str, JsonValue] = field(default_factory=dict)
    #: Cross-block prior changes conditioned on the parent's option state.
    conditional_priors: tuple[ConditionalPriors, ...] = ()

    def __post_init__(self) -> None:
        if not self.mutation_id:
            raise ValueError("mutation_id cannot be empty")
        if self.operation not in {"change_family", "set_option"}:
            raise ValueError(f"unsupported mutation operation {self.operation!r}")
        if self.operation == "set_option" and not self.option:
            raise ValueError("set_option mutations require an option name")
        if self.operation == "change_family" and self.option is not None:
            raise ValueError("change_family mutations cannot name an option")
        if self.companion_options and self.operation != "set_option":
            raise ValueError("companion_options require a set_option mutation")
        if self.option is not None and self.option in self.companion_options:
            raise ValueError("companion_options cannot repeat the mutated option")
        if (self.family_options or self.carry_options) and self.operation != "change_family":
            raise ValueError("family_options/carry_options require a change_family mutation")
        if set(self.family_options) & set(self.carry_options):
            raise ValueError("an option cannot be both set and carried")
        object.__setattr__(self, "carry_options", tuple(self.carry_options))
        object.__setattr__(self, "requires_any_family", tuple(self.requires_any_family))
        object.__setattr__(self, "conditional_priors", tuple(self.conditional_priors))

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
    if mutation.requires_any_family and block.family not in mutation.requires_any_family:
        raise InapplicableMutation(
            f"{mutation.mutation_id} requires {mutation.block} family in "
            f"{list(mutation.requires_any_family)}, found {block.family!r}"
        )
    for key, value in mutation.requires_options.items():
        if block.options.get(key) != value:
            raise InapplicableMutation(
                f"{mutation.mutation_id} requires {mutation.block}.{key}={value!r}, "
                f"found {block.options.get(key)!r}"
            )

    if mutation.operation == "change_family":
        family = str(mutation.value)
        if block.family == family:
            raise InapplicableMutation(
                f"{mutation.mutation_id} would leave family unchanged"
            )
        child_block = registry.block(mutation.block, family)
        if mutation.family_options or mutation.carry_options:
            definition = registry.definition(mutation.block, family)
            declared = (
                set(definition.default_options)
                | set(definition.option_choices)
                | set(definition.optional_options)
            )
            options = dict(child_block.options)
            for key in mutation.carry_options:
                if key in block.options and key in declared:
                    options[str(key)] = block.options[key]
            options.update({str(k): v for k, v in mutation.family_options.items()})
            child_block = type(child_block)(family=family, options=options)
    else:
        options = dict(block.options)
        if options.get(mutation.option) == mutation.value:
            raise InapplicableMutation(
                f"{mutation.mutation_id} would leave option unchanged"
            )
        options[str(mutation.option)] = mutation.value
        for key, value in mutation.companion_options.items():
            options.setdefault(str(key), value)
        child_block = type(block)(family=block.family, options=options)

    priors = dict(parent.priors)
    for name in mutation.prior_removals:
        priors.pop(name, None)
    priors.update(mutation.prior_updates)
    for rule in mutation.conditional_priors:
        if rule.applies(parent):
            for name in rule.prior_removals:
                priors.pop(name, None)
            priors.update(rule.prior_updates)

    child = replace(parent, **{mutation.block: child_block, "priors": priors})
    registry.validate_model(child)

    axes = structural_diff_axes(parent, child)
    if mutation.companion_options:
        companions = {
            f"{mutation.block}.options.{key}" for key in mutation.companion_options
        }
        axes = tuple(axis for axis in axes if axis not in companions)
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


# ---------------------------------------------------------------------------
# Follow-up chi_eff atoms (exploratory depth-2 follow-up of the GWTC-5 search).
#
# Deliberately NOT part of DEFAULT_MUTATIONS: adding them there would change the
# enumerated graph (and every campaign keyed on it). Pass them explicitly, e.g.
# ``enumerate_model_graph(root, mutations=DEFAULT_MUTATIONS + FOLLOWUP_MUTATIONS)``
# or ``apply_mutation(parent, FOLLOWUP_MUTATION_TABLE[...])``.
# ---------------------------------------------------------------------------

_CHIEFF_SINGLE_EXTENSION_PRIORS = (
    "chi_mu_m1_slope",
    "chi_mu_q_slope",
    "chi_mu_z_slope",
    "log_chi_sigma_m1_slope",
    "log_chi_sigma_q_slope",
    "log_chi_sigma_z_slope",
    "delta_log_chi_sigma",
    "q_transition",
    "q_transition_width",
)

FOLLOWUP_MUTATIONS: tuple[MutationSpec, ...] = (
    MutationSpec(
        "chieff.family.student_t",
        "chieff",
        "change_family",
        "truncated_student_t",
        requires_family="truncated_gaussian",
        # chi_mu and chi_sigma are inherited from the parent (the gwtc5-v1 root
        # has chi_mu ~ U(-0.3, 0.3), chi_sigma ~ LU(0.03, 0.5)).
        prior_removals=_CHIEFF_SINGLE_EXTENSION_PRIORS,
        prior_updates={"chi_nu": _lu(1.0, 100.0)},
        description=(
            "replace the truncated-Gaussian chi_eff with a heavy-tailed truncated "
            "Student-t (Gaussian recovered as chi_nu -> infinity)"
        ),
    ),
    MutationSpec(
        "chieff.fraction.logistic_q",
        "chieff",
        "set_option",
        "logistic_q",
        option="fraction_dependence",
        requires_family="gaussian_mixture",
        companion_options={"q_pivot": 0.7},
        prior_removals=("chi_fraction",),
        # Components are no longer exchangeable (the weight depends on q), so the
        # label symmetry is broken by the priors: component 1 is the bulk,
        # component 2 the spinning component.
        prior_updates={
            "chi_mu_1": _u(-0.5, 0.5),
            "chi_sigma_1": _lu(0.02, 0.5),
            "chi_mu_2": _u(0.0, 1.0),
            "chi_sigma_2": _lu(0.02, 0.5),
            "logit_chi_fraction": _u(-8.0, 2.0),
            "chi_fraction_q_slope": _u(-10.0, 10.0),
        },
        description=(
            "let the weight of chi_eff mixture component 2 vary logistically with q"
        ),
    ),
    MutationSpec(
        "chieff.width.logistic_q",
        "chieff",
        "set_option",
        "logistic_q",
        option="width_dependence",
        requires_family="truncated_gaussian",
        prior_removals=(
            "log_chi_sigma_m1_slope",
            "log_chi_sigma_q_slope",
            "log_chi_sigma_z_slope",
        ),
        prior_updates={
            "delta_log_chi_sigma": _u(-1.0, 4.0),
            "q_transition": _u(0.1, 1.0),
            "q_transition_width": _lu(0.01, 0.3),
        },
        description="allow a logistic step in log chi_eff width versus q",
    ),
)

FOLLOWUP_MUTATION_TABLE: dict[str, MutationSpec] = {
    item.mutation_id: item for item in FOLLOWUP_MUTATIONS
}
