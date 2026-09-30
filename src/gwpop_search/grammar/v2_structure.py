"""Structure of the v2 (GWTC-5 atom search) model families.

The v2 families port the gate-verified popres/LVK population models (see
``gwpop_search.models.components``, section "v2"). This module holds only the
*grammar* side: family definitions, the exact hyperparameter set every v2
model must declare, and the model-wide ``support`` context. It imports nothing
but the schema so that the registry can validate v2 specs without importing
JAX.

A v2 model is a :class:`~gwpop_search.grammar.schema.ModelSpec` whose blocks all
use v2 families and whose ``support`` mapping is complete. v1 and v2 families
never mix inside one model.

Mass families (``lam_u_<c>`` are Dirichlet(1, ..., 1) weight coordinates, see
below):

======================  ====================================================
``bp2p``                broken power law + peaks p10, p35 (LVK GWTC-4/5 default)
``bp1p_low``            broken power law + p10            (atom M1)
``bp1p_high``           broken power law + p35            (atom M2)
``bp3p``                broken power law + p10, p35, p3   (atoms M3, M4)
``pl2p``                single power law + p10, p35       (atom M5)
======================  ====================================================

Mixture weights. ``lam_u_<c> ~ U(0, 1)`` independently for every mass component
``c`` (``pl`` for the continuum, then the peaks); the weights are
``lam_c = E_c / sum_k E_k`` with ``E_c = -ln(1 - lam_u_c)`` (independent unit
exponentials), which is exactly Dirichlet(1, ..., 1). The overall scale
``sum_k E_k`` is a redundant, likelihood-free direction. The coordinates are
chosen so that dropping component ``c`` is the single boundary null
``lam_u_c = 0`` with every other prior unchanged, which makes the mass
add/drop-a-peak edges exactly nested for the Savage-Dickey check.

Secondary floor. ``mlow_2 = mmin + mlow_2_frac * (mlow_1 - mmin)`` with
``mlow_2_frac ~ U(0, 1)``: exactly ``mlow_2 | mlow_1 ~ U(mmin, mlow_1)`` (GWTC-5
Table 5 with ``mmin = 3``).
"""

from __future__ import annotations

import math
from typing import Mapping

from .schema import JsonValue, ModelSpec

#: Mass family -> (continuum kind, peak labels). Components are ("pl",) + peaks.
V2_MASS_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "bp2p": ("broken", ("p10", "p35")),
    "bp1p_low": ("broken", ("p10",)),
    "bp1p_high": ("broken", ("p35",)),
    "bp3p": ("broken", ("p10", "p35", "p3")),
    "pl2p": ("single", ("p10", "p35")),
}
V2_PAIRING_FAMILIES = ("tapered_powerlaw_q",)
V2_CHIEFF_FAMILIES = (
    "linear_gaussian",
    "linear_gaussian_mixture",
    "linear_skew_normal",
    "linear_student_t",
)
V2_REDSHIFT_FAMILIES = ("powerlaw_1pz", "madau_dickinson_psi")
V2_MIXTURE_FAMILIES = ("single",)

#: chi_eff correlation switches (each its own structural axis) -> slope parameter.
#: Mean and ln-width are linear in (q - q_pivot), (z - z_pivot) and
#: ln(m1 / m1_pivot) (the LVK "linear correlation" form; m1 in log).
CHIEFF_CORRELATION_OPTIONS: dict[str, str] = {
    "mean_q": "chi_mu_q_slope",
    "mean_z": "chi_mu_z_slope",
    "mean_log_m1": "chi_mu_log_m1_slope",
    "log_sigma_q": "chi_log_sigma_q_slope",
    "log_sigma_z": "chi_log_sigma_z_slope",
    "log_sigma_log_m1": "chi_log_sigma_log_m1_slope",
}
CHIEFF_CORRELATION_COVARIATE = {
    "mean_q": "q",
    "mean_z": "z",
    "mean_log_m1": "log_m1",
    "log_sigma_q": "q",
    "log_sigma_z": "z",
    "log_sigma_log_m1": "log_m1",
}
_CORRELATION_DEFAULTS: dict[str, JsonValue] = {
    **{name: "constant" for name in CHIEFF_CORRELATION_OPTIONS},
    # LVK linear form: intercepts at q = 1 and z = 0 (operator spec
    # 2026-09-30), m1 pivot 30 Msun in ln m1.
    "q_pivot": 1.0,
    "z_pivot": 0.0,
    "m1_pivot": 30.0,
}
_CORRELATION_CHOICES = {name: ("constant", "linear") for name in CHIEFF_CORRELATION_OPTIONS}

#: Required keys of ``ModelSpec.support`` for a v2 model.
V2_SUPPORT_KEYS = (
    "cosmology",
    "m1_grid",
    "mmax",
    "mmin",
    "n_m1",
    "n_q",
    "q_floor",
    "redshift_quadrature_order",
    "sky",
    "zmax",
)


def v2_family_definitions():
    """FamilyDefinition entries for the v2 families (registered by the registry)."""
    from .registry import FamilyDefinition

    defs = [FamilyDefinition("mass", family) for family in V2_MASS_FAMILIES]
    defs.append(
        FamilyDefinition(
            "pairing",
            "tapered_powerlaw_q",
            {"beta_dependence": "constant"},
            {"beta_dependence": ("constant", "logistic_log_m1", "per_mass_component")},
        )
    )
    defs.append(
        FamilyDefinition("chieff", "linear_gaussian", dict(_CORRELATION_DEFAULTS), dict(_CORRELATION_CHOICES))
    )
    defs.append(
        FamilyDefinition(
            "chieff",
            "linear_gaussian_mixture",
            {**_CORRELATION_DEFAULTS, "fraction_dependence": "constant"},
            {**_CORRELATION_CHOICES, "fraction_dependence": ("constant", "logistic_log_m1")},
        )
    )
    defs.append(
        FamilyDefinition("chieff", "linear_skew_normal", dict(_CORRELATION_DEFAULTS), dict(_CORRELATION_CHOICES))
    )
    defs.append(
        FamilyDefinition("chieff", "linear_student_t", dict(_CORRELATION_DEFAULTS), dict(_CORRELATION_CHOICES))
    )
    defs.append(
        FamilyDefinition(
            "redshift",
            "powerlaw_1pz",
            {"kappa_dependence": "constant", "m1_pivot": 30.0},
            {"kappa_dependence": ("constant", "linear_log_m1")},
        )
    )
    defs.append(FamilyDefinition("redshift", "madau_dickinson_psi"))
    return tuple(defs)


def _v2_family_sets():
    return {
        "mass": set(V2_MASS_FAMILIES),
        "pairing": set(V2_PAIRING_FAMILIES),
        "chieff": set(V2_CHIEFF_FAMILIES),
        "redshift": set(V2_REDSHIFT_FAMILIES),
    }


def is_v2_model(model: ModelSpec) -> bool:
    """True when any block uses a v2 family (then all must; see ``validate_v2_model``)."""
    sets = _v2_family_sets()
    return any(getattr(model, block).family in families for block, families in sets.items())


def mass_components(model: ModelSpec) -> tuple[str, ...]:
    _, peaks = V2_MASS_FAMILIES[model.mass.family]
    return ("pl",) + peaks


def required_hyperparameters(model: ModelSpec) -> tuple[str, ...]:
    """The exact hyperparameter set of a v2 model (sorted)."""
    names: set[str] = {"mlow_1", "delta_m_1"}
    continuum, peaks = V2_MASS_FAMILIES[model.mass.family]
    if continuum == "broken":
        names |= {"alpha_1", "alpha_2", "m_break"}
    else:
        names.add("alpha")
    for peak in peaks:
        names |= {f"mu_{peak}", f"sigma_{peak}"}
    components = mass_components(model)
    names |= {f"lam_u_{c}" for c in components}

    names |= {"mlow_2_frac", "delta_m_2"}
    beta = model.pairing.options["beta_dependence"]
    if beta == "constant":
        names.add("beta")
    elif beta == "logistic_log_m1":
        names |= {"beta_low", "beta_high", "beta_m_t", "beta_width"}
    else:  # per_mass_component
        names |= {f"beta_{c}" for c in components}

    chi = model.chieff
    names |= {"chi_mu", "chi_log_sigma"}
    for option, slope in CHIEFF_CORRELATION_OPTIONS.items():
        if chi.options[option] == "linear":
            names.add(slope)
    if chi.family == "linear_gaussian_mixture":
        names |= {"chi_mu_2_frac", "chi_log_sigma_2"}
        if chi.options["fraction_dependence"] == "constant":
            names.add("chi_fraction")
        else:
            names |= {"chi_fraction_low", "chi_fraction_high", "chi_fraction_m_t", "chi_fraction_width"}
    elif chi.family == "linear_skew_normal":
        names.add("chi_eps")
    elif chi.family == "linear_student_t":
        names.add("chi_nu")

    if model.redshift.family == "powerlaw_1pz":
        names.add("kappa")
        if model.redshift.options["kappa_dependence"] == "linear_log_m1":
            names.add("kappa_log_m1_slope")
    else:
        names |= {"md_gamma", "md_kappa", "md_z_peak"}
    return tuple(sorted(names))


def _number(support: Mapping[str, JsonValue], key: str) -> float:
    value = support[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"v2 support {key!r} must be a finite number; got {value!r}")
    return float(value)


def validate_v2_support(support: Mapping[str, JsonValue]) -> None:
    missing = [key for key in V2_SUPPORT_KEYS if key not in support]
    if missing:
        raise ValueError(f"v2 model support is missing {missing} (v2 has no hidden support defaults)")
    unknown = sorted(set(support) - set(V2_SUPPORT_KEYS))
    if unknown:
        raise ValueError(f"unknown v2 support key(s) {unknown}")
    zmax = _number(support, "zmax")
    q_floor = _number(support, "q_floor")
    mmin = _number(support, "mmin")
    mmax = _number(support, "mmax")
    if not 0.0 < zmax < 5.0:
        raise ValueError("v2 support zmax must lie in (0, 5)")
    if not 0.0 < q_floor < 1.0:
        raise ValueError("v2 support q_floor must lie in (0, 1)")
    if not 0.0 < mmin < mmax:
        raise ValueError("v2 support needs 0 < mmin < mmax")
    for key in ("n_m1", "n_q", "redshift_quadrature_order"):
        value = support[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 16:
            raise ValueError(f"v2 support {key!r} must be an integer >= 16")
    if support["m1_grid"] not in ("geomspace", "linspace"):
        raise ValueError("v2 support m1_grid must be 'geomspace' (LVK) or 'linspace'")
    if support["sky"] not in ("marginalized", "isotropic"):
        raise ValueError("v2 support sky must be 'marginalized' or 'isotropic'")
    cosmology = support["cosmology"]
    if not isinstance(cosmology, Mapping) or set(cosmology) != {"H0", "Om0"}:
        raise ValueError("v2 support cosmology must be {'H0': ..., 'Om0': ...}")


def _prior_bounds(prior) -> tuple[float, float]:
    params = dict(prior.parameters)
    if prior.family in ("uniform", "log_uniform"):
        return float(params["low"]), float(params["high"])
    return -math.inf, math.inf


def validate_v2_model(model: ModelSpec) -> None:
    """Family consistency, the complete support context and the exact prior set."""
    sets = _v2_family_sets()
    for block, families in sets.items():
        family = getattr(model, block).family
        if family not in families:
            raise ValueError(
                f"v2 model mixes a v1 family: {block}.{family} (v2 families: {sorted(families)})"
            )
    if model.mixture.family not in V2_MIXTURE_FAMILIES:
        raise ValueError("v2 models use mixture.single")
    validate_v2_support(model.support)
    required = set(required_hyperparameters(model))
    declared = set(model.priors)
    if required != declared:
        raise ValueError(
            "v2 model priors must equal its hyperparameters exactly: "
            f"missing={sorted(required - declared)} unused={sorted(declared - required)}"
        )
    mmin = float(model.support["mmin"])
    mmax = float(model.support["mmax"])
    lo, hi = _prior_bounds(model.priors["mlow_1"])
    if lo < mmin or hi >= mmax:
        raise ValueError(f"mlow_1 prior [{lo}, {hi}] must lie inside [mmin={mmin}, mmax={mmax})")
    for name in ("lam_u_" + c for c in mass_components(model)):
        lo, hi = _prior_bounds(model.priors[name])
        if lo < 0.0 or hi > 1.0:
            raise ValueError(f"{name} is a unit-interval weight coordinate; prior must lie in [0, 1]")
    lo, hi = _prior_bounds(model.priors["mlow_2_frac"])
    if lo < 0.0 or hi > 1.0:
        raise ValueError("mlow_2_frac prior must lie in [0, 1]")
    if model.chieff.family == "linear_gaussian_mixture":
        lo, hi = _prior_bounds(model.priors["chi_mu_2_frac"])
        if lo < 0.0 or hi > 1.0:
            raise ValueError("chi_mu_2_frac prior must lie in [0, 1]")
