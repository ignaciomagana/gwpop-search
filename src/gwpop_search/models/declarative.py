"""Compile declarative Phase-4 model specs into gwcat-basis JAX densities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "declarative population models require JAX: install gwpop-search[inference]"
    ) from exc

from gwpop_search.grammar import (
    DEFAULT_COMPONENT_REGISTRY,
    ModelSpec,
)

from gwpop_search.grammar.v2_structure import (
    CHIEFF_CORRELATION_COVARIATE,
    CHIEFF_CORRELATION_OPTIONS,
    V2_MASS_FAMILIES,
    is_v2_model,
    mass_components,
)

from .components import (
    LOG_4PI,
    BrokenPowerLawPeaksMass,
    LVKMassGrid,
    PowerLawRedshiftNormTable,
    TaperedPowerLawPairing,
    log_eps_skewnorm,
    log_mix,
    redshift_madau_dickinson_psi_logpdf,
    truncated_normal_logpdf,
    truncated_student_t_logpdf,
    chi_eff_logistic_mixture_logpdf,
    chi_eff_logpdf,
    chi_eff_mixture_logpdf,
    chi_eff_student_t_logpdf,
    mass_ratio_logpdf,
    mass_ratio_truncated_normal_logpdf,
    powerlaw_logpdf,
    primary_mass_broken_powerlaw_logpdf,
    primary_mass_powerlaw_peak_logpdf,
    primary_mass_powerlaw_two_peak_logpdf,
    redshift_madau_dickinson_logpdf,
    redshift_rate_logpdf,
)
from .cosmology import FlatLambdaCDM


def _mass_logpdf(spec, m1, hp):
    family = spec.family
    if family == "powerlaw":
        return powerlaw_logpdf(
            m1,
            alpha=hp["alpha"],
            xmin=hp["mmin"],
            xmax=hp["mmax"],
        )
    if family == "pl_peak":
        return primary_mass_powerlaw_peak_logpdf(
            m1,
            alpha=hp["alpha"],
            mmin=hp["mmin"],
            mmax=hp["mmax"],
            peak_fraction=hp["peak_fraction"],
            peak_mu=hp["peak_mu"],
            peak_sigma=hp["peak_sigma"],
        )
    if family == "broken_powerlaw":
        return primary_mass_broken_powerlaw_logpdf(
            m1,
            alpha1=hp["alpha1"],
            alpha2=hp["alpha2"],
            mmin=hp["mmin"],
            mmax=hp["mmax"],
            break_fraction=hp["break_fraction"],
        )
    if family == "pl_two_peak":
        return primary_mass_powerlaw_two_peak_logpdf(
            m1,
            alpha=hp["alpha"],
            mmin=hp["mmin"],
            mmax=hp["mmax"],
            peak1_fraction=hp["peak1_fraction"],
            peak1_mu=hp["peak1_mu"],
            peak1_sigma=hp["peak1_sigma"],
            peak2_fraction=hp["peak2_fraction"],
            peak2_mu=hp["peak2_mu"],
            peak2_sigma=hp["peak2_sigma"],
        )
    raise ValueError(f"unsupported mass family {family!r}")


def _pairing_logpdf(spec, q, m1, hp, q_floor):
    family = spec.family
    if family == "truncated_gaussian_q":
        return mass_ratio_truncated_normal_logpdf(
            q,
            m1,
            mu=hp["q_mu"],
            sigma=hp["q_sigma"],
            mmin=hp["mmin"],
            q_floor=q_floor,
        )
    if family != "powerlaw_q":
        raise ValueError(f"unsupported pairing family {family!r}")

    dependence = spec.options["beta_dependence"]
    beta = hp["beta_q"]
    pivot = float(spec.options["m1_pivot"])
    if dependence == "linear_m1":
        beta = beta + hp["beta_q_m1_slope"] * (m1 - pivot)
    elif dependence == "logistic_m1":
        width = hp["beta_q_m1_width"]
        argument = (m1 - hp["beta_q_m1_transition"]) / width
        beta = beta + hp["delta_beta_q"] / (1.0 + jnp.exp(-argument))
    elif dependence != "constant":
        raise ValueError(f"unsupported beta dependence {dependence!r}")

    return mass_ratio_logpdf(
        q,
        m1,
        beta=beta,
        mmin=hp["mmin"],
        q_floor=q_floor,
    )


def _chieff_logpdf(spec, chi, m1, q, z, hp):
    if spec.family == "gaussian_mixture":
        fraction_dep = spec.options["fraction_dependence"]
        if fraction_dep == "logistic_q":
            # Weight of component 2: sigmoid(logit + slope * (q - q_pivot)).
            logit = hp["logit_chi_fraction"] + hp["chi_fraction_q_slope"] * (
                q - float(spec.options["q_pivot"])
            )
            return chi_eff_logistic_mixture_logpdf(
                chi,
                mu1=hp["chi_mu_1"],
                sigma1=hp["chi_sigma_1"],
                mu2=hp["chi_mu_2"],
                sigma2=hp["chi_sigma_2"],
                fraction_logit=logit,
            )
        if fraction_dep != "constant":
            raise ValueError(
                f"unsupported chi_eff mixture fraction dependence {fraction_dep!r}"
            )
        return chi_eff_mixture_logpdf(
            chi,
            mu1=hp["chi_mu_1"],
            sigma1=hp["chi_sigma_1"],
            mu2=hp["chi_mu_2"],
            sigma2=hp["chi_sigma_2"],
            fraction=hp["chi_fraction"],
        )
    if spec.family == "truncated_student_t":
        for option in ("mean_dependence", "width_dependence"):
            if spec.options[option] != "constant":
                raise ValueError(
                    f"truncated_student_t supports only constant {option}"
                )
        return chi_eff_student_t_logpdf(
            chi,
            mu=hp["chi_mu"],
            sigma=hp["chi_sigma"],
            nu=hp["chi_nu"],
        )
    if spec.family != "truncated_gaussian":
        raise ValueError(f"unsupported chi_eff family {spec.family!r}")

    mu = hp["chi_mu"]
    mean_dep = spec.options["mean_dependence"]
    if mean_dep == "linear_m1":
        mu = mu + hp["chi_mu_m1_slope"] * (
            m1 - float(spec.options["m1_pivot"])
        )
    elif mean_dep == "linear_q":
        mu = mu + hp["chi_mu_q_slope"] * (
            q - float(spec.options["q_pivot"])
        )
    elif mean_dep == "linear_z":
        mu = mu + hp["chi_mu_z_slope"] * (
            z - float(spec.options["z_pivot"])
        )
    elif mean_dep != "constant":
        raise ValueError(f"unsupported chi_eff mean dependence {mean_dep!r}")

    log_sigma = jnp.log(hp["chi_sigma"])
    width_dep = spec.options["width_dependence"]
    if width_dep == "linear_m1":
        log_sigma = log_sigma + hp["log_chi_sigma_m1_slope"] * (
            m1 - float(spec.options["m1_pivot"])
        )
    elif width_dep == "linear_q":
        log_sigma = log_sigma + hp["log_chi_sigma_q_slope"] * (
            q - float(spec.options["q_pivot"])
        )
    elif width_dep == "linear_z":
        log_sigma = log_sigma + hp["log_chi_sigma_z_slope"] * (
            z - float(spec.options["z_pivot"])
        )
    elif width_dep == "logistic_q":
        # Smooth step in q: log sigma rises by delta_log_chi_sigma below
        # q_transition (broader at low q when the step is positive).
        argument = (hp["q_transition"] - q) / hp["q_transition_width"]
        log_sigma = log_sigma + hp["delta_log_chi_sigma"] * jax.nn.sigmoid(argument)
    elif width_dep != "constant":
        raise ValueError(f"unsupported chi_eff width dependence {width_dep!r}")

    return chi_eff_logpdf(chi, mu=mu, sigma=jnp.exp(log_sigma))


def _redshift_logpdf(spec, z, hp, cosmology, zmax, quadrature_order):
    if spec.family == "powerlaw":
        if spec.options["kappa_dependence"] != "constant":
            raise ValueError("only constant kappa is implemented in the initial graph")
        return redshift_rate_logpdf(
            z,
            kappa=hp["kappa"],
            zmax=zmax,
            cosmology=cosmology,
            quadrature_order=quadrature_order,
        )
    if spec.family == "madau_dickinson":
        return redshift_madau_dickinson_logpdf(
            z,
            a=hp["rate_a"],
            b=hp["rate_b"],
            z_turnover=hp["rate_z_turnover"],
            zmax=zmax,
            cosmology=cosmology,
            quadrature_order=quadrature_order,
        )
    raise ValueError(f"unsupported redshift family {spec.family!r}")


def pairing_logpdf_from_spec(spec, q, m1_source, hyperparameters, *, q_floor=0.05):
    """Evaluate the normalized pairing block from a declarative model spec."""
    return _pairing_logpdf(
        spec,
        q,
        m1_source,
        hyperparameters,
        q_floor,
    )


def chieff_logpdf_from_spec(
    spec,
    chi_eff,
    m1_source,
    q,
    z,
    hyperparameters,
):
    """Evaluate the normalized chi_eff block from a declarative model spec."""
    return _chieff_logpdf(
        spec,
        chi_eff,
        m1_source,
        q,
        z,
        hyperparameters,
    )


# ---------------------------------------------------------------------------
# v2 (GWTC-5 atom search) dispatch
# ---------------------------------------------------------------------------


def v2_physical_hyperparameters(spec: ModelSpec, hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    """Sampled v2 hyperparameters plus the physical quantities they define.

    * ``lam_<c>``, ``log_lam_<c>``: Dirichlet(1, ..., 1) mass weights from the
      unit coordinates ``lam_u_<c>`` (``E_c = -ln(1 - u_c)``, ``lam = E / sum E``);
    * ``mlow_2 = mmin + mlow_2_frac (mlow_1 - mmin)``;
    * ``chi_mu_2 = chi_mu + chi_mu_2_frac (1 - chi_mu)`` for the chi_eff mixture;
    * ``mmax`` (the fixed support ceiling).
    """
    hp = dict(hyperparameters)
    mmin = float(spec.support["mmin"])
    components = mass_components(spec)
    log_e = []
    for c in components:
        u = jnp.asarray(hp[f"lam_u_{c}"])
        # u = 1 (E = inf) is a measure-zero prior edge; keep E finite there
        e = -jnp.log1p(-jnp.clip(u, 0.0, 1.0 - 2.0**-53))
        log_e.append(jnp.where(e > 0, jnp.log(jnp.where(e > 0, e, 1.0)), -jnp.inf))
    log_total = log_mix(log_e)
    for c, le in zip(components, log_e):
        hp[f"log_lam_{c}"] = le - log_total
        hp[f"lam_{c}"] = jnp.exp(le - log_total)
    hp["mlow_2"] = mmin + jnp.asarray(hp["mlow_2_frac"]) * (jnp.asarray(hp["mlow_1"]) - mmin)
    if spec.chieff.family == "linear_gaussian_mixture":
        mu = jnp.asarray(hp["chi_mu"])
        hp["chi_mu_2"] = mu + jnp.asarray(hp["chi_mu_2_frac"]) * (1.0 - mu)
    hp["mmax"] = float(spec.support["mmax"])
    return hp


def _v2_chi_moments(spec, hp, *, q, z, log_m1):
    """chi_eff location and ln width with the active LVK-linear correlations."""
    options = spec.chieff.options
    covariates = {
        "q": q - float(options["q_pivot"]),
        "z": z - float(options["z_pivot"]),
        "log_m1": log_m1 - jnp.log(float(options["m1_pivot"])),
    }
    mu = jnp.asarray(hp["chi_mu"]) + 0.0 * q
    log_sigma = jnp.asarray(hp["chi_log_sigma"]) + 0.0 * q
    for option, slope in CHIEFF_CORRELATION_OPTIONS.items():
        if options[option] != "linear":
            continue
        term = hp[slope] * covariates[CHIEFF_CORRELATION_COVARIATE[option]]
        if option.startswith("mean_"):
            mu = mu + term
        else:
            log_sigma = log_sigma + term
    return mu, log_sigma


def _v2_chieff_logpdf(spec, chi, hp, *, q, z, log_m1):
    family = spec.chieff.family
    mu, log_sigma = _v2_chi_moments(spec, hp, q=q, z=z, log_m1=log_m1)
    sigma = jnp.exp(log_sigma)
    if family == "linear_gaussian":
        return truncated_normal_logpdf(chi, mu=mu, sigma=sigma, low=-1.0, high=1.0)
    if family == "linear_skew_normal":
        return log_eps_skewnorm(chi, mu, sigma, hp["chi_eps"])
    if family == "linear_student_t":
        return truncated_student_t_logpdf(chi, mu=mu, sigma=sigma, nu=hp["chi_nu"], low=-1.0, high=1.0)
    if family != "linear_gaussian_mixture":
        raise ValueError(f"unsupported v2 chi_eff family {family!r}")
    # Component 1 (the bulk) carries the correlations; component 2 has an
    # ordered constant mean mu_2 in [mu_1, 1] and its own width.
    log_p1 = truncated_normal_logpdf(chi, mu=mu, sigma=sigma, low=-1.0, high=1.0)
    log_p2 = truncated_normal_logpdf(chi, mu=hp["chi_mu_2"], sigma=jnp.exp(hp["chi_log_sigma_2"]),
                                     low=-1.0, high=1.0)
    if spec.chieff.options["fraction_dependence"] == "constant":
        f = jnp.asarray(hp["chi_fraction"]) + 0.0 * chi
    else:
        step = jax.nn.sigmoid((log_m1 - jnp.log(hp["chi_fraction_m_t"])) / hp["chi_fraction_width"])
        f = hp["chi_fraction_low"] + (hp["chi_fraction_high"] - hp["chi_fraction_low"]) * step
    valid_f = jnp.isfinite(f) & (f >= 0.0) & (f <= 1.0)
    fc = jnp.clip(f, 0.0, 1.0)
    log_f = jnp.where(fc > 0, jnp.log(jnp.where(fc > 0, fc, 1.0)), -jnp.inf)
    log_1mf = jnp.where(fc < 1, jnp.log1p(-jnp.where(fc < 1, fc, 0.0)), -jnp.inf)
    return jnp.where(valid_f, log_mix([log_1mf + log_p1, log_f + log_p2]), -jnp.inf)


class _V2Compiled:
    """Static (hyperparameter-independent) pieces of a compiled v2 model."""

    def __init__(self, spec: ModelSpec, cosmology: FlatLambdaCDM):
        support = spec.support
        self.grid = LVKMassGrid(
            mmin=float(support["mmin"]), mmax=float(support["mmax"]),
            n_m1=int(support["n_m1"]), n_q=int(support["n_q"]),
            q_floor=float(support["q_floor"]), m1_grid=str(support["m1_grid"]),
        )
        continuum, peaks = V2_MASS_FAMILIES[spec.mass.family]
        self.mass = BrokenPowerLawPeaksMass(continuum, peaks, self.grid)
        self.pairing = TaperedPowerLawPairing(
            str(spec.pairing.options["beta_dependence"]), self.grid, self.mass.components
        )
        self.kappa_table = None
        if (spec.redshift.family == "powerlaw_1pz"
                and spec.redshift.options["kappa_dependence"] == "linear_log_m1"):
            self.kappa_table = PowerLawRedshiftNormTable(
                cosmology, float(support["zmax"]),
                quadrature_order=int(support["redshift_quadrature_order"]),
            )


def _v2_redshift_logpdf(model, z, log_m1, hp):
    spec = model.spec
    if spec.redshift.family == "madau_dickinson_psi":
        return redshift_madau_dickinson_psi_logpdf(
            z, gamma=hp["md_gamma"], kappa=hp["md_kappa"], z_peak=hp["md_z_peak"],
            zmax=model.zmax, cosmology=model.cosmology,
            quadrature_order=model.redshift_quadrature_order,
        )
    if spec.redshift.options["kappa_dependence"] == "constant":
        return redshift_rate_logpdf(
            z, kappa=hp["kappa"], zmax=model.zmax, cosmology=model.cosmology,
            quadrature_order=model.redshift_quadrature_order,
        )
    from .components import redshift_powerlaw_conditional_logpdf

    pivot = float(spec.redshift.options["m1_pivot"])
    kappa = hp["kappa"] + hp["kappa_log_m1_slope"] * (log_m1 - jnp.log(pivot))
    return redshift_powerlaw_conditional_logpdf(
        z, kappa, zmax=model.zmax, cosmology=model.cosmology, norm_table=model._v2.kappa_table
    )


def _v2_mass_pairing(compiled, m1, q, hp):
    m1 = jnp.asarray(m1)
    q = jnp.asarray(q)
    safe_m1 = jnp.where(m1 > 0.0, m1, 1.0)
    safe_q = jnp.where(q > 0.0, q, 1.0)
    log_m1 = jnp.log(safe_m1)
    log_q = jnp.log(safe_q)
    terms = compiled.mass.log_component_terms(safe_m1, log_m1, hp)
    lp_m1 = compiled.mass.log_prob_from_terms(safe_m1, terms, hp)
    lp_q = compiled.pairing.log_prob(safe_q, log_q, safe_m1, log_m1, hp, mass_terms=terms)
    logp = jnp.where(jnp.isneginf(lp_m1), -jnp.inf, lp_m1 + lp_q)
    return jnp.where((m1 > 0.0) & (q > 0.0), logp, -jnp.inf), log_m1


def v2_mass_logpdf(model, m1, q, hyperparameters, *, physical: bool = False):
    """ln p(m1_source, q | Lambda) of a compiled v2 model.

    ``physical=True`` takes the physical parameters (``log_lam_<c>``,
    ``mlow_2``, ...) directly, e.g. an LVK release draw mapped by
    :func:`lvk_default_physical`; otherwise the sampled coordinates.
    """
    hp = dict(hyperparameters) if physical else v2_physical_hyperparameters(model.spec, hyperparameters)
    return _v2_mass_pairing(model._v2, m1, q, hp)[0]


#: LVK GWTC-4/5 default-family (popsummary) names -> v2 physical names.
LVK_DEFAULT_TO_V2 = {
    "alpha_1": "alpha_1", "alpha_2": "alpha_2", "break_mass": "m_break",
    "mlow_1": "mlow_1", "delta_m_1": "delta_m_1", "mlow_2": "mlow_2", "delta_m_2": "delta_m_2",
    "mpp_1": "mu_p10", "sigpp_1": "sigma_p10", "mpp_2": "mu_p35", "sigpp_2": "sigma_p35",
    "beta": "beta", "lamb": "kappa",
}


def lvk_default_physical(params: Mapping[str, Any]) -> dict[str, Any]:
    """Physical v2 parameters of an LVK default-family draw (lam_2 = 1 - lam_0 - lam_1)."""
    out = {v: params[k] for k, v in LVK_DEFAULT_TO_V2.items() if k in params}
    lam = {"pl": params["lam_0"], "p10": params["lam_1"], "p35": 1.0 - params["lam_0"] - params["lam_1"]}
    for c, value in lam.items():
        value = jnp.asarray(value)
        out[f"lam_{c}"] = value
        out[f"log_lam_{c}"] = jnp.where(value > 0, jnp.log(jnp.where(value > 0, value, 1.0)), -jnp.inf)
    return out


def lvk_default_coordinates(params: Mapping[str, Any], *, mmin: float = 3.0) -> dict[str, float]:
    """Sampled v2 root coordinates that reproduce an LVK default-family draw.

    Weights: ``lam_u_c = 1 - exp(-lam_c)`` (E_c = lam_c, sum E = 1); secondary
    floor: ``mlow_2_frac = (mlow_2 - mmin) / (mlow_1 - mmin)``.
    """
    import math

    out = {v: float(params[k]) for k, v in LVK_DEFAULT_TO_V2.items() if k in params}
    out.pop("mlow_2", None)
    lam = {"pl": float(params["lam_0"]), "p10": float(params["lam_1"]),
           "p35": 1.0 - float(params["lam_0"]) - float(params["lam_1"])}
    for c, value in lam.items():
        out[f"lam_u_{c}"] = -math.expm1(-value)
    out["mlow_2_frac"] = (float(params["mlow_2"]) - mmin) / (float(params["mlow_1"]) - mmin)
    return out


def v2_source_frame_logpdf(model, *, m1, q, z, chi_eff, hyperparameters):
    """Normalised v2 source-frame density ln p(m1, q, z, chi_eff | Lambda)."""
    spec = model.spec
    hp = v2_physical_hyperparameters(spec, hyperparameters)
    z = jnp.asarray(z)
    q = jnp.asarray(q)
    logp, log_m1 = _v2_mass_pairing(model._v2, m1, q, hp)
    logp = logp + _v2_redshift_logpdf(model, z, log_m1, hp)
    logp = logp + _v2_chieff_logpdf(spec, jnp.asarray(chi_eff), hp, q=q, z=z, log_m1=log_m1)
    return jnp.where(jnp.isfinite(logp), logp, -jnp.inf)


_V1_LEGACY_DEFAULTS = {"zmax": 2.5, "q_floor": 0.05, "redshift_quadrature_order": 96}


@dataclass(frozen=True)
class DeclarativeGwcatChiEffModel:
    """A validated ModelSpec compiled to the gwcat-v2 chi_eff density basis.

    v2 specs (``grammar.v2_structure``) take ``zmax``, ``q_floor``, the mass
    floor/ceiling, the normalisation grids, the cosmology and the sky convention
    from ``spec.support``; there are no hidden defaults, and an explicit
    constructor value that disagrees with the spec is an error. v1 specs keep
    their historical defaults (zmax 2.5, q_floor 0.05, 96 quadrature nodes,
    FlatLambdaCDM(67.74, 0.3089)) so the frozen v1 record reproduces exactly.
    """

    spec: ModelSpec
    cosmology: FlatLambdaCDM | None = None
    zmax: float | None = None
    q_floor: float | None = None
    redshift_quadrature_order: int | None = None
    _v2: Any = field(default=None, init=False, repr=False, compare=False)

    required_fields = (
        "m1_detector",
        "q",
        "luminosity_distance",
        "ra",
        "dec",
        "chi_eff",
    )

    def __post_init__(self) -> None:
        DEFAULT_COMPONENT_REGISTRY.validate_model(self.spec)
        if self.spec.mixture.family != "single":
            raise ValueError(
                "the initial declarative compiler only supports mixture.single"
            )
        v2 = is_v2_model(self.spec)
        if v2:
            support = self.spec.support
            resolved = {
                "zmax": float(support["zmax"]),
                "q_floor": float(support["q_floor"]),
                "redshift_quadrature_order": int(support["redshift_quadrature_order"]),
            }
            for key, value in resolved.items():
                given = getattr(self, key)
                if given is not None and float(given) != float(value):
                    raise ValueError(
                        f"{key}={given!r} disagrees with the v2 model support {key}={value!r}"
                    )
                object.__setattr__(self, key, value)
            cosmo = FlatLambdaCDM(H0=float(support["cosmology"]["H0"]), Om0=float(support["cosmology"]["Om0"]))
            if self.cosmology is not None and (
                self.cosmology.H0 != cosmo.H0 or self.cosmology.Om0 != cosmo.Om0
            ):
                raise ValueError("cosmology disagrees with the v2 model support cosmology")
            object.__setattr__(self, "cosmology", cosmo if self.cosmology is None else self.cosmology)
            fields = ("m1_detector", "q", "luminosity_distance", "chi_eff")
            if support["sky"] == "isotropic":
                fields = ("m1_detector", "q", "luminosity_distance", "ra", "dec", "chi_eff")
            object.__setattr__(self, "required_fields", fields)
        else:
            for key, value in _V1_LEGACY_DEFAULTS.items():
                if getattr(self, key) is None:
                    object.__setattr__(self, key, value)
            if self.cosmology is None:
                object.__setattr__(self, "cosmology", FlatLambdaCDM())
        if self.zmax <= 0.0 or self.zmax >= self.cosmology.interpolation_z_max:
            raise ValueError(
                "zmax must be positive and inside the cosmology interpolation grid"
            )
        if not 0.0 < self.q_floor < 1.0:
            raise ValueError("q_floor must lie between zero and one")
        if v2:
            object.__setattr__(self, "_v2", _V2Compiled(self.spec, self.cosmology))

    @property
    def is_v2(self) -> bool:
        return self._v2 is not None

    def to_config(self) -> dict[str, object]:
        config = {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_hash": self.spec.model_hash,
            "model_spec": self.spec.to_dict(),
            "cosmology": self.cosmology.to_config(),
            "zmax": float(self.zmax),
            "q_floor": float(self.q_floor),
            "redshift_quadrature_order": int(self.redshift_quadrature_order),
        }
        if self.is_v2:
            config["support"] = dict(self.spec.support)
            config["required_fields"] = list(self.required_fields)
        return config

    def source_frame_logpdf(self, *, m1, q, z, chi_eff, hyperparameters):
        """ln p(m1_source, q, z, chi_eff) (v2 models only)."""
        if not self.is_v2:
            raise ValueError("source_frame_logpdf is defined for v2 models")
        return v2_source_frame_logpdf(self, m1=m1, q=q, z=z, chi_eff=chi_eff,
                                      hyperparameters=hyperparameters)

    def __call__(
        self,
        samples: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ):
        missing = set(self.spec.priors) - set(hyperparameters)
        if missing:
            raise KeyError(
                f"missing model hyperparameter(s): {sorted(missing)}"
            )
        if self.is_v2:
            return self._call_v2(samples, hyperparameters)

        m1det = jnp.asarray(samples["m1_detector"])
        q = jnp.asarray(samples["q"])
        d_l = jnp.asarray(samples["luminosity_distance"])
        ra = jnp.asarray(samples["ra"])
        dec = jnp.asarray(samples["dec"])
        chi = jnp.asarray(samples["chi_eff"])

        z = self.cosmology.z_of_dL(d_l)
        m1src = m1det / (1.0 + z)

        logp = _mass_logpdf(self.spec.mass, m1src, hyperparameters)
        logp = logp + _pairing_logpdf(
            self.spec.pairing,
            q,
            m1src,
            hyperparameters,
            self.q_floor,
        )
        logp = logp + _redshift_logpdf(
            self.spec.redshift,
            z,
            hyperparameters,
            self.cosmology,
            self.zmax,
            self.redshift_quadrature_order,
        )
        logp = logp + _chieff_logpdf(
            self.spec.chieff,
            chi,
            m1src,
            q,
            z,
            hyperparameters,
        )

        logp = (
            logp
            - jnp.log1p(z)
            - jnp.log(self.cosmology.ddL_dz(z))
            - jnp.log(4.0 * jnp.pi)
        )

        valid_sky = (
            (ra >= 0.0)
            & (ra <= 2.0 * jnp.pi)
            & (dec >= -0.5 * jnp.pi)
            & (dec <= 0.5 * jnp.pi)
        )
        valid_transform = jnp.isfinite(z) & (z > 0.0) & (z <= self.zmax)
        return jnp.where(valid_sky & valid_transform, logp, -jnp.inf)

    def _call_v2(self, samples, hyperparameters):
        m1det = jnp.asarray(samples["m1_detector"])
        q = jnp.asarray(samples["q"])
        d_l = jnp.asarray(samples["luminosity_distance"])
        chi = jnp.asarray(samples["chi_eff"])
        z = self.cosmology.z_of_dL(d_l)
        valid = jnp.isfinite(z) & (z > 0.0) & (z <= self.zmax)
        safe_z = jnp.where(valid, z, 0.5 * self.zmax)
        m1src = m1det / (1.0 + safe_z)
        logp = v2_source_frame_logpdf(self, m1=m1src, q=q, z=safe_z, chi_eff=chi,
                                      hyperparameters=hyperparameters)
        # (m1_source, z) -> (m1_detector, d_L)
        logp = logp - jnp.log1p(safe_z) - jnp.log(self.cosmology.ddL_dz(safe_z))
        if self.spec.support["sky"] == "isotropic":
            ra = jnp.asarray(samples["ra"])
            dec = jnp.asarray(samples["dec"])
            valid = valid & (ra >= 0.0) & (ra <= 2.0 * jnp.pi) & (dec >= -0.5 * jnp.pi) & (dec <= 0.5 * jnp.pi)
            logp = logp - LOG_4PI
        return jnp.where(valid, logp, -jnp.inf)


def compile_model_spec(
    spec: ModelSpec,
    **kwargs,
) -> DeclarativeGwcatChiEffModel:
    return DeclarativeGwcatChiEffModel(spec=spec, **kwargs)
