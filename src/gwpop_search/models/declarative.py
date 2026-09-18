"""Compile declarative Phase-4 model specs into gwcat-basis JAX densities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "declarative population models require JAX: install gwpop-search[inference]"
    ) from exc

from gwpop_search.grammar import (
    DEFAULT_COMPONENT_REGISTRY,
    ModelSpec,
)

from .components import (
    chi_eff_logpdf,
    chi_eff_mixture_logpdf,
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
        if spec.options["fraction_dependence"] != "constant":
            raise ValueError("only constant chi_eff mixture fraction is implemented")
        return chi_eff_mixture_logpdf(
            chi,
            mu1=hp["chi_mu_1"],
            sigma1=hp["chi_sigma_1"],
            mu2=hp["chi_mu_2"],
            sigma2=hp["chi_sigma_2"],
            fraction=hp["chi_fraction"],
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


@dataclass(frozen=True)
class DeclarativeGwcatChiEffModel:
    """A validated ModelSpec compiled to the gwcat-v2 chi_eff density basis."""

    spec: ModelSpec
    cosmology: FlatLambdaCDM = field(default_factory=FlatLambdaCDM)
    zmax: float = 2.5
    q_floor: float = 0.05
    redshift_quadrature_order: int = 96

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
        if self.zmax <= 0.0 or self.zmax >= self.cosmology.interpolation_z_max:
            raise ValueError(
                "zmax must be positive and inside the cosmology interpolation grid"
            )
        if not 0.0 < self.q_floor < 1.0:
            raise ValueError("q_floor must lie between zero and one")

    def to_config(self) -> dict[str, object]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "model_hash": self.spec.model_hash,
            "model_spec": self.spec.to_dict(),
            "cosmology": self.cosmology.to_config(),
            "zmax": float(self.zmax),
            "q_floor": float(self.q_floor),
            "redshift_quadrature_order": int(self.redshift_quadrature_order),
        }

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


def compile_model_spec(
    spec: ModelSpec,
    **kwargs,
) -> DeclarativeGwcatChiEffModel:
    return DeclarativeGwcatChiEffModel(spec=spec, **kwargs)
