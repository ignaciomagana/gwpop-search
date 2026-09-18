"""Conventional BBH baseline evaluated in the gwcat chi_eff detector-frame basis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Any

try:
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "baseline population models require JAX: install gwpop-search[inference]"
    ) from exc

from .components import (
    chi_eff_logpdf,
    mass_ratio_logpdf,
    primary_mass_powerlaw_peak_logpdf,
    redshift_rate_logpdf,
)
from .cosmology import FlatLambdaCDM


@dataclass(frozen=True)
class GwcatChiEffBBHModel:
    """Power-law+peak, q-power-law, rate-evolution, truncated-chi_eff baseline.

    The source density is normalized in (m1_source, q, z, chi_eff). This class
    explicitly transforms it to the gwcat detector-frame measure
    (m1_detector, q, dL, dOmega, chi_eff):

        p_det = p_src / [(1+z) * ddL/dz] * 1/(4*pi).

    z is inferred from dL using the fixed cosmology context; source mass is then
    m1_detector/(1+z). No advisory source-frame columns are required.
    """

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
        if self.zmax <= 0 or self.zmax >= self.cosmology.interpolation_z_max:
            raise ValueError("zmax must be positive and inside the cosmology interpolation grid")
        if not 0 < self.q_floor < 1:
            raise ValueError("q_floor must lie between 0 and 1")

    def __call__(self, samples: Mapping[str, Any], hyperparameters: Mapping[str, Any]):
        m1det = jnp.asarray(samples["m1_detector"])
        q = jnp.asarray(samples["q"])
        dL = jnp.asarray(samples["luminosity_distance"])
        ra = jnp.asarray(samples["ra"])
        dec = jnp.asarray(samples["dec"])
        chi = jnp.asarray(samples["chi_eff"])

        z = self.cosmology.z_of_dL(dL)
        m1src = m1det / (1.0 + z)

        logp = primary_mass_powerlaw_peak_logpdf(
            m1src,
            alpha=hyperparameters["alpha"],
            mmin=hyperparameters["mmin"],
            mmax=hyperparameters["mmax"],
            peak_fraction=hyperparameters["peak_fraction"],
            peak_mu=hyperparameters["peak_mu"],
            peak_sigma=hyperparameters["peak_sigma"],
        )
        logp = logp + mass_ratio_logpdf(
            q,
            m1src,
            beta=hyperparameters["beta_q"],
            mmin=hyperparameters["mmin"],
            q_floor=self.q_floor,
        )
        logp = logp + redshift_rate_logpdf(
            z,
            kappa=hyperparameters["kappa"],
            zmax=self.zmax,
            cosmology=self.cosmology,
            quadrature_order=self.redshift_quadrature_order,
        )
        logp = logp + chi_eff_logpdf(
            chi,
            mu=hyperparameters["chi_mu"],
            sigma=hyperparameters["chi_sigma"],
        )

        # Change of variables (m1_source,z) -> (m1_detector,dL).
        logp = logp - jnp.log1p(z) - jnp.log(self.cosmology.ddL_dz(z))
        # Density is with respect to dOmega, matching the gwcat basis contract.
        logp = logp - jnp.log(4.0 * jnp.pi)

        valid_sky = (
            (ra >= 0.0)
            & (ra <= 2.0 * jnp.pi)
            & (dec >= -0.5 * jnp.pi)
            & (dec <= 0.5 * jnp.pi)
        )
        valid_transform = jnp.isfinite(z) & (z > 0.0) & (z <= self.zmax)
        return jnp.where(valid_sky & valid_transform, logp, -jnp.inf)


DEFAULT_BASELINE_HYPERPARAMETERS = {
    "alpha": 3.0,
    "mmin": 5.0,
    "mmax": 90.0,
    "peak_fraction": 0.10,
    "peak_mu": 35.0,
    "peak_sigma": 4.0,
    "beta_q": 1.0,
    "kappa": 2.0,
    "chi_mu": 0.05,
    "chi_sigma": 0.20,
}
