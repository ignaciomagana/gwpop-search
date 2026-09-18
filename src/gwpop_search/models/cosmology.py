"""Fixed flat-LambdaCDM transforms used by baseline population densities.

Cosmology is intentionally a fixed model context in Phase 3, not an inferred
hyperparameter. The source-to-detector density Jacobian still lives explicitly
in the population model rather than in the HBI engine or data adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - inference extra
    raise ImportError(
        "baseline population models require JAX: install gwpop-search[inference]"
    ) from exc


C_KM_S = 299792.458


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Small JAX-compatible flat-LambdaCDM context with dL -> z interpolation."""

    H0: float = 67.74
    Om0: float = 0.3089
    interpolation_z_max: float = 5.0
    interpolation_size: int = 16384
    _z_grid: np.ndarray = field(init=False, repr=False, compare=False)
    _dL_grid: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.H0) or self.H0 <= 0:
            raise ValueError("H0 must be finite and positive")
        if not np.isfinite(self.Om0) or not 0 < self.Om0 < 1:
            raise ValueError("Om0 must lie strictly between 0 and 1")
        if self.interpolation_z_max <= 0 or self.interpolation_size < 1024:
            raise ValueError("cosmology interpolation grid is too small")

        z = np.linspace(0.0, float(self.interpolation_z_max), int(self.interpolation_size))
        inv_e = 1.0 / np.sqrt(self.Om0 * (1.0 + z) ** 3 + (1.0 - self.Om0))
        dz = np.diff(z)
        increments = 0.5 * (inv_e[1:] + inv_e[:-1]) * dz
        dc = np.concatenate(([0.0], np.cumsum(increments))) * (C_KM_S / self.H0)
        dL = (1.0 + z) * dc
        object.__setattr__(self, "_z_grid", z)
        object.__setattr__(self, "_dL_grid", dL)

    def to_config(self) -> dict[str, float | int]:
        return {
            "H0": float(self.H0),
            "Om0": float(self.Om0),
            "interpolation_z_max": float(self.interpolation_z_max),
            "interpolation_size": int(self.interpolation_size),
        }

    def e(self, z):
        z = jnp.asarray(z)
        return jnp.sqrt(self.Om0 * (1.0 + z) ** 3 + (1.0 - self.Om0))

    def z_of_dL(self, dL):
        """Invert luminosity distance on the fixed precomputed cosmology grid."""
        dL = jnp.asarray(dL)
        z = jnp.interp(dL, jnp.asarray(self._dL_grid), jnp.asarray(self._z_grid))
        valid = (dL >= 0.0) & (dL <= self._dL_grid[-1])
        return jnp.where(valid, z, jnp.nan)

    def dL_of_z(self, z):
        z = jnp.asarray(z)
        dL = jnp.interp(z, jnp.asarray(self._z_grid), jnp.asarray(self._dL_grid))
        valid = (z >= 0.0) & (z <= self._z_grid[-1])
        return jnp.where(valid, dL, jnp.nan)

    def comoving_distance_from_z(self, z):
        return self.dL_of_z(z) / (1.0 + jnp.asarray(z))

    def ddL_dz(self, z):
        """Analytic derivative d dL / dz for flat LambdaCDM."""
        z = jnp.asarray(z)
        dc = self.comoving_distance_from_z(z)
        return dc + (1.0 + z) * (C_KM_S / self.H0) / self.e(z)

    def dVc_dz(self, z):
        """Full-sky differential comoving volume in Mpc^3 per unit redshift."""
        z = jnp.asarray(z)
        dc = self.comoving_distance_from_z(z)
        return 4.0 * jnp.pi * (C_KM_S / self.H0) * dc**2 / self.e(z)
