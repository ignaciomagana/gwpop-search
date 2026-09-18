"""Conditionally normalized HSGP residual scouts for BBH population structure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

try:
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "HSGP residual scouts require JAX: install gwpop-search[inference]"
    ) from exc

from gwpop_search.grammar import ModelSpec
from gwpop_search.inference.priors import PriorSpec
from gwpop_search.models.declarative import (
    DeclarativeGwcatChiEffModel,
    chieff_logpdf_from_spec,
    pairing_logpdf_from_spec,
)

from .hsgp import (
    HSGPAxis,
    laplacian_frequencies,
    squared_exponential_spectral_weights,
)


_ALLOWED = {
    "q": {"m1_source", "z"},
    "chi_eff": {"m1_source", "q", "z"},
}


@dataclass(frozen=True)
class ConditionalHSGPConfig:
    """One target/covariate residual surface with fixed GP hyperparameters."""

    target: str
    covariate: str
    target_axis: HSGPAxis
    covariate_axis: HSGPAxis
    amplitude: float = 1.0
    target_length_scale: float = 0.2
    covariate_length_scale: float = 0.2
    quadrature_order: int = 48
    coefficient_prefix: str = "hsgp"

    def __post_init__(self) -> None:
        if self.target not in _ALLOWED:
            raise ValueError(f"unsupported HSGP target {self.target!r}")
        if self.covariate not in _ALLOWED[self.target]:
            raise ValueError(
                f"unsupported {self.target!r} covariate {self.covariate!r}"
            )
        if self.target_axis.name != self.target:
            raise ValueError("target_axis name must equal target")
        if self.covariate_axis.name != self.covariate:
            raise ValueError("covariate_axis name must equal covariate")
        if not np.isfinite(self.amplitude) or self.amplitude <= 0.0:
            raise ValueError("amplitude must be finite and positive")
        if self.target_length_scale <= 0.0 or self.covariate_length_scale <= 0.0:
            raise ValueError("HSGP length scales must be positive")
        if self.quadrature_order < 16:
            raise ValueError("quadrature_order must be at least 16")
        if not self.coefficient_prefix:
            raise ValueError("coefficient_prefix cannot be empty")

    @property
    def n_coefficients(self) -> int:
        return self.target_axis.modes * self.covariate_axis.modes

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        width = max(3, len(str(self.n_coefficients - 1)))
        return tuple(
            f"{self.coefficient_prefix}_{index:0{width}d}"
            for index in range(self.n_coefficients)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "covariate": self.covariate,
            "target_axis": asdict(self.target_axis),
            "covariate_axis": asdict(self.covariate_axis),
            "amplitude": float(self.amplitude),
            "target_length_scale": float(self.target_length_scale),
            "covariate_length_scale": float(self.covariate_length_scale),
            "quadrature_order": int(self.quadrature_order),
            "coefficient_prefix": self.coefficient_prefix,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConditionalHSGPConfig":
        return cls(
            target=str(payload["target"]),
            covariate=str(payload["covariate"]),
            target_axis=HSGPAxis(**dict(payload["target_axis"])),
            covariate_axis=HSGPAxis(**dict(payload["covariate_axis"])),
            amplitude=float(payload.get("amplitude", 1.0)),
            target_length_scale=float(payload["target_length_scale"]),
            covariate_length_scale=float(payload["covariate_length_scale"]),
            quadrature_order=int(payload.get("quadrature_order", 48)),
            coefficient_prefix=str(payload.get("coefficient_prefix", "hsgp")),
        )


@lru_cache(maxsize=16)
def _legendre_nodes(order: int):
    return np.polynomial.legendre.leggauss(int(order))


def _jax_basis(values, axis: HSGPAxis):
    values = jnp.asarray(values)
    frequencies = jnp.asarray(laplacian_frequencies(axis))
    L = float(axis.domain_half_width)
    shifted = values[..., None] - float(axis.center) + L
    return jnp.sin(shifted * frequencies) / jnp.sqrt(L)


def _tensor_basis(target, covariate, config: ConditionalHSGPConfig):
    phi_target = _jax_basis(target, config.target_axis)
    phi_covariate = _jax_basis(covariate, config.covariate_axis)
    outer = phi_target[..., :, None] * phi_covariate[..., None, :]
    return outer.reshape(outer.shape[:-2] + (config.n_coefficients,))


def _spectral_scale(config: ConditionalHSGPConfig):
    target = squared_exponential_spectral_weights(
        config.target_axis,
        amplitude=1.0,
        length_scale=config.target_length_scale,
    )
    covariate = squared_exponential_spectral_weights(
        config.covariate_axis,
        amplitude=1.0,
        length_scale=config.covariate_length_scale,
    )
    scale = np.multiply.outer(target, covariate).reshape(-1)
    return config.amplitude * scale


def coefficient_priors(
    config: ConditionalHSGPConfig,
) -> dict[str, PriorSpec]:
    """Independent standard-normal weights before the fixed spectral scaling."""
    return {
        name: PriorSpec("normal", loc=0.0, scale=1.0)
        for name in config.coefficient_names
    }


@dataclass(frozen=True)
class ConditionalHSGPResidualModel:
    """Replace one baseline conditional factor by an HSGP-residual conditional.

    The baseline model and its hyperparameters are frozen. Only the HSGP
    coefficients are inferred. Conditional normalization is performed by
    Gauss--Legendre quadrature for every covariate point.
    """

    base_spec: ModelSpec
    base_hyperparameters: Mapping[str, float]
    config: ConditionalHSGPConfig
    base_model: DeclarativeGwcatChiEffModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        base = DeclarativeGwcatChiEffModel(self.base_spec)
        missing = set(self.base_spec.priors) - set(self.base_hyperparameters)
        if missing:
            raise ValueError(
                f"base_hyperparameters missing {sorted(missing)}"
            )
        object.__setattr__(
            self,
            "base_hyperparameters",
            {
                str(name): float(value)
                for name, value in self.base_hyperparameters.items()
            },
        )
        object.__setattr__(self, "base_model", base)

    @property
    def required_fields(self):
        return self.base_model.required_fields

    @property
    def coefficient_names(self):
        return self.config.coefficient_names

    def to_config(self) -> dict[str, object]:
        return {
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "base_model_hash": self.base_spec.model_hash,
            "base_hyperparameters": dict(self.base_hyperparameters),
            "hsgp": self.config.to_dict(),
            "normalization": "conditional_gauss_legendre",
        }

    def _coefficients(self, hyperparameters):
        missing = set(self.coefficient_names) - set(hyperparameters)
        if missing:
            raise KeyError(
                f"missing HSGP coefficient(s): {sorted(missing)}"
            )
        raw = jnp.stack(
            [jnp.asarray(hyperparameters[name]) for name in self.coefficient_names]
        )
        return raw * jnp.asarray(_spectral_scale(self.config))

    def _residual(self, target, covariate, hyperparameters):
        design = _tensor_basis(target, covariate, self.config)
        return jnp.sum(
            design * self._coefficients(hyperparameters),
            axis=-1,
        )

    def _source_coordinates(self, samples):
        m1_detector = jnp.asarray(samples["m1_detector"])
        q = jnp.asarray(samples["q"])
        d_l = jnp.asarray(samples["luminosity_distance"])
        chi_eff = jnp.asarray(samples["chi_eff"])
        z = self.base_model.cosmology.z_of_dL(d_l)
        m1_source = m1_detector / (1.0 + z)
        return {
            "m1_source": m1_source,
            "q": q,
            "z": z,
            "chi_eff": chi_eff,
        }

    def _covariate(self, coordinates):
        return coordinates[self.config.covariate]

    def _q_conditional_logpdf(
        self,
        q,
        m1_source,
        z,
        hyperparameters,
    ):
        hp = self.base_hyperparameters
        q = jnp.asarray(q)
        m1_source = jnp.asarray(m1_source)
        z = jnp.asarray(z)
        covariate = (
            m1_source if self.config.covariate == "m1_source" else z
        )

        base_logp = pairing_logpdf_from_spec(
            self.base_spec.pairing,
            q,
            m1_source,
            hp,
            q_floor=self.base_model.q_floor,
        )
        residual = self._residual(q, covariate, hyperparameters)

        nodes, weights = _legendre_nodes(self.config.quadrature_order)
        nodes = jnp.asarray(nodes)
        weights = jnp.asarray(weights)
        qmin = jnp.maximum(
            self.base_model.q_floor,
            hp["mmin"] / jnp.where(m1_source > 0.0, m1_source, 1.0),
        )
        valid = qmin < 1.0
        safe_width = jnp.where(valid, 1.0 - qmin, 1.0)
        q_nodes = (
            qmin[..., None]
            + 0.5 * safe_width[..., None] * (nodes + 1.0)
        )
        q_weights = 0.5 * safe_width[..., None] * weights

        m1_nodes = jnp.broadcast_to(m1_source[..., None], q_nodes.shape)
        z_nodes = jnp.broadcast_to(z[..., None], q_nodes.shape)
        cov_nodes = (
            m1_nodes
            if self.config.covariate == "m1_source"
            else z_nodes
        )
        base_nodes = pairing_logpdf_from_spec(
            self.base_spec.pairing,
            q_nodes,
            m1_nodes,
            hp,
            q_floor=self.base_model.q_floor,
        )
        residual_nodes = self._residual(
            q_nodes,
            cov_nodes,
            hyperparameters,
        )
        safe_weights = jnp.where(q_weights > 0.0, q_weights, 1.0)
        log_norm = logsumexp(
            base_nodes + residual_nodes + jnp.log(safe_weights),
            axis=-1,
        )
        adjusted = base_logp + residual - log_norm
        return jnp.where(valid, adjusted, -jnp.inf)

    def _chieff_conditional_logpdf(
        self,
        chi_eff,
        m1_source,
        q,
        z,
        hyperparameters,
    ):
        hp = self.base_hyperparameters
        chi_eff = jnp.asarray(chi_eff)
        m1_source = jnp.asarray(m1_source)
        q = jnp.asarray(q)
        z = jnp.asarray(z)
        covariate = {
            "m1_source": m1_source,
            "q": q,
            "z": z,
        }[self.config.covariate]

        base_logp = chieff_logpdf_from_spec(
            self.base_spec.chieff,
            chi_eff,
            m1_source,
            q,
            z,
            hp,
        )
        residual = self._residual(
            chi_eff,
            covariate,
            hyperparameters,
        )

        nodes, weights = _legendre_nodes(self.config.quadrature_order)
        nodes = jnp.asarray(nodes)
        weights = jnp.asarray(weights)
        chi_nodes = jnp.broadcast_to(nodes, chi_eff.shape + nodes.shape)
        chi_weights = jnp.broadcast_to(weights, chi_nodes.shape)

        m1_nodes = jnp.broadcast_to(m1_source[..., None], chi_nodes.shape)
        q_nodes = jnp.broadcast_to(q[..., None], chi_nodes.shape)
        z_nodes = jnp.broadcast_to(z[..., None], chi_nodes.shape)
        cov_nodes = {
            "m1_source": m1_nodes,
            "q": q_nodes,
            "z": z_nodes,
        }[self.config.covariate]

        base_nodes = chieff_logpdf_from_spec(
            self.base_spec.chieff,
            chi_nodes,
            m1_nodes,
            q_nodes,
            z_nodes,
            hp,
        )
        residual_nodes = self._residual(
            chi_nodes,
            cov_nodes,
            hyperparameters,
        )
        log_norm = logsumexp(
            base_nodes + residual_nodes + jnp.log(weights),
            axis=-1,
        )
        return base_logp + residual - log_norm

    def conditional_logpdf_source(
        self,
        target,
        *,
        m1_source,
        q,
        z,
        hyperparameters,
    ):
        """Evaluate only the adjusted normalized target conditional."""
        if self.config.target == "q":
            return self._q_conditional_logpdf(
                target,
                m1_source,
                z,
                hyperparameters,
            )
        return self._chieff_conditional_logpdf(
            target,
            m1_source,
            q,
            z,
            hyperparameters,
        )

    def __call__(
        self,
        samples: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ):
        coordinates = self._source_coordinates(samples)
        base_full = self.base_model(samples, self.base_hyperparameters)

        if self.config.target == "q":
            old_conditional = pairing_logpdf_from_spec(
                self.base_spec.pairing,
                coordinates["q"],
                coordinates["m1_source"],
                self.base_hyperparameters,
                q_floor=self.base_model.q_floor,
            )
            new_conditional = self._q_conditional_logpdf(
                coordinates["q"],
                coordinates["m1_source"],
                coordinates["z"],
                hyperparameters,
            )
        else:
            old_conditional = chieff_logpdf_from_spec(
                self.base_spec.chieff,
                coordinates["chi_eff"],
                coordinates["m1_source"],
                coordinates["q"],
                coordinates["z"],
                self.base_hyperparameters,
            )
            new_conditional = self._chieff_conditional_logpdf(
                coordinates["chi_eff"],
                coordinates["m1_source"],
                coordinates["q"],
                coordinates["z"],
                hyperparameters,
            )

        valid = jnp.isfinite(base_full) & jnp.isfinite(old_conditional)
        safe_base = jnp.where(valid, base_full, 0.0)
        safe_old = jnp.where(valid, old_conditional, 0.0)
        safe_new = jnp.where(jnp.isfinite(new_conditional), new_conditional, 0.0)
        result = safe_base - safe_old + safe_new
        return jnp.where(
            valid & jnp.isfinite(new_conditional),
            result,
            -jnp.inf,
        )
