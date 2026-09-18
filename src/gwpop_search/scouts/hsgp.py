"""Hilbert-space GP basis utilities for population-structure scouting.

These utilities construct deterministic basis functions and squared-exponential
spectral weights. They do not by themselves define a normalized population
model or promote a scientific feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class HSGPAxis:
    name: str
    lower: float
    upper: float
    modes: int
    boundary_factor: float = 1.5

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("axis name cannot be empty")
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("axis bounds must be finite")
        if self.upper <= self.lower:
            raise ValueError("axis upper bound must exceed lower bound")
        if self.modes <= 0:
            raise ValueError("modes must be positive")
        if self.boundary_factor <= 1.0:
            raise ValueError("boundary_factor must exceed one")

    @property
    def center(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    @property
    def domain_half_width(self) -> float:
        return self.boundary_factor * self.half_width


def laplacian_frequencies(axis: HSGPAxis) -> np.ndarray:
    """Dirichlet Laplacian square-root eigenvalues on the expanded interval."""
    indices = np.arange(1, axis.modes + 1, dtype=float)
    return indices * np.pi / (2.0 * axis.domain_half_width)


def basis_matrix(values, axis: HSGPAxis) -> np.ndarray:
    """Orthonormal Dirichlet sine basis evaluated on one physical coordinate."""
    values = np.asarray(values, dtype=float)
    L = axis.domain_half_width
    shifted = values - axis.center + L
    frequencies = laplacian_frequencies(axis)
    return np.sin(np.outer(shifted, frequencies)) / np.sqrt(L)


def squared_exponential_spectral_weights(
    axis: HSGPAxis,
    *,
    amplitude: float,
    length_scale: float,
) -> np.ndarray:
    """Square roots of the 1D squared-exponential spectral density."""
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive")
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")

    omega = laplacian_frequencies(axis)
    spectral_density = (
        amplitude**2
        * np.sqrt(2.0 * np.pi)
        * length_scale
        * np.exp(-0.5 * (length_scale * omega) ** 2)
    )
    return np.sqrt(spectral_density)


@dataclass(frozen=True)
class HSGPDesign:
    axes: tuple[HSGPAxis, ...]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("at least one HSGP axis is required")
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError("HSGP axis names must be unique")

    @property
    def n_coefficients(self) -> int:
        result = 1
        for axis in self.axes:
            result *= axis.modes
        return result

    def tensor_basis(self, samples: Mapping[str, np.ndarray]) -> np.ndarray:
        """Kronecker-product basis with rows corresponding to input samples."""
        matrices = []
        n_rows = None
        for axis in self.axes:
            if axis.name not in samples:
                raise KeyError(f"missing HSGP coordinate {axis.name!r}")
            matrix = basis_matrix(samples[axis.name], axis)
            if n_rows is None:
                n_rows = matrix.shape[0]
            elif matrix.shape[0] != n_rows:
                raise ValueError("HSGP coordinates must have equal sample lengths")
            matrices.append(matrix)

        design = matrices[0]
        for matrix in matrices[1:]:
            design = np.einsum("ni,nj->nij", design, matrix).reshape(
                design.shape[0], -1
            )
        return design

    def tensor_spectral_scale(
        self,
        *,
        amplitudes: Mapping[str, float],
        length_scales: Mapping[str, float],
    ) -> np.ndarray:
        scales = []
        for axis in self.axes:
            scales.append(
                squared_exponential_spectral_weights(
                    axis,
                    amplitude=float(amplitudes[axis.name]),
                    length_scale=float(length_scales[axis.name]),
                )
            )
        result = scales[0]
        for scale in scales[1:]:
            result = np.multiply.outer(result, scale).reshape(-1)
        return result
