"""Typed coordinate and reference-density contracts for population inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np


class DataContractError(ValueError):
    """Base class for errors at the data/HBI boundary."""


class MissingCoordinateError(DataContractError):
    """A requested coordinate is absent or unavailable."""


class BasisMismatchError(DataContractError):
    """PE and selection products describe different density measures."""


class ReferenceDensityError(DataContractError):
    """A denominator density is invalid on a retained sample."""


@dataclass(frozen=True)
class CoordinateBasis:
    """Identity of the measure with respect to which a density is expressed.

    coordinates contains only independent density coordinates. A data product
    may expose additional derived columns in samples without putting them in
    this tuple.
    """

    name: str
    coordinates: tuple[str, ...]
    frame: str
    spin_parameterization: str
    density_measure: str
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise DataContractError("CoordinateBasis.name must be non-empty")
        if not self.coordinates:
            raise DataContractError("CoordinateBasis.coordinates must be non-empty")
        if len(set(self.coordinates)) != len(self.coordinates):
            raise DataContractError(
                f"CoordinateBasis coordinates are not unique: {self.coordinates!r}"
            )
        if any(not str(c).strip() for c in self.coordinates):
            raise DataContractError("Coordinate names must be non-empty strings")
        for label, value in (
            ("frame", self.frame),
            ("spin_parameterization", self.spin_parameterization),
            ("density_measure", self.density_measure),
            ("version", self.version),
        ):
            if not str(value).strip():
                raise DataContractError(f"CoordinateBasis.{label} must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "coordinates": list(self.coordinates),
            "frame": self.frame,
            "spin_parameterization": self.spin_parameterization,
            "density_measure": self.density_measure,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CoordinateBasis":
        return cls(
            name=str(payload["name"]),
            coordinates=tuple(str(x) for x in payload["coordinates"]),
            frame=str(payload["frame"]),
            spin_parameterization=str(payload["spin_parameterization"]),
            density_measure=str(payload["density_measure"]),
            version=str(payload.get("version", "1")),
        )

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def require_coordinates(self, fields: Iterable[str]) -> None:
        missing = [field for field in fields if field not in self.coordinates]
        if missing:
            raise MissingCoordinateError(
                f"Basis {self.name!r} does not contain density coordinate(s) {missing}; "
                f"coordinates={list(self.coordinates)}"
            )


def validate_log_denominator(log_density: np.ndarray, *, what: str) -> np.ndarray:
    """Validate a log density that will appear in an importance denominator.

    Retained samples must have a finite, strictly positive density. In log space
    that means every value is finite. Generic inference code must not repair
    zeros with numerical floors.
    """

    arr = np.asarray(log_density, dtype=float)
    if arr.ndim != 1:
        raise ReferenceDensityError(f"{what} must be one-dimensional; got {arr.shape}")
    bad = ~np.isfinite(arr)
    if bad.any():
        idx = np.flatnonzero(bad)[:8].tolist()
        raise ReferenceDensityError(
            f"{what} has {int(bad.sum())} non-finite retained value(s); "
            f"first indices={idx}. Zero/out-of-support denominator densities must "
            "be handled explicitly by the adapter, never floored in HBI code."
        )
    return arr


def require_compatible_bases(left: CoordinateBasis, right: CoordinateBasis) -> None:
    """Require exact density-basis compatibility."""

    if left.identity != right.identity:
        raise BasisMismatchError(
            "PE/selection density-basis mismatch: "
            f"PE={left.to_dict()} (id={left.identity}), "
            f"selection={right.to_dict()} (id={right.identity})"
        )


def availability_from_samples(
    samples: Mapping[str, np.ndarray], offsets: Sequence[int]
) -> tuple[tuple[str, ...], np.ndarray]:
    """Infer event-by-field availability from finite values in each event slice."""

    fields = tuple(samples)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    n_events = len(offsets_arr) - 1
    mask = np.zeros((n_events, len(fields)), dtype=bool)
    for i in range(n_events):
        sl = slice(int(offsets_arr[i]), int(offsets_arr[i + 1]))
        for j, field in enumerate(fields):
            values = np.asarray(samples[field])[sl]
            mask[i, j] = values.size > 0 and bool(np.all(np.isfinite(values)))
    return fields, mask
