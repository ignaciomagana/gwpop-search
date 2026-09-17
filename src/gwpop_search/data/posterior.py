"""Canonical ragged posterior-sample container."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from .schema import (
    CoordinateBasis,
    DataContractError,
    MissingCoordinateError,
    availability_from_samples,
    validate_log_denominator,
)

FORMAT_VERSION = "gwpop-search-pe-1.0"


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass
class PosteriorCatalog:
    """Posterior samples with ragged event slices and explicit PE reference density."""

    event_names: tuple[str, ...]
    offsets: np.ndarray
    samples: Mapping[str, np.ndarray]
    log_ref_density: np.ndarray
    basis: CoordinateBasis
    availability: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_names = tuple(str(x) for x in self.event_names)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.samples = {str(k): np.asarray(v, dtype=float) for k, v in self.samples.items()}
        self.log_ref_density = validate_log_denominator(
            self.log_ref_density, what="PE log_ref_density"
        )

        if self.offsets.ndim != 1 or len(self.offsets) != len(self.event_names) + 1:
            raise DataContractError(
                "offsets must be one-dimensional with len(event_names)+1 entries"
            )
        if self.offsets.size == 0 or self.offsets[0] != 0:
            raise DataContractError("offsets must start at zero")
        if np.any(np.diff(self.offsets) <= 0):
            raise DataContractError("every posterior event must contain at least one sample")
        n_total = int(self.offsets[-1])
        if n_total != self.log_ref_density.size:
            raise DataContractError(
                f"offsets[-1]={n_total} but log_ref_density has length "
                f"{self.log_ref_density.size}"
            )
        if not self.samples:
            raise DataContractError("PosteriorCatalog.samples cannot be empty")
        for name, values in self.samples.items():
            if values.ndim != 1 or values.size != n_total:
                raise DataContractError(
                    f"sample column {name!r} must be 1-D with length {n_total}; "
                    f"got {values.shape}"
                )
        missing_basis = [x for x in self.basis.coordinates if x not in self.samples]
        if missing_basis:
            raise MissingCoordinateError(
                f"PosteriorCatalog is missing basis coordinate(s) {missing_basis}"
            )

        self._field_names = tuple(self.samples)
        if self.availability is None:
            _, self.availability = availability_from_samples(self.samples, self.offsets)
        else:
            self.availability = np.asarray(self.availability, dtype=bool)
        expected = (self.n_events, len(self._field_names))
        if self.availability.shape != expected:
            raise DataContractError(
                f"availability must have shape {expected}; got {self.availability.shape}"
            )

        for i in range(self.n_events):
            sl = self.event_slice(i)
            for j, name in enumerate(self._field_names):
                if self.availability[i, j] and not np.all(np.isfinite(self.samples[name][sl])):
                    raise DataContractError(
                        f"field {name!r} is marked available for event "
                        f"{self.event_names[i]!r} but contains non-finite values"
                    )

    @property
    def n_events(self) -> int:
        return len(self.event_names)

    @property
    def n_samples_total(self) -> int:
        return int(self.offsets[-1])

    @property
    def field_names(self) -> tuple[str, ...]:
        return self._field_names

    def event_slice(self, event_index: int) -> slice:
        if not 0 <= int(event_index) < self.n_events:
            raise IndexError(event_index)
        i = int(event_index)
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def sample_count(self, event_index: int) -> int:
        sl = self.event_slice(event_index)
        return int(sl.stop - sl.start)

    def require(self, fields: Iterable[str]) -> None:
        problems: list[str] = []
        field_index = {name: i for i, name in enumerate(self._field_names)}
        for field_name in fields:
            if field_name not in field_index:
                problems.append(f"{field_name}: absent from catalog")
                continue
            col = self.availability[:, field_index[field_name]]
            if not np.all(col):
                missing_events = [
                    self.event_names[i] for i in np.flatnonzero(~col)[:8]
                ]
                problems.append(
                    f"{field_name}: unavailable for event(s) {missing_events}"
                )
        if problems:
            raise MissingCoordinateError("; ".join(problems))

    def get_event(
        self, event_index: int, fields: Iterable[str] | None = None
    ) -> dict[str, np.ndarray]:
        selected = tuple(self._field_names if fields is None else fields)
        self.require(selected)
        sl = self.event_slice(event_index)
        return {name: self.samples[name][sl] for name in selected}

    def event_log_ref_density(self, event_index: int) -> np.ndarray:
        return self.log_ref_density[self.event_slice(event_index)]

    def to_hdf5(self, path: str | Path) -> str:
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["basis_json"] = json.dumps(self.basis.to_dict(), sort_keys=True)
            f.attrs["metadata_json"] = json.dumps(
                self.metadata, sort_keys=True, default=_json_default
            )
            f.attrs["field_names"] = np.asarray(
                self._field_names, dtype=h5py.string_dtype("utf-8")
            )
            idx = f.create_group("index")
            idx.create_dataset(
                "event_names",
                data=np.asarray(self.event_names, dtype=h5py.string_dtype("utf-8")),
            )
            idx.create_dataset("offsets", data=self.offsets)
            grp = f.create_group("samples")
            for name, values in self.samples.items():
                grp.create_dataset(name, data=values)
            f.create_dataset("log_ref_density", data=self.log_ref_density)
            avail = f.create_group("availability")
            avail.create_dataset("mask", data=self.availability)
        return str(path)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "PosteriorCatalog":
        with h5py.File(path, "r") as f:
            fmt = _decode(f.attrs.get("format_version", ""))
            if fmt != FORMAT_VERSION:
                raise DataContractError(
                    f"expected {FORMAT_VERSION!r}, found {fmt!r} in {path}"
                )
            basis = CoordinateBasis.from_dict(json.loads(_decode(f.attrs["basis_json"])))
            metadata = json.loads(_decode(f.attrs.get("metadata_json", "{}")))
            names = tuple(_decode(x) for x in f["index/event_names"][:])
            offsets = f["index/offsets"][:]
            field_names = tuple(_decode(x) for x in f.attrs["field_names"])
            samples = {name: f[f"samples/{name}"][:] for name in field_names}
            log_ref = f["log_ref_density"][:]
            availability = f["availability/mask"][:]
        return cls(
            event_names=names,
            offsets=offsets,
            samples=samples,
            log_ref_density=log_ref,
            basis=basis,
            availability=availability,
            metadata=metadata,
        )
