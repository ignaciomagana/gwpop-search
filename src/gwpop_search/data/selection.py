"""Canonical selection-injection container."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from .schema import (
    CoordinateBasis,
    DataContractError,
    MissingCoordinateError,
    validate_log_denominator,
)

FORMAT_VERSION = "gwpop-search-selection-1.0"


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class SelectionMode(str, Enum):
    RAW_DRAW = "raw_draw"
    ESTIMATOR_READY = "estimator_ready"


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    n_draw: int | None = None
    observing_time_yr: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_id.strip():
            raise DataContractError("campaign_id must be non-empty")
        if self.n_draw is not None and int(self.n_draw) <= 0:
            raise DataContractError("campaign n_draw must be positive when supplied")
        if self.observing_time_yr is not None and (
            not np.isfinite(self.observing_time_yr) or self.observing_time_yr <= 0
        ):
            raise DataContractError(
                "campaign observing_time_yr must be finite and positive when supplied"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "n_draw": self.n_draw,
            "observing_time_yr": self.observing_time_yr,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Campaign":
        return cls(
            campaign_id=str(payload["campaign_id"]),
            n_draw=None if payload.get("n_draw") is None else int(payload["n_draw"]),
            observing_time_yr=(
                None
                if payload.get("observing_time_yr") is None
                else float(payload["observing_time_yr"])
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class SelectionCatalog:
    """Detected/usable injections with explicit denominator semantics."""

    samples: Mapping[str, np.ndarray]
    log_draw_density: np.ndarray
    campaign_id: np.ndarray
    campaigns: tuple[Campaign, ...]
    basis: CoordinateBasis
    mode: SelectionMode
    estimator_semantics: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples = {str(k): np.asarray(v, dtype=float) for k, v in self.samples.items()}
        self.log_draw_density = validate_log_denominator(
            self.log_draw_density, what="selection log_draw_density"
        )
        self.campaign_id = np.asarray([str(x) for x in self.campaign_id], dtype=str)
        self.campaigns = tuple(self.campaigns)
        self.mode = SelectionMode(self.mode)

        n = self.log_draw_density.size
        if n == 0:
            raise DataContractError("SelectionCatalog must contain at least one retained row")
        if self.campaign_id.ndim != 1 or self.campaign_id.size != n:
            raise DataContractError(
                f"campaign_id must be 1-D with length {n}; got {self.campaign_id.shape}"
            )
        if not self.samples:
            raise DataContractError("SelectionCatalog.samples cannot be empty")
        for name, values in self.samples.items():
            if values.ndim != 1 or values.size != n:
                raise DataContractError(
                    f"selection column {name!r} must be 1-D with length {n}; "
                    f"got {values.shape}"
                )
            if not np.all(np.isfinite(values)):
                bad = np.flatnonzero(~np.isfinite(values))[:8].tolist()
                raise DataContractError(
                    f"selection column {name!r} contains non-finite rows {bad}"
                )
        missing_basis = [x for x in self.basis.coordinates if x not in self.samples]
        if missing_basis:
            raise MissingCoordinateError(
                f"SelectionCatalog is missing basis coordinate(s) {missing_basis}"
            )

        if not self.campaigns:
            raise DataContractError("at least one campaign metadata record is required")
        ids = [campaign.campaign_id for campaign in self.campaigns]
        if len(ids) != len(set(ids)):
            raise DataContractError(f"duplicate campaign IDs: {ids}")
        unknown = sorted(set(self.campaign_id.tolist()) - set(ids))
        if unknown:
            raise DataContractError(
                f"selection rows refer to undeclared campaign ID(s) {unknown}"
            )
        if self.mode is SelectionMode.RAW_DRAW:
            missing_n = [c.campaign_id for c in self.campaigns if c.n_draw is None]
            if missing_n:
                raise DataContractError(
                    f"raw-draw selection requires n_draw for every campaign; missing {missing_n}"
                )
            if self.estimator_semantics is not None:
                raise DataContractError(
                    "raw-draw selection must not carry estimator_ready semantics"
                )
        else:
            if not self.estimator_semantics or not self.estimator_semantics.strip():
                raise DataContractError(
                    "estimator-ready selection requires a non-empty estimator_semantics string"
                )

    @property
    def n_selected(self) -> int:
        return self.log_draw_density.size

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.samples)

    def require(self, fields: Iterable[str]) -> None:
        missing = [field for field in fields if field not in self.samples]
        if missing:
            raise MissingCoordinateError(
                f"SelectionCatalog is missing required coordinate(s) {missing}; "
                f"available={list(self.samples)}"
            )

    def rows_for_campaign(self, campaign: str) -> np.ndarray:
        return np.flatnonzero(self.campaign_id == str(campaign))

    def to_hdf5(self, path: str | Path) -> str:
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["basis_json"] = json.dumps(self.basis.to_dict(), sort_keys=True)
            f.attrs["mode"] = self.mode.value
            f.attrs["estimator_semantics"] = self.estimator_semantics or ""
            f.attrs["metadata_json"] = json.dumps(
                self.metadata, sort_keys=True, default=_json_default
            )
            f.attrs["campaigns_json"] = json.dumps(
                [c.to_dict() for c in self.campaigns],
                sort_keys=True,
                default=_json_default,
            )
            f.attrs["field_names"] = np.asarray(
                self.field_names, dtype=h5py.string_dtype("utf-8")
            )
            grp = f.create_group("samples")
            for name, values in self.samples.items():
                grp.create_dataset(name, data=values)
            f.create_dataset("log_draw_density", data=self.log_draw_density)
            f.create_dataset(
                "campaign_id",
                data=np.asarray(self.campaign_id, dtype=h5py.string_dtype("utf-8")),
            )
        return str(path)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "SelectionCatalog":
        with h5py.File(path, "r") as f:
            fmt = _decode(f.attrs.get("format_version", ""))
            if fmt != FORMAT_VERSION:
                raise DataContractError(
                    f"expected {FORMAT_VERSION!r}, found {fmt!r} in {path}"
                )
            basis = CoordinateBasis.from_dict(json.loads(_decode(f.attrs["basis_json"])))
            metadata = json.loads(_decode(f.attrs.get("metadata_json", "{}")))
            campaigns = tuple(
                Campaign.from_dict(x)
                for x in json.loads(_decode(f.attrs["campaigns_json"]))
            )
            fields = tuple(_decode(x) for x in f.attrs["field_names"])
            samples = {name: f[f"samples/{name}"][:] for name in fields}
            log_draw = f["log_draw_density"][:]
            campaign_id = np.asarray([_decode(x) for x in f["campaign_id"][:]])
            mode = SelectionMode(_decode(f.attrs["mode"]))
            semantics = _decode(f.attrs.get("estimator_semantics", "")) or None
        return cls(
            samples=samples,
            log_draw_density=log_draw,
            campaign_id=campaign_id,
            campaigns=campaigns,
            basis=basis,
            mode=mode,
            estimator_semantics=semantics,
            metadata=metadata,
        )
