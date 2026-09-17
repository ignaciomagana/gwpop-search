"""Read validated gwcat v2 exports into the canonical Phase-1 containers.

This adapter consumes exported gwcat products. It does not ingest PESummary
files or reconstruct LVK priors. The exported p_pe and pdraw values are the
authoritative denominator densities.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..pair import validate_pair
from ..posterior import PosteriorCatalog
from ..schema import CoordinateBasis, DataContractError, ReferenceDensityError
from ..selection import Campaign, SelectionCatalog, SelectionMode

_PE_FORMATS = {"gwcat-pe-2.0", "gwcat-pe-2.1"}
_SEL_FORMATS = {"gwcat-selection-2.0", "gwcat-selection-2.1"}
_SUPPORTED_BASES = {"chieff", "chieff_chip", "component"}


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _attr(f: h5py.File, key: str, *, required: bool = True, default=None):
    if key in f.attrs:
        return f.attrs[key]
    if required:
        raise DataContractError(f"gwcat v2 file is missing required attr {key!r}")
    return default


def _positive_log(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr) | (arr <= 0.0)
    if bad.any():
        idx = np.flatnonzero(bad)[:8].tolist()
        raise ReferenceDensityError(
            f"gwcat {name} must be finite and strictly positive; "
            f"{int(bad.sum())} bad row(s), first indices={idx}"
        )
    return np.log(arr)


def _basis(spin_basis: str) -> CoordinateBasis:
    if spin_basis not in _SUPPORTED_BASES:
        raise DataContractError(
            f"gwcat v2 spin_basis={spin_basis!r} is not supported by the Phase-1 "
            f"adapter; supported={sorted(_SUPPORTED_BASES)}"
        )
    base = ("m1_detector", "q", "luminosity_distance", "ra", "dec")
    if spin_basis == "chieff":
        spin = ("chi_eff",)
        measure = "dm1_detector dq ddL dOmega dchi_eff"
    elif spin_basis == "chieff_chip":
        spin = ("chi_eff", "chi_p")
        measure = "dm1_detector dq ddL dOmega dchi_eff dchi_p"
    else:
        spin = ("a1", "a2", "cos_tilt1", "cos_tilt2")
        measure = "dm1_detector dq ddL dOmega da1 da2 dcos_tilt1 dcos_tilt2"
    return CoordinateBasis(
        name=f"gwcat_v2_{spin_basis}",
        coordinates=base + spin,
        frame="detector_mass_luminosity_distance",
        spin_parameterization=spin_basis,
        density_measure=measure,
        version="gwcat-v2",
    )


def _sample_columns(f: h5py.File, spin_basis: str) -> dict[str, np.ndarray]:
    required = ["m1det", "m2det", "dL", "ra", "dec"]
    if spin_basis in {"chieff", "chieff_chip"}:
        required.append("chieff")
    if spin_basis == "chieff_chip":
        required.append("chip")
    if spin_basis == "component":
        required.extend(["a1", "a2", "cost1", "cost2"])
    missing = [name for name in required if name not in f]
    if missing:
        raise DataContractError(
            f"gwcat v2 {spin_basis!r} product is missing dataset(s) {missing}"
        )

    m1 = np.asarray(f["m1det"][:], dtype=float)
    m2 = np.asarray(f["m2det"][:], dtype=float)
    q = np.asarray(f["q"][:], dtype=float) if "q" in f else m2 / m1
    columns: dict[str, np.ndarray] = {
        "m1_detector": m1,
        "q": q,
        "luminosity_distance": np.asarray(f["dL"][:], dtype=float),
        "ra": np.asarray(f["ra"][:], dtype=float),
        "dec": np.asarray(f["dec"][:], dtype=float),
    }
    if "m1src" in f:
        columns["m1_source"] = np.asarray(f["m1src"][:], dtype=float)
    if "m2src" in f:
        columns["m2_source"] = np.asarray(f["m2src"][:], dtype=float)
    if "redshift" in f:
        columns["z"] = np.asarray(f["redshift"][:], dtype=float)
    if "chieff" in f:
        columns["chi_eff"] = np.asarray(f["chieff"][:], dtype=float)
    if "chip" in f:
        columns["chi_p"] = np.asarray(f["chip"][:], dtype=float)
    for source, target in (
        ("a1", "a1"),
        ("a2", "a2"),
        ("cost1", "cos_tilt1"),
        ("cost2", "cos_tilt2"),
    ):
        if source in f:
            columns[target] = np.asarray(f[source][:], dtype=float)
    return columns


def load_pe(path: str | Path) -> PosteriorCatalog:
    """Load a gwcat-pe-2.0/2.1 export without changing its density basis."""

    with h5py.File(path, "r") as f:
        fmt = _decode(_attr(f, "format_version"))
        if fmt not in _PE_FORMATS:
            raise DataContractError(
                f"expected a gwcat v2 PE export, found format_version={fmt!r}"
            )
        spin_basis = _decode(_attr(f, "spin_basis"))
        basis = _basis(spin_basis)
        nobs = int(_attr(f, "nobs"))
        nsamp = int(_attr(f, "nsamp"))
        if nobs <= 0 or nsamp <= 0:
            raise DataContractError(f"invalid gwcat PE dimensions nobs={nobs}, nsamp={nsamp}")
        if "p_pe" not in f:
            raise DataContractError("gwcat v2 PE product is missing p_pe")
        log_ref = _positive_log(f["p_pe"][:], name="p_pe")
        expected = nobs * nsamp
        if log_ref.size != expected:
            raise DataContractError(
                f"gwcat PE p_pe length={log_ref.size}, expected nobs*nsamp={expected}"
            )
        event_names_raw = _attr(f, "event_names")
        event_names = tuple(_decode(x) for x in np.atleast_1d(event_names_raw))
        if len(event_names) != nobs:
            raise DataContractError(
                f"gwcat PE event_names has {len(event_names)} entries, expected {nobs}"
            )
        samples = _sample_columns(f, spin_basis)
        if any(len(x) != expected for x in samples.values()):
            bad = {k: len(v) for k, v in samples.items() if len(v) != expected}
            raise DataContractError(
                f"gwcat PE sample-column length mismatch; expected {expected}, got {bad}"
            )
        metadata = {
            "adapter": "gwcat_v2",
            "format_version": fmt,
            "spin_basis": spin_basis,
            "writer_commit": _decode(f.attrs.get("writer_commit", "unknown")),
            "p_pe_state": _decode(f.attrs.get("p_pe_state", "")),
        }
    offsets = np.arange(nobs + 1, dtype=np.int64) * nsamp
    return PosteriorCatalog(
        event_names=event_names,
        offsets=offsets,
        samples=samples,
        log_ref_density=log_ref,
        basis=basis,
        metadata=metadata,
    )


def load_selection(path: str | Path) -> SelectionCatalog:
    """Load a gwcat v2 selection export as an estimator-ready product.

    gwcat's exported pdraw already carries its multi-campaign mixture and
    exposure convention. This adapter therefore does not divide by ndraw or
    by observing time; Phase 2 must dispatch on SelectionMode.ESTIMATOR_READY.
    """

    with h5py.File(path, "r") as f:
        fmt = _decode(_attr(f, "format_version"))
        if fmt not in _SEL_FORMATS:
            raise DataContractError(
                f"expected a gwcat v2 selection export, found format_version={fmt!r}"
            )
        spin_basis = _decode(_attr(f, "spin_basis"))
        basis = _basis(spin_basis)
        if "pdraw" not in f:
            raise DataContractError("gwcat v2 selection product is missing pdraw")
        log_draw = _positive_log(f["pdraw"][:], name="pdraw")
        n_detected = int(_attr(f, "n_detected"))
        if log_draw.size != n_detected:
            raise DataContractError(
                f"gwcat selection pdraw length={log_draw.size}, "
                f"n_detected={n_detected}"
            )
        samples = _sample_columns(f, spin_basis)
        if any(len(x) != n_detected for x in samples.values()):
            bad = {k: len(v) for k, v in samples.items() if len(v) != n_detected}
            raise DataContractError(
                f"gwcat selection sample-column length mismatch; "
                f"expected {n_detected}, got {bad}"
            )
        ndraw = int(_attr(f, "ndraw"))
        tobs = float(_attr(f, "T_obs_yr"))
        semantics = _decode(_attr(f, "pdraw_state"))
        n_campaigns = int(f.attrs.get("n_campaigns", 1))
        campaign_ndraws = np.asarray(
            f.attrs.get("campaign_ndraws", [ndraw]), dtype=np.int64
        ).tolist()
        campaign = Campaign(
            "gwcat_combined",
            n_draw=ndraw,
            observing_time_yr=tobs,
            metadata={
                "n_campaigns": n_campaigns,
                "campaign_ndraws": campaign_ndraws,
            },
        )
        metadata = {
            "adapter": "gwcat_v2",
            "format_version": fmt,
            "spin_basis": spin_basis,
            "ndraw": ndraw,
            "T_obs_yr": tobs,
            "n_campaigns": n_campaigns,
            "campaign_ndraws": campaign_ndraws,
            "pdraw_state": semantics,
            "writer_commit": _decode(f.attrs.get("writer_commit", "unknown")),
        }
    return SelectionCatalog(
        samples=samples,
        log_draw_density=log_draw,
        campaign_id=np.asarray(["gwcat_combined"] * n_detected),
        campaigns=(campaign,),
        basis=basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics=semantics,
        metadata=metadata,
    )


def load_pair(
    pe_path: str | Path,
    selection_path: str | Path,
    *,
    required_coordinates: tuple[str, ...] | None = None,
) -> tuple[PosteriorCatalog, SelectionCatalog]:
    pe = load_pe(pe_path)
    selection = load_selection(selection_path)
    validate_pair(pe, selection, required_coordinates)
    return pe, selection
