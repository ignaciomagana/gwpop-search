"""Read validated gwcat v2 exports into the canonical Phase-1 containers.

This adapter consumes exported gwcat products. It does not ingest PESummary
files or reconstruct LVK priors. The exported p_pe and pdraw values are the
authoritative denominator densities.

Reference-reweighted chi_eff selection (gwcat >= 8f9e2f1, GW-38)
----------------------------------------------------------------
gwcat refuses the substituting ``chieff`` selection basis for campaigns whose
spins were not drawn uniform-magnitude/isotropic (e.g. the O4ab injections).
Its ``chieff_reference`` selection basis keeps each campaign's exact component
spin draw and reweights it to a declared isotropic uniform-magnitude reference
prior with ceiling ``a_ref``:

    pdraw = pdraw_component * p_iso(chi_eff | q, a_ref) / p_ref(a, cos t),
    p_ref = 1 / (4 a_ref**2).

That file is a density in the same ``(m1det, q, dL, ra, dec, chi_eff)``
measure as a ``chieff`` PE export, and it is exact only when paired with a
``chieff`` PE export whose divided-out prior has the SAME ceiling on every
event. This adapter therefore loads it in the ``gwcat_v2_chieff`` basis and
:func:`validate_reference_pairing` requires that ceiling equality.

Rows outside the reference support (``a1 > a_ref`` or ``a2 > a_ref``) carry
exactly zero reference weight; gwcat encodes them with a declared sentinel
``pdraw``. Per the data contract such encoded zero-importance rows get an
explicit treatment here: they are identified by the recorded sentinel value
AND the recorded support rule, counted against the recorded total, and
removed. Removal is exact for the estimator-ready sum
``A = sum p_pop / pdraw`` because their weight is identically zero.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ..pair import validate_pair
from ..posterior import PosteriorCatalog
from ..schema import (
    BasisMismatchError,
    CoordinateBasis,
    DataContractError,
    ReferenceDensityError,
)
from ..selection import Campaign, SelectionCatalog, SelectionMode

_PE_FORMATS = {"gwcat-pe-2.0", "gwcat-pe-2.1"}
_SEL_FORMATS = {"gwcat-selection-2.0", "gwcat-selection-2.1"}
_SUPPORTED_BASES = {"chieff", "chieff_chip", "component"}

#: Selection-only bases mapped to the PE basis (and density measure) they pair with.
REFERENCE_SELECTION_BASES = {"chieff_reference": "chieff"}

#: gwcat 8f9e2f1 ``PDRAW_STATE_CHIEFF_REFERENCE``, pinned verbatim: a file not
#: written by the reviewed reference-basis builder fails closed.
GWCAT_CHIEFF_REFERENCE_PDRAW_STATE = (
    "draw_density_in_(m1det,q,dL,chieff)_basis_with_1D_chi_eff_prior_included; "
    "built from the campaign's EXACT per-injection component spin draw and "
    "REWEIGHTED to a declared isotropic uniform-magnitude reference spin prior "
    "(ceiling spin_reference_amax) -- the campaign's own spin density is "
    "divided out, not discarded, so this is valid for a campaign of any spin "
    "distribution; rows outside the reference support carry weight exactly zero "
    "via spin_reference_excluded_pdraw; normalised by T_obs and injection "
    "weights. Detector-frame masses in Msun, dL in Mpc."
)


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


def basis_for_spin(spin_basis: str) -> CoordinateBasis:
    """Return the exact independent density basis used by gwcat-v2 exports."""
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


# Backward-compatible private alias used by the original Phase-1 implementation.
_basis = basis_for_spin


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


def _chieff_amax_provenance(f: h5py.File, nobs: int) -> dict[str, object] | None:
    """Per-event chi_eff prior ceilings a gwcat ``chieff`` PE export divided out."""
    names = ("chi_eff_amax_1_per_event", "chi_eff_amax_2_per_event")
    if not all(name in f.attrs for name in names):
        return None
    amax_1 = np.asarray(f.attrs[names[0]], dtype=float).reshape(-1)
    amax_2 = np.asarray(f.attrs[names[1]], dtype=float).reshape(-1)
    if amax_1.size != nobs or amax_2.size != nobs:
        raise DataContractError(
            f"gwcat PE per-event chi_eff amax has {amax_1.size}/{amax_2.size} "
            f"entries, expected nobs={nobs}"
        )
    sources = [
        _decode(x)
        for x in np.atleast_1d(f.attrs.get("chi_eff_amax_source_per_event", []))
    ]
    unrecognized = [
        _decode(x)
        for x in np.atleast_1d(f.attrs.get("spin_prior_unrecognized_events", []))
    ]
    return {
        "amax_1_per_event": amax_1.tolist(),
        "amax_2_per_event": amax_2.tolist(),
        "source_per_event": sources,
        "mode": _decode(f.attrs.get("chi_eff_amax_mode", "")),
        "spin_prior_unrecognized_events": unrecognized,
    }


def _reference_support(
    f: h5py.File,
    pdraw: np.ndarray,
    samples: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    """Identify gwcat ``chieff_reference`` zero-weight rows; return the keep mask."""
    state = _decode(_attr(f, "pdraw_state"))
    if state != GWCAT_CHIEFF_REFERENCE_PDRAW_STATE:
        raise DataContractError(
            "chieff_reference selection pdraw_state does not match the reviewed "
            "gwcat reference-basis constant"
        )
    a_ref = float(_attr(f, "spin_reference_amax"))
    if not np.isfinite(a_ref) or not 0.0 < a_ref <= 1.0:
        raise DataContractError(
            f"chieff_reference spin_reference_amax={a_ref!r} is not a ceiling in (0, 1]"
        )
    sentinel = float(_attr(f, "spin_reference_excluded_pdraw"))
    if not np.isfinite(sentinel) or sentinel <= 0.0:
        raise DataContractError(
            f"chieff_reference excluded-row sentinel {sentinel!r} is not finite positive"
        )
    n_declared = int(_attr(f, "spin_reference_excluded_rows"))
    if not bool(_attr(f, "spin_reference_coverage_ok")):
        raise DataContractError(
            "chieff_reference selection records a reference-support coverage hole "
            "(spin_reference_coverage_ok=False); the reweighted selection integral "
            "would be biased low"
        )
    missing = [name for name in ("a1", "a2", "chi_eff") if name not in samples]
    if missing:
        raise DataContractError(
            f"chieff_reference selection lacks reference-support coordinates {missing}"
        )

    excluded = pdraw == sentinel
    outside = (
        (samples["a1"] > a_ref)
        | (samples["a2"] > a_ref)
        | (np.abs(samples["chi_eff"]) > a_ref)
    )
    if int(excluded.sum()) != n_declared:
        raise DataContractError(
            f"chieff_reference selection has {int(excluded.sum())} sentinel rows but "
            f"records spin_reference_excluded_rows={n_declared}"
        )
    if np.any(outside & ~excluded):
        raise DataContractError(
            f"{int(np.sum(outside & ~excluded))} chieff_reference row(s) outside the "
            "declared reference support carry a non-sentinel pdraw"
        )
    if np.any(excluded & ~outside):
        raise DataContractError(
            f"{int(np.sum(excluded & ~outside))} chieff_reference sentinel row(s) lie "
            "inside the declared reference support"
        )
    return ~excluded, {
        "spin_reference_amax": a_ref,
        "excluded_pdraw_sentinel": sentinel,
        "n_detected_in_export": int(pdraw.size),
        "n_zero_weight_rows_removed": int(excluded.sum()),
        "removal_rule": (
            "rows with pdraw == spin_reference_excluded_pdraw, required to be exactly "
            "the rows with a1 > a_ref or a2 > a_ref or |chi_eff| > a_ref; zero "
            "reference weight, so removal leaves sum(p_pop / pdraw) unchanged"
        ),
    }


def load_pe(path: str | Path) -> PosteriorCatalog:
    """Load a gwcat-pe-2.0/2.1 export without changing its density basis."""

    with h5py.File(path, "r") as f:
        fmt = _decode(_attr(f, "format_version"))
        if fmt not in _PE_FORMATS:
            raise DataContractError(
                f"expected a gwcat v2 PE export, found format_version={fmt!r}"
            )
        spin_basis = _decode(_attr(f, "spin_basis"))
        basis = basis_for_spin(spin_basis)
        nobs = int(_attr(f, "nobs"))
        nsamp = int(_attr(f, "nsamp"))
        if nobs <= 0 or nsamp <= 0:
            raise DataContractError(
                f"invalid gwcat PE dimensions nobs={nobs}, nsamp={nsamp}"
            )
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
        if spin_basis == "chieff":
            amax = _chieff_amax_provenance(f, nobs)
            if amax is not None:
                metadata["chi_eff_amax"] = amax
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

    A ``chieff_reference`` export is loaded in the ``chieff`` density basis
    with its declared zero-weight rows removed explicitly (see module
    docstring); ``ndraw`` is unchanged because those rows are genuine draws.
    """

    with h5py.File(path, "r") as f:
        fmt = _decode(_attr(f, "format_version"))
        if fmt not in _SEL_FORMATS:
            raise DataContractError(
                f"expected a gwcat v2 selection export, found format_version={fmt!r}"
            )
        spin_basis = _decode(_attr(f, "spin_basis"))
        density_basis = REFERENCE_SELECTION_BASES.get(spin_basis, spin_basis)
        basis = basis_for_spin(density_basis)
        if "pdraw" not in f:
            raise DataContractError("gwcat v2 selection product is missing pdraw")
        pdraw = np.asarray(f["pdraw"][:], dtype=float)
        n_detected = int(_attr(f, "n_detected"))
        if pdraw.size != n_detected:
            raise DataContractError(
                f"gwcat selection pdraw length={pdraw.size}, "
                f"n_detected={n_detected}"
            )
        samples = _sample_columns(f, density_basis)
        if any(len(x) != n_detected for x in samples.values()):
            bad = {k: len(v) for k, v in samples.items() if len(v) != n_detected}
            raise DataContractError(
                f"gwcat selection sample-column length mismatch; "
                f"expected {n_detected}, got {bad}"
            )
        reference = None
        if spin_basis in REFERENCE_SELECTION_BASES:
            keep, reference = _reference_support(f, pdraw, samples)
            pdraw = pdraw[keep]
            samples = {name: values[keep] for name, values in samples.items()}
        log_draw = _positive_log(pdraw, name="pdraw")
        n_selected = int(log_draw.size)
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
        if reference is not None:
            metadata["spin_reference"] = reference
    return SelectionCatalog(
        samples=samples,
        log_draw_density=log_draw,
        campaign_id=np.asarray(["gwcat_combined"] * n_selected),
        campaigns=(campaign,),
        basis=basis,
        mode=SelectionMode.ESTIMATOR_READY,
        estimator_semantics=semantics,
        metadata=metadata,
    )


def validate_reference_pairing(
    pe: PosteriorCatalog,
    selection: SelectionCatalog,
) -> dict[str, object] | None:
    """Require one reference ceiling across a (chieff, chieff_reference) pair.

    The reference selection density was reweighted TO the prior the ``chieff``
    PE export divides OUT, so the two ceilings are the same object: any
    difference leaves an uncancelled chi_eff-dependent factor in every event
    weight. Each PE ceiling must also be the event's own sampling-prior
    ceiling (not a fallback or caller override) with a recognized
    uniform-magnitude/isotropic spin prior. Returns None for other pairs.
    """
    selection_basis = str(selection.metadata.get("spin_basis", ""))
    if selection_basis not in REFERENCE_SELECTION_BASES:
        return None
    pe_basis = str(pe.metadata.get("spin_basis", ""))
    required = REFERENCE_SELECTION_BASES[selection_basis]
    if pe_basis != required:
        raise BasisMismatchError(
            f"a {selection_basis!r} selection export pairs only with a "
            f"{required!r} PE export, got PE spin_basis={pe_basis!r}"
        )
    a_ref = float(dict(selection.metadata["spin_reference"])["spin_reference_amax"])
    amax = pe.metadata.get("chi_eff_amax")
    if amax is None:
        raise DataContractError(
            "chieff PE export records no per-event chi_eff prior ceilings "
            "(chi_eff_amax_1/2_per_event); cannot verify the reference pairing"
        )
    amax = dict(amax)
    ceilings = np.concatenate(
        [
            np.asarray(amax["amax_1_per_event"], dtype=float),
            np.asarray(amax["amax_2_per_event"], dtype=float),
        ]
    )
    if not np.all(np.isfinite(ceilings)) or not np.allclose(
        ceilings, a_ref, rtol=1e-9, atol=1e-12
    ):
        distinct = sorted({round(float(x), 9) for x in ceilings})
        raise DataContractError(
            f"PE chi_eff prior ceilings {distinct} != selection "
            f"spin_reference_amax {a_ref}"
        )
    sources = sorted(set(amax["source_per_event"]))
    if len(amax["source_per_event"]) != pe.n_events or sources != ["analytic"]:
        raise DataContractError(
            "every PE chi_eff ceiling must come from the event's own sampling prior "
            f"(source 'analytic'); got sources={sources}"
        )
    if amax["spin_prior_unrecognized_events"]:
        raise DataContractError(
            "PE events without a recognized uniform-magnitude/isotropic spin prior: "
            f"{amax['spin_prior_unrecognized_events']}"
        )
    return {
        "pe_spin_basis": pe_basis,
        "selection_spin_basis": selection_basis,
        "spin_reference_amax": a_ref,
        "pe_chi_eff_amax_all_equal_reference": True,
        "n_selection_zero_weight_rows_removed": int(
            dict(selection.metadata["spin_reference"])["n_zero_weight_rows_removed"]
        ),
    }


def load_pair(
    pe_path: str | Path,
    selection_path: str | Path,
    *,
    required_coordinates: tuple[str, ...] | None = None,
) -> tuple[PosteriorCatalog, SelectionCatalog]:
    pe = load_pe(pe_path)
    selection = load_selection(selection_path)
    validate_reference_pairing(pe, selection)
    validate_pair(pe, selection, required_coordinates)
    return pe, selection
