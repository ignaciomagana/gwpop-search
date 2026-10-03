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

That file is a density in the same ``(m1det, q, dL, [ra, dec,] chi_eff)``
measure as a ``chieff`` PE export, and it is exact only when paired with a
``chieff`` PE export whose divided-out prior has the SAME ceiling on every
event. This adapter therefore loads it in the ``gwcat_v2_chieff`` basis and
:func:`validate_reference_pairing` requires that ceiling equality.

Rows outside the reference support (``a1 > a_ref`` or ``a2 > a_ref``) carry
exactly zero reference weight; gwcat encodes them with a declared sentinel
``pdraw``. Per the data contract such encoded zero-importance rows get an
explicit treatment here: they are identified by the recorded sentinel value
AND the recorded support rule, counted against the recorded total, and
removed (BUILD_PLAN OD-9). Removal is exact for the estimator-ready sum
``A = sum p_pop / pdraw`` because their weight is identically zero.

Sky-marginal products (gwcat PR-A, GW-39)
-----------------------------------------
The LVK cumulative O1-O4b mixture has no sky columns. gwcat's
``--sky-marginal`` export omits ra/dec and records ``sky_marginalized=True``
(and ``sky_position_available=False`` per campaign). Neither p_pe nor pdraw
contains a sky density and every population model is isotropic, so the sky is
marginalised exactly: the pair is loaded in a *sky-marginal* basis whose
coordinates and measure omit ``(ra, dec)`` / ``dOmega``. The PE half follows
the selection's declaration (its ra/dec columns are dropped, never used). The
basis identity differs from the sky-resolved one, so a v1 (sky-resolved) and a
v2 (sky-marginal) product can never be paired by accident.

Cumulative-mixture campaigns (gwcat PR-A, GW-39)
------------------------------------------------
A ``cumulative_mixture`` selection records, per observing run, the detection
rule that was applied (semianalytic SNR for O1/O2, the run's own minimum
search FAR for O3/O4) and, per exposure component, the derived draw counts
``N_per_component`` and times ``T_per_component_s`` with their
``T_definition_per_component``, mapped by ``component_of_run``. The adapter
validates that these records are complete, aligned and mutually consistent,
and that they add up to the file's ``ndraw`` and ``T_obs_yr``.

Per-prior provenance (gwcat PR-B, GW-40)
----------------------------------------
Since GW-40c ``chi_eff_amax_source_per_event`` is the spin prior's SOURCE KIND
(``own_analytic`` / ``sibling_inherited`` / ``config_file_declared`` /
``assumed_default``); the historical resolution (``analytic`` / ``fallback`` /
``caller``) moved to ``chi_eff_amax_resolution_per_event``. A reference pair
is exact only for ``own_analytic`` events; every other event must be named on
an explicit allow-list (BUILD_PLAN OD-7), with its kind, or the pair is
refused.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
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
_SKY_COLUMNS = ("ra", "dec")

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

#: gwcat GW-40b prior-source kinds. Only ``own_analytic`` is exact for a
#: reference pair; ``unrecorded`` is what a store without provenance yields.
PRIOR_SOURCE_KINDS = frozenset(
    {
        "own_analytic",
        "sibling_inherited",
        "config_file_declared",
        "assumed_default",
        "release_reweighted",
        "constituent_mixture",
        "unrecorded",
    }
)
SPIN_PRIOR_SOURCE_OK = "own_analytic"

#: gwcat GW-40i exact prior implementations.
DL_PRIOR_IMPLS_EXACT = frozenset({"exact", "analytic"})
CHI_EFF_PRIOR_IMPL_EXACT = "exact"

#: gwcat ``observing_runs`` tables (PR-A), pinned: a mixture that names an
#: unknown run, component or exposure definition fails closed.
MIXTURE_RUN_LABELS = ("O1", "O2", "O3a", "O3b", "O4a", "O4b")
MIXTURE_COMPONENT_OF_RUN = {
    "O1": "O1",
    "O2": "O2",
    "O3a": "O3",
    "O3b": "O3",
    "O4a": "O4",
    "O4b": "O4",
}
MIXTURE_T_DEFINITIONS = frozenset(
    {"coincident_livetime_semianalytic", "endo3_analysis_time", "monthly_wall_clock"}
)
#: T_obs_yr = total_analysis_time / Julian year (gwcat selection loader).
_SECONDS_PER_JULIAN_YEAR = 365.25 * 86400.0
_FAR_RULE = re.compile(r"^min\((?P<cols>[^()]+)\)<(?P<thr>[-+0-9.eE]+)$")
_SNR_RULE = re.compile(r"^(?P<col>[A-Za-z0-9_]+)>(?P<thr>[-+0-9.eE]+)$")
_MIXTURE_REQUIRED_ATTRS = (
    "run_labels",
    "n_rows_per_run",
    "n_detected_per_run",
    "detection_rule_per_run",
    "T_definition_per_run",
    "component_of_run",
    "mixture_components",
    "N_per_component",
    "T_per_component_s",
    "T_definition_per_component",
)


def _decode(value):
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _strings(value) -> list[str]:
    return [_decode(x) for x in np.atleast_1d(value)]


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


def basis_for_spin(spin_basis: str, *, sky_marginal: bool = False) -> CoordinateBasis:
    """Return the exact independent density basis used by gwcat-v2 exports.

    ``sky_marginal=True`` gives the basis of a sky-marginalised product: no
    ``(ra, dec)`` coordinates and no ``dOmega`` in the measure. It has its own
    name and identity, so it never pairs with a sky-resolved product.
    """
    if spin_basis not in _SUPPORTED_BASES:
        raise DataContractError(
            f"gwcat v2 spin_basis={spin_basis!r} is not supported by the Phase-1 "
            f"adapter; supported={sorted(_SUPPORTED_BASES)}"
        )
    mass_distance = ("m1_detector", "q", "luminosity_distance")
    base = mass_distance if sky_marginal else mass_distance + _SKY_COLUMNS
    sky_measure = "" if sky_marginal else " dOmega"
    if spin_basis == "chieff":
        spin = ("chi_eff",)
        spin_measure = "dchi_eff"
    elif spin_basis == "chieff_chip":
        spin = ("chi_eff", "chi_p")
        spin_measure = "dchi_eff dchi_p"
    else:
        spin = ("a1", "a2", "cos_tilt1", "cos_tilt2")
        spin_measure = "da1 da2 dcos_tilt1 dcos_tilt2"
    measure = f"dm1_detector dq ddL{sky_measure} {spin_measure}"
    if sky_marginal:
        return CoordinateBasis(
            name=f"gwcat_v2_{spin_basis}_sky_marginal",
            coordinates=base + spin,
            frame="detector_mass_luminosity_distance_sky_marginal",
            spin_parameterization=spin_basis,
            density_measure=measure,
            version="gwcat-v2-sky-marginal",
        )
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


def _sample_columns(
    f: h5py.File, spin_basis: str, *, sky_marginal: bool = False
) -> dict[str, np.ndarray]:
    required = ["m1det", "m2det", "dL"]
    if not sky_marginal:
        required.extend(_SKY_COLUMNS)
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
    }
    if not sky_marginal:
        columns["ra"] = np.asarray(f["ra"][:], dtype=float)
        columns["dec"] = np.asarray(f["dec"][:], dtype=float)
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
    """Per-event chi_eff prior ceilings a gwcat ``chieff`` PE export divided out.

    ``source_per_event`` is read verbatim. On a GW-40c (PR-B) file it holds the
    spin prior's source KIND and ``resolution_per_event`` holds the historical
    analytic/fallback/caller resolution; an older file has only the latter
    meaning, under the ``source`` name, and ``resolution_per_event`` is None.
    """
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
    sources = _strings(f.attrs.get("chi_eff_amax_source_per_event", []))
    resolution = (
        _strings(f.attrs["chi_eff_amax_resolution_per_event"])
        if "chi_eff_amax_resolution_per_event" in f.attrs
        else None
    )
    if resolution is not None and len(resolution) != nobs:
        raise DataContractError(
            f"gwcat PE chi_eff_amax_resolution_per_event has {len(resolution)} "
            f"entries, expected nobs={nobs}"
        )
    unrecognized = _strings(f.attrs.get("spin_prior_unrecognized_events", []))
    return {
        "amax_1_per_event": amax_1.tolist(),
        "amax_2_per_event": amax_2.tolist(),
        "source_per_event": sources,
        "resolution_per_event": resolution,
        "mode": _decode(f.attrs.get("chi_eff_amax_mode", "")),
        "spin_prior_unrecognized_events": unrecognized,
    }


#: PR-B per-event provenance arrays carried into the canonical metadata.
_PE_PER_EVENT_PROVENANCE = (
    "prior_source_kind_spin_per_event",
    "prior_source_kind_mass_per_event",
    "prior_source_kind_dL_per_event",
    "prior_source_label_spin_per_event",
    "dL_prior_impl_per_event",
    "sample_set_name_per_event",
)
_PE_PER_EVENT_COUNTS = (
    "n_samples_cut_by_z_max",
    "n_unique_samples_per_event",
    "n_dropped_spin_above_ceiling_per_event",
    "n_above_spin_ceiling_raw_per_event",
)
_PE_EVENT_LISTS = (
    "spin_prior_non_own_analytic_events",
    "spin_prior_assumed_events",
    "mass_prior_unverified_events",
    "sample_set_substitute_events",
)
_PE_SCALARS = (
    "chi_eff_prior_impl",
    "z_max",
    "spin_ceiling_cut_applied",
    "spin_ceiling_cut_order",
    "pe_cosmology_H0",
    "pe_cosmology_Om0",
    "population_resolver",
    "sample_set_map_sha256",
    "waveform_policy",
    "nrsur_q_rule",
    "cosmology_override_used",
)


def _json_scalar(value):
    if isinstance(value, (bytes, np.bytes_, str)):
        return _decode(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        return _decode(item) if isinstance(item, bytes) else item
    return [_json_scalar(x) for x in arr.tolist()]


def _pe_provenance(f: h5py.File, nobs: int) -> dict[str, object]:
    """PR-B provenance a v2 policy checks; absent keys are simply not recorded."""
    out: dict[str, object] = {}
    for key in _PE_PER_EVENT_PROVENANCE:
        if key in f.attrs:
            values = _strings(f.attrs[key])
            if len(values) != nobs:
                raise DataContractError(
                    f"gwcat PE {key} has {len(values)} entries, expected nobs={nobs}"
                )
            out[key] = values
    for key in _PE_PER_EVENT_COUNTS:
        if key in f.attrs:
            values = np.asarray(f.attrs[key]).reshape(-1)
            if values.size != nobs:
                raise DataContractError(
                    f"gwcat PE {key} has {values.size} entries, expected nobs={nobs}"
                )
            out[key] = [int(x) for x in values]
    for key in _PE_EVENT_LISTS:
        if key in f.attrs:
            out[key] = _strings(f.attrs[key]) if np.size(f.attrs[key]) else []
    for key in _PE_SCALARS:
        if key in f.attrs:
            out[key] = _json_scalar(f.attrs[key])
    return out


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
    record: dict[str, object] = {
        "spin_reference_amax": a_ref,
        "excluded_pdraw_sentinel": sentinel,
        "n_detected_in_export": int(pdraw.size),
        "n_zero_weight_rows_removed": int(excluded.sum()),
        "n_retained_rows": int(pdraw.size - excluded.sum()),
        "removal_rule": (
            "rows with pdraw == spin_reference_excluded_pdraw, required to be exactly "
            "the rows with a1 > a_ref or a2 > a_ref or |chi_eff| > a_ref; zero "
            "reference weight, so removal leaves sum(p_pop / pdraw) unchanged"
        ),
    }
    if "spin_reference_coverage_per_run" in f.attrs:
        record["spin_reference_coverage_per_run"] = [
            bool(x) for x in np.atleast_1d(f.attrs["spin_reference_coverage_per_run"])
        ]
    if "spin_reference_coverage_bound_per_run" in f.attrs:
        record["spin_reference_coverage_bound_per_run"] = [
            float(x)
            for x in np.atleast_1d(f.attrs["spin_reference_coverage_bound_per_run"])
        ]
    return ~excluded, record


def selection_sky_mode(f: h5py.File) -> tuple[bool, str]:
    """Decide whether a selection export is sky-marginal, and why.

    * ``sky_marginalized=True``: marginal. The file must then carry no ra/dec
      (a file that says marginalised and ships a sky is ambiguous).
    * ``sky_position_available`` False for EVERY campaign: marginal; any NaN
      ra/dec fill columns are dropped, never read.
    * ``sky_position_available`` False for SOME campaigns only: refused. Such
      a product has neither a sky-resolved density on every row nor a
      declaration that it marginalised the sky.
    * otherwise sky-resolved (ra/dec required).
    """
    marginalized = bool(f.attrs.get("sky_marginalized", False))
    available = f.attrs.get("sky_position_available")
    avail = (
        None
        if available is None or np.size(available) == 0
        else np.atleast_1d(np.asarray(available, dtype=bool))
    )
    has_sky = [name for name in _SKY_COLUMNS if name in f]
    if marginalized:
        if has_sky:
            raise DataContractError(
                f"selection declares sky_marginalized=True but carries {has_sky}; "
                "a sky-marginal product must omit the sky columns"
            )
        return True, "sky_marginalized"
    if avail is not None and not bool(avail.all()):
        if bool((~avail).all()):
            return True, "sky_position_available_false"
        raise DataContractError(
            "selection sky_position_available is False for some campaigns and True "
            f"for others ({avail.tolist()}) without sky_marginalized=True; the "
            "product is neither sky-resolved on every row nor declared sky-marginal"
        )
    return False, "sky_resolved"


def _require_aligned(record: Mapping[str, object], keys: Iterable[str], n: int, what: str):
    for key in keys:
        if key in record and len(record[key]) != n:
            raise DataContractError(
                f"cumulative-mixture attr {key} has {len(record[key])} entries, "
                f"but {what} has {n}"
            )


def _cumulative_mixture_record(
    f: h5py.File, *, n_detected: int, ndraw: int, tobs_yr: float
) -> dict[str, object] | None:
    """Validate gwcat PR-A cumulative-mixture attrs; None for other campaigns."""
    kinds = (
        _strings(f.attrs["campaign_kind_per_campaign"])
        if "campaign_kind_per_campaign" in f.attrs
        else []
    )
    flag = bool(f.attrs.get("cumulative_mixture", False))
    if not flag:
        if "cumulative_mixture" in kinds:
            raise DataContractError(
                "campaign_kind_per_campaign names a cumulative_mixture campaign but "
                "the file does not record cumulative_mixture=True"
            )
        return None
    if "cumulative_mixture" not in kinds:
        raise DataContractError(
            "cumulative_mixture=True but campaign_kind_per_campaign="
            f"{kinds} names no cumulative_mixture campaign"
        )
    n_campaigns = int(f.attrs.get("n_campaigns", len(kinds) or 1))
    if len(kinds) != n_campaigns:
        raise DataContractError(
            f"campaign_kind_per_campaign has {len(kinds)} entries, "
            f"n_campaigns={n_campaigns}"
        )
    missing = [key for key in _MIXTURE_REQUIRED_ATTRS if key not in f.attrs]
    if missing:
        raise DataContractError(
            f"cumulative_mixture=True but attr(s) {missing} are missing"
        )

    runs = _strings(f.attrs["run_labels"])
    unknown = [r for r in runs if r not in MIXTURE_RUN_LABELS]
    if unknown or len(set(runs)) != len(runs):
        raise DataContractError(
            f"cumulative-mixture run_labels {runs} contain unknown or repeated runs "
            f"(known {list(MIXTURE_RUN_LABELS)})"
        )
    record: dict[str, object] = {
        "run_labels": runs,
        "n_rows_per_run": [int(x) for x in np.atleast_1d(f.attrs["n_rows_per_run"])],
        "n_detected_per_run": [
            int(x) for x in np.atleast_1d(f.attrs["n_detected_per_run"])
        ],
        "detection_rule_per_run": _strings(f.attrs["detection_rule_per_run"]),
        "T_definition_per_run": _strings(f.attrs["T_definition_per_run"]),
        "component_of_run": _strings(f.attrs["component_of_run"]),
        "mixture_components": _strings(f.attrs["mixture_components"]),
        "N_per_component": [float(x) for x in np.atleast_1d(f.attrs["N_per_component"])],
        "T_per_component_s": [
            float(x) for x in np.atleast_1d(f.attrs["T_per_component_s"])
        ],
        "T_definition_per_component": _strings(f.attrs["T_definition_per_component"]),
    }
    if "n_passing_detection_rule_per_run" in f.attrs:
        record["n_passing_detection_rule_per_run"] = [
            int(x) for x in np.atleast_1d(f.attrs["n_passing_detection_rule_per_run"])
        ]
    if "z_draw_max_per_run" in f.attrs:
        record["z_draw_max_per_run"] = [
            float(x) for x in np.atleast_1d(f.attrs["z_draw_max_per_run"])
        ]
    _require_aligned(
        record,
        (
            "n_rows_per_run",
            "n_detected_per_run",
            "detection_rule_per_run",
            "T_definition_per_run",
            "component_of_run",
            "n_passing_detection_rule_per_run",
            "z_draw_max_per_run",
        ),
        len(runs),
        "run_labels",
    )
    comps = record["mixture_components"]
    if len(set(comps)) != len(comps):
        raise DataContractError(f"mixture_components {comps} repeat a component")
    _require_aligned(
        record,
        ("N_per_component", "T_per_component_s", "T_definition_per_component"),
        len(comps),
        "mixture_components",
    )

    # run -> component -> exposure definition: one consistent mapping.
    for run, comp in zip(runs, record["component_of_run"]):
        if comp not in comps:
            raise DataContractError(
                f"component_of_run maps {run} to {comp!r}, not a mixture component"
            )
        if MIXTURE_COMPONENT_OF_RUN[run] != comp:
            raise DataContractError(
                f"component_of_run maps {run} to {comp!r}; the LVK cumulative "
                f"mixture defines it as {MIXTURE_COMPONENT_OF_RUN[run]!r}"
            )
    if "mixture_component_runs" in f.attrs:
        declared = json.loads(_decode(f.attrs["mixture_component_runs"]))
        for run, comp in zip(runs, record["component_of_run"]):
            if run not in declared.get(comp, []):
                raise DataContractError(
                    f"mixture_component_runs {declared} does not list {run} under "
                    f"{comp}, contradicting component_of_run"
                )
    tdef_of = dict(zip(comps, record["T_definition_per_component"]))
    bad_tdef = sorted(set(tdef_of.values()) - MIXTURE_T_DEFINITIONS)
    if bad_tdef:
        raise DataContractError(
            f"unknown T_definition_per_component value(s) {bad_tdef}; known "
            f"{sorted(MIXTURE_T_DEFINITIONS)}"
        )
    for run, comp, tdef in zip(
        runs, record["component_of_run"], record["T_definition_per_run"]
    ):
        if tdef != tdef_of[comp]:
            raise DataContractError(
                f"T_definition_per_run gives {run} {tdef!r} but its component {comp} "
                f"is defined as {tdef_of[comp]!r}"
            )

    # Detection rule per run: semianalytic SNR on O1/O2, own-run FAR elsewhere.
    semianalytic = (
        _strings(f.attrs["semianalytic_components"])
        if "semianalytic_components" in f.attrs
        else ["O1", "O2"]
    )
    snr_col = _decode(f.attrs.get("significance_snr_column", ""))
    snr_thr = float(f.attrs.get("significance_snr_threshold", np.nan))
    far_thr = float(
        f.attrs.get("significance_far_threshold", f.attrs.get("far_threshold", np.nan))
    )
    far_columns = (
        set(_strings(f.attrs["significance_columns"]))
        if "significance_columns" in f.attrs
        else None
    )
    for run, comp, rule in zip(
        runs, record["component_of_run"], record["detection_rule_per_run"]
    ):
        if comp in semianalytic:
            if tdef_of[comp] != "coincident_livetime_semianalytic":
                raise DataContractError(
                    f"semianalytic component {comp} has T definition "
                    f"{tdef_of[comp]!r}, not coincident_livetime_semianalytic"
                )
            match = _SNR_RULE.match(rule)
            if match is None:
                raise DataContractError(
                    f"{run} is semianalytic but its detection rule is {rule!r}: its "
                    "rows carry no search FAR, so a rule other than the semianalytic "
                    "SNR threshold drops its detections while its exposure stays in "
                    "ndraw and T_obs (the O1/O2 exposure bias)"
                )
            if match["col"] != snr_col or not np.isclose(float(match["thr"]), snr_thr):
                raise DataContractError(
                    f"{run} detection rule {rule!r} disagrees with the recorded "
                    f"significance_snr_column={snr_col!r} / "
                    f"significance_snr_threshold={snr_thr}"
                )
        else:
            match = _FAR_RULE.match(rule)
            if match is None:
                raise DataContractError(
                    f"{run} detection rule {rule!r} is not a minimum-search-FAR rule"
                )
            if not np.isclose(float(match["thr"]), far_thr):
                raise DataContractError(
                    f"{run} detection rule {rule!r} disagrees with the recorded FAR "
                    f"threshold {far_thr}"
                )
            cols = [c.strip() for c in match["cols"].split(",") if c.strip()]
            prefixes = (run.lower() + "_", run.lower().rstrip("ab") + "_")
            foreign = [c for c in cols if not c.startswith(prefixes)]
            if not cols or foreign:
                raise DataContractError(
                    f"{run} detection rule {rule!r} uses search column(s) {foreign} "
                    "of another run (each run must be detected by its own searches)"
                )
            if far_columns is not None and not set(cols) <= far_columns:
                raise DataContractError(
                    f"{run} detection rule uses columns {sorted(set(cols) - far_columns)} "
                    "absent from significance_columns"
                )

    # Counts: detected <= passing <= rows; the mixture's rows add up.
    for i, run in enumerate(runs):
        n_det = record["n_detected_per_run"][i]
        n_rows = record["n_rows_per_run"][i]
        n_pass = record.get("n_passing_detection_rule_per_run", [n_det] * len(runs))[i]
        if not 0 <= n_det <= n_pass <= n_rows:
            raise DataContractError(
                f"{run}: n_detected {n_det}, n_passing {n_pass}, n_rows {n_rows} are "
                "not ordered 0 <= detected <= passing <= rows"
            )
    n_mix = int(f.attrs.get("n_detected_mixture", n_detected if n_campaigns == 1 else -1))
    if n_mix < 0:
        raise DataContractError(
            "a multi-campaign cumulative-mixture product must record n_detected_mixture"
        )
    if sum(record["n_detected_per_run"]) != n_mix or n_mix > n_detected or (
        n_campaigns == 1 and n_mix != n_detected
    ):
        raise DataContractError(
            f"sum(n_detected_per_run)={sum(record['n_detected_per_run'])}, "
            f"n_detected_mixture={n_mix}, n_detected={n_detected}, "
            f"n_campaigns={n_campaigns} do not add up"
        )
    record["n_detected_mixture"] = n_mix
    record["n_campaigns"] = n_campaigns

    # Exposure bookkeeping: the per-component (N, T) add up to ndraw, T_obs.
    status = _decode(f.attrs.get("mixture_bookkeeping_status", ""))
    record["mixture_bookkeeping_status"] = status
    n_arr = np.asarray(record["N_per_component"], dtype=float)
    t_arr = np.asarray(record["T_per_component_s"], dtype=float)
    derived = bool(np.all(np.isfinite(n_arr)) and np.all(np.isfinite(t_arr)))
    if status == "derived_from_weights_and_anchors" and not derived:
        raise DataContractError(
            "mixture_bookkeeping_status says derived but N/T_per_component "
            "are not finite"
        )
    if derived:
        if np.any(n_arr <= 0) or np.any(t_arr <= 0):
            raise DataContractError(
                f"N_per_component {n_arr.tolist()} / T_per_component_s "
                f"{t_arr.tolist()} must be positive"
            )
        if n_campaigns == 1:
            if not np.isclose(n_arr.sum(), float(ndraw), rtol=1e-9, atol=1e-3):
                raise DataContractError(
                    f"sum(N_per_component)={n_arr.sum()!r} != ndraw={ndraw}"
                )
            t_sum_yr = t_arr.sum() / _SECONDS_PER_JULIAN_YEAR
            if not np.isclose(t_sum_yr, float(tobs_yr), rtol=1e-9, atol=0.0):
                raise DataContractError(
                    f"sum(T_per_component_s)={t_arr.sum()!r} s = {t_sum_yr!r} yr != "
                    f"T_obs_yr={tobs_yr!r}"
                )
        residual = f.attrs.get("N_integrality_residual_per_semianalytic_component")
        if residual is not None:
            record["N_integrality_residual_per_semianalytic_component"] = [
                float(x) for x in np.atleast_1d(residual)
            ]
    record["bookkeeping_derived"] = derived
    return record


def load_pe(path: str | Path, *, sky_marginal: bool = False) -> PosteriorCatalog:
    """Load a gwcat-pe-2.0/2.1 export without changing its density basis.

    ``sky_marginal=True`` loads it in the sky-marginal basis (used when the
    paired selection is sky-marginal): p_pe has no sky density, so dropping
    the ra/dec columns is exact.
    """

    with h5py.File(path, "r") as f:
        fmt = _decode(_attr(f, "format_version"))
        if fmt not in _PE_FORMATS:
            raise DataContractError(
                f"expected a gwcat v2 PE export, found format_version={fmt!r}"
            )
        spin_basis = _decode(_attr(f, "spin_basis"))
        basis = basis_for_spin(spin_basis, sky_marginal=sky_marginal)
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
        samples = _sample_columns(f, spin_basis, sky_marginal=sky_marginal)
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
            "sky_marginal": bool(sky_marginal),
        }
        if sky_marginal:
            metadata["sky_columns_dropped"] = [c for c in _SKY_COLUMNS if c in f]
        provenance = _pe_provenance(f, nobs)
        if provenance:
            metadata["prior_provenance"] = provenance
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
    A sky-marginal export (see :func:`selection_sky_mode`) is loaded in the
    sky-marginal basis, and a cumulative-mixture export has its per-run and
    per-component records validated and carried in the metadata.
    """

    with h5py.File(path, "r") as f:
        fmt = _decode(_attr(f, "format_version"))
        if fmt not in _SEL_FORMATS:
            raise DataContractError(
                f"expected a gwcat v2 selection export, found format_version={fmt!r}"
            )
        spin_basis = _decode(_attr(f, "spin_basis"))
        density_basis = REFERENCE_SELECTION_BASES.get(spin_basis, spin_basis)
        sky_marginal, sky_reason = selection_sky_mode(f)
        basis = basis_for_spin(density_basis, sky_marginal=sky_marginal)
        if "pdraw" not in f:
            raise DataContractError("gwcat v2 selection product is missing pdraw")
        pdraw = np.asarray(f["pdraw"][:], dtype=float)
        n_detected = int(_attr(f, "n_detected"))
        if pdraw.size != n_detected:
            raise DataContractError(
                f"gwcat selection pdraw length={pdraw.size}, "
                f"n_detected={n_detected}"
            )
        samples = _sample_columns(f, density_basis, sky_marginal=sky_marginal)
        if any(len(x) != n_detected for x in samples.values()):
            bad = {k: len(v) for k, v in samples.items() if len(v) != n_detected}
            raise DataContractError(
                f"gwcat selection sample-column length mismatch; "
                f"expected {n_detected}, got {bad}"
            )
        ndraw = int(_attr(f, "ndraw"))
        tobs = float(_attr(f, "T_obs_yr"))
        mixture = _cumulative_mixture_record(
            f, n_detected=n_detected, ndraw=ndraw, tobs_yr=tobs
        )
        reference = None
        if spin_basis in REFERENCE_SELECTION_BASES:
            keep, reference = _reference_support(f, pdraw, samples)
            pdraw = pdraw[keep]
            samples = {name: values[keep] for name, values in samples.items()}
        log_draw = _positive_log(pdraw, name="pdraw")
        n_selected = int(log_draw.size)
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
            "sky_marginal": bool(sky_marginal),
            "sky_mode_reason": sky_reason,
        }
        for key in ("chi_eff_prior_impl", "z_max", "cosmology_override_used"):
            if key in f.attrs:
                metadata[key] = _json_scalar(f.attrs[key])
        if "campaign_kind_per_campaign" in f.attrs:
            metadata["campaign_kind_per_campaign"] = _strings(
                f.attrs["campaign_kind_per_campaign"]
            )
        if mixture is not None:
            metadata["cumulative_mixture"] = mixture
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


def load_spin_prior_allow_list(spec) -> dict[str, str | None] | None:
    """Normalise a spin-prior allow-list to ``{event_name: kind_or_None}``.

    Mirrors gwcat ``export.validate.load_spin_prior_allow_list`` (GW-40d):
    None (no list); a mapping ``{event: kind}`` or ``{event: {"kind": kind}}``
    (the event is allowed ONLY with that kind; ``None``/``""``/``"*"`` means
    any kind); an iterable of names (any kind); or a path to a ``.json`` file
    holding either, or a text file with ``NAME [KIND]`` per line.
    """
    if spec is None:
        return None
    if isinstance(spec, (str, os.PathLike)):
        path = os.fspath(spec)
        text = Path(path).read_text()
        if path.lower().endswith(".json"):
            return load_spin_prior_allow_list(json.loads(text))
        out: dict[str, str | None] = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            bits = line.split()
            out[bits[0]] = bits[1] if len(bits) > 1 else None
        return out
    if isinstance(spec, Mapping):
        out = {}
        for key, value in spec.items():
            if isinstance(value, Mapping):
                value = value.get("kind")
            out[str(key)] = None if value in (None, "", "*") else str(value)
        return out
    return {str(name): None for name in spec}


def validate_spin_prior_sources(
    pe: PosteriorCatalog,
    *,
    spin_prior_allow_list=None,
    require_exact_allow_list: bool = False,
) -> dict[str, object]:
    """Accept non-``own_analytic`` spin priors only through the allow-list.

    Requires a GW-40c (PR-B) PE file: ``chi_eff_amax_resolution_per_event``
    all ``analytic`` (every ceiling resolved from the event's own store
    record, not a fallback or caller override), ``chi_eff_amax_source_per_event``
    holding source kinds that equal ``prior_source_kind_spin_per_event``, and
    every event whose kind is not ``own_analytic`` named on the allow-list
    with that same kind (or with any kind, if the list gives none).

    ``require_exact_allow_list`` additionally requires the allow-list to name
    exactly the non-``own_analytic`` events (BUILD_PLAN G6: they "are exactly
    the OD-7 allow-list"), so a stale entry fails too.
    """
    allow = load_spin_prior_allow_list(spin_prior_allow_list) or {}
    amax = pe.metadata.get("chi_eff_amax")
    if amax is None:
        raise DataContractError(
            "chieff PE export records no per-event chi_eff prior ceilings "
            "(chi_eff_amax_1/2_per_event); cannot verify the spin-prior sources"
        )
    amax = dict(amax)
    resolution = amax.get("resolution_per_event")
    sources = list(amax.get("source_per_event") or [])
    provenance = dict(pe.metadata.get("prior_provenance") or {})
    kinds = provenance.get("prior_source_kind_spin_per_event")
    if resolution is None or kinds is None:
        raise DataContractError(
            "the PE export predates gwcat GW-40c: it records no "
            "chi_eff_amax_resolution_per_event / prior_source_kind_spin_per_event, "
            "so whether each event's spin prior is its own declared U(0, a_ref) "
            "cannot be checked. Re-export the PE file with gwcat >= GW-40c."
        )
    names = list(pe.event_names)
    if len(sources) != len(names) or len(kinds) != len(names):
        raise DataContractError(
            f"chi_eff_amax_source_per_event ({len(sources)}) / "
            f"prior_source_kind_spin_per_event ({len(kinds)}) do not have one "
            f"entry per event ({len(names)})"
        )
    not_resolved = sorted({r for r in resolution if r != "analytic"})
    if not_resolved:
        bad = [n for n, r in zip(names, resolution) if r != "analytic"]
        raise DataContractError(
            "every PE chi_eff ceiling must be resolved from the event's own stored "
            f"spin prior (resolution 'analytic'); got {not_resolved} for {bad[:10]}"
        )
    mismatch = [
        (n, s, k) for n, s, k in zip(names, sources, kinds) if s != k
    ]
    if mismatch:
        raise DataContractError(
            "chi_eff_amax_source_per_event disagrees with "
            f"prior_source_kind_spin_per_event for {mismatch[:10]}; under GW-40c the "
            "ceiling's source is the spin prior's source kind"
        )
    unknown = sorted(set(kinds) - PRIOR_SOURCE_KINDS)
    if unknown:
        raise DataContractError(
            f"unknown spin prior source kind(s) {unknown}; known "
            f"{sorted(PRIOR_SOURCE_KINDS)}"
        )

    non_own = [(n, k) for n, k in zip(names, kinds) if k != SPIN_PRIOR_SOURCE_OK]
    offenders = [(n, k) for n, k in non_own if n not in allow]
    wrong_kind = [
        (n, k, allow[n])
        for n, k in non_own
        if n in allow and allow[n] is not None and allow[n] != k
    ]
    if offenders or wrong_kind:
        by_kind: dict[str, list[str]] = {}
        for n, k in offenders:
            by_kind.setdefault(k, []).append(n)
        parts = [
            f"{k} [{len(v)}]: {v[:10]}" + ("" if len(v) <= 10 else f" (+{len(v) - 10} more)")
            for k, v in sorted(by_kind.items())
        ]
        if wrong_kind:
            parts.append(
                "allow-listed with a DIFFERENT kind: "
                + ", ".join(f"{n} is {k}, list says {w}" for n, k, w in wrong_kind[:10])
            )
        raise DataContractError(
            f"{len(offenders) + len(wrong_kind)} PE event(s) divide out a spin prior "
            "that is NOT their label's own analytic declaration and are not on the "
            "spin-prior allow-list (BUILD_PLAN OD-7): "
            + "; ".join(parts)
            + ". The chieff_reference pair assumes each event's spin prior IS the "
            "declared U(0, a_ref) reference; for these that is an assumption, so it "
            "must be approved explicitly (spin_prior_allow_list) or the events dropped."
        )

    derived_non_own = [n for n, _ in non_own]
    recorded_non_own = provenance.get("spin_prior_non_own_analytic_events")
    if recorded_non_own is not None and sorted(recorded_non_own) != sorted(derived_non_own):
        raise DataContractError(
            "spin_prior_non_own_analytic_events disagrees with the per-event kinds"
        )
    derived_assumed = [n for n, k in zip(names, kinds) if k == "assumed_default"]
    recorded_assumed = provenance.get("spin_prior_assumed_events")
    if recorded_assumed is not None and sorted(recorded_assumed) != sorted(derived_assumed):
        raise DataContractError(
            "spin_prior_assumed_events disagrees with the per-event kinds"
        )
    unused = sorted(set(allow) - set(derived_non_own))
    if require_exact_allow_list and unused:
        raise DataContractError(
            f"the spin-prior allow-list names {len(unused)} event(s) that are not "
            f"non-own_analytic events of this PE export: {unused[:10]}; the list must "
            "equal the non-own_analytic set exactly"
        )
    counts: dict[str, int] = {}
    for k in kinds:
        counts[k] = counts.get(k, 0) + 1
    mass_kinds = provenance.get("prior_source_kind_mass_per_event")
    mass_counts: dict[str, int] | None = None
    if mass_kinds is not None:
        mass_counts = {}
        for k in mass_kinds:
            mass_counts[k] = mass_counts.get(k, 0) + 1
    return {
        "spin_prior_source_kind_counts": dict(sorted(counts.items())),
        "n_non_own_analytic_allow_listed": len(non_own),
        "allow_list_size": len(allow),
        "allow_list_unused_entries": unused,
        "mass_prior_source_kind_counts": (
            None if mass_counts is None else dict(sorted(mass_counts.items()))
        ),
        "mass_prior_assumed_default_events": (
            None
            if mass_kinds is None
            else [n for n, k in zip(names, mass_kinds) if k == "assumed_default"]
        ),
    }


def validate_reference_pairing(
    pe: PosteriorCatalog,
    selection: SelectionCatalog,
    *,
    spin_prior_allow_list=None,
    require_exact_allow_list: bool = False,
) -> dict[str, object] | None:
    """Require one reference ceiling across a (chieff, chieff_reference) pair.

    The reference selection density was reweighted TO the prior the ``chieff``
    PE export divides OUT, so the two ceilings are the same object: any
    difference leaves an uncancelled chi_eff-dependent factor in every event
    weight. Each PE ceiling must also be the event's own sampling-prior
    ceiling (not a fallback or caller override) with a recognized
    uniform-magnitude/isotropic spin prior.

    On a GW-40c (PR-B) PE file the ceiling's source is the spin prior's
    source kind; events whose kind is not ``own_analytic`` are accepted only
    through ``spin_prior_allow_list`` (see :func:`validate_spin_prior_sources`).
    A pre-GW-40c file keeps the legacy rule: every source must be
    ``analytic``. Returns None for other pairs.
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
    spin_sources: dict[str, object] | None = None
    if amax.get("resolution_per_event") is None:
        # Pre-GW-40c file: ``source`` is still the analytic/fallback/caller
        # resolution, and a non-own prior is not recorded at all.
        if spin_prior_allow_list is not None:
            raise DataContractError(
                "a spin-prior allow-list was given but the PE export predates gwcat "
                "GW-40c and records no per-event spin prior source kinds to check "
                "it against"
            )
        sources = sorted(set(amax["source_per_event"]))
        if len(amax["source_per_event"]) != pe.n_events or sources != ["analytic"]:
            raise DataContractError(
                "every PE chi_eff ceiling must come from the event's own sampling "
                f"prior (source 'analytic'); got sources={sources}"
            )
    else:
        spin_sources = validate_spin_prior_sources(
            pe,
            spin_prior_allow_list=spin_prior_allow_list,
            require_exact_allow_list=require_exact_allow_list,
        )
    if amax["spin_prior_unrecognized_events"]:
        raise DataContractError(
            "PE events without a recognized uniform-magnitude/isotropic spin prior: "
            f"{amax['spin_prior_unrecognized_events']}"
        )
    out: dict[str, object] = {
        "pe_spin_basis": pe_basis,
        "selection_spin_basis": selection_basis,
        "spin_reference_amax": a_ref,
        "pe_chi_eff_amax_all_equal_reference": True,
        "n_selection_zero_weight_rows_removed": int(
            dict(selection.metadata["spin_reference"])["n_zero_weight_rows_removed"]
        ),
    }
    if spin_sources is not None:
        out["spin_prior_sources"] = spin_sources
    return out


def validate_exact_priors(
    pe: PosteriorCatalog, selection: SelectionCatalog
) -> dict[str, object]:
    """Require gwcat's exact prior implementations on both halves (GW-41).

    PE: ``chi_eff_prior_impl == 'exact'`` and every ``dL_prior_impl_per_event``
    in ``{'exact', 'analytic'}``. Selection: ``chi_eff_prior_impl == 'exact'``
    when its basis carries a chi_eff prior. A legacy grid/bilby-interpolated
    denominator, or one that does not record its implementation, is refused.
    """
    provenance = dict(pe.metadata.get("prior_provenance") or {})
    pe_chi = provenance.get("chi_eff_prior_impl")
    if pe_chi != CHI_EFF_PRIOR_IMPL_EXACT:
        raise DataContractError(
            f"PE chi_eff_prior_impl={pe_chi!r}; v2 requires "
            f"{CHI_EFF_PRIOR_IMPL_EXACT!r} (gwcat GW-41 exact p_iso)"
        )
    dl = provenance.get("dL_prior_impl_per_event")
    if dl is None or len(dl) != pe.n_events:
        raise DataContractError(
            "PE records no dL_prior_impl_per_event for every event; v2 requires the "
            "exact UniformSourceFrame distance prior on every event"
        )
    bad = [(n, v) for n, v in zip(pe.event_names, dl) if v not in DL_PRIOR_IMPLS_EXACT]
    if bad:
        raise DataContractError(
            f"{len(bad)} PE event(s) have a non-exact distance prior "
            f"(dL_prior_impl_per_event), e.g. {bad[:5]}; v2 requires one of "
            f"{sorted(DL_PRIOR_IMPLS_EXACT)}"
        )
    selection_basis = str(selection.metadata.get("spin_basis", ""))
    sel_chi = selection.metadata.get("chi_eff_prior_impl")
    carries_chi_eff_prior = selection_basis in {"chieff", "chieff_chip", "chieff_reference"}
    if carries_chi_eff_prior and sel_chi != CHI_EFF_PRIOR_IMPL_EXACT:
        raise DataContractError(
            f"selection chi_eff_prior_impl={sel_chi!r}; v2 requires "
            f"{CHI_EFF_PRIOR_IMPL_EXACT!r} so p_pe and pdraw carry the same exact "
            "chi_eff prior"
        )
    counts: dict[str, int] = {}
    for v in dl:
        counts[v] = counts.get(v, 0) + 1
    return {
        "pe_chi_eff_prior_impl": pe_chi,
        "selection_chi_eff_prior_impl": sel_chi,
        "pe_dL_prior_impl_counts": dict(sorted(counts.items())),
    }


def load_pair(
    pe_path: str | Path,
    selection_path: str | Path,
    *,
    required_coordinates: tuple[str, ...] | None = None,
    spin_prior_allow_list=None,
) -> tuple[PosteriorCatalog, SelectionCatalog]:
    """Load and cross-validate one PE/selection pair.

    The selection's sky declaration decides the basis of both halves.
    """
    selection = load_selection(selection_path)
    pe = load_pe(pe_path, sky_marginal=bool(selection.metadata.get("sky_marginal")))
    validate_reference_pairing(pe, selection, spin_prior_allow_list=spin_prior_allow_list)
    validate_pair(pe, selection, required_coordinates)
    return pe, selection
