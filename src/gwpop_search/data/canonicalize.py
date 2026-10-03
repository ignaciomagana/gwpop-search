"""Idempotent conversion of validated gwcat-v2 exports to canonical HDF5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .adapters.gwcat_v2 import (
    REFERENCE_SELECTION_BASES,
    load_pe,
    load_selection,
    load_spin_prior_allow_list,
    validate_reference_pairing,
)
from .pair import validate_pair
from .v2_policy import GwcatV2DataPolicy, evaluate_gwcat_v2_policy

#: 2.0 (v2 adapter): the sky-marginal basis, the cumulative-mixture record,
#: the GW-40c spin-prior source check and the optional v2 data policy (whose
#: hash the report records).
_FORMAT_VERSION = "gwpop-search-gwcat-canonicalization-2.0"
# 1.0 reports predate an explicit selection-basis requirement: both halves
# shared ``required_spin_basis``. 1.1 reports predate the v2 policy and the
# sky-marginal basis; both are accepted only for an identical policy-free rerun.
_LEGACY_FORMAT_VERSIONS = {
    "gwpop-search-gwcat-canonicalization-1.0",
    "gwpop-search-gwcat-canonicalization-1.1",
}
_PE_SPIN_BASES = {"chieff", "chieff_chip", "component"}


def _json_default(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _required_selection_spin_basis(
    required_spin_basis: str,
    required_selection_spin_basis: str | None,
) -> str:
    """Selection basis that must accompany the PE basis (default: the same one)."""
    selection = (
        required_spin_basis
        if required_selection_spin_basis is None
        else required_selection_spin_basis
    )
    if (
        selection != required_spin_basis
        and REFERENCE_SELECTION_BASES.get(selection) != required_spin_basis
    ):
        raise ValueError(
            "required_selection_spin_basis must equal required_spin_basis or be a "
            f"reference basis paired with it {sorted(REFERENCE_SELECTION_BASES.items())}; "
            f"got pe={required_spin_basis!r}, selection={selection!r}"
        )
    return selection


def _validate_existing_outputs(
    report_path: Path,
    *,
    source_pe: dict[str, object],
    source_selection: dict[str, object],
    required_spin_basis: str,
    required_selection_spin_basis: str,
    policy_hash: str | None,
    spin_prior_allow_list: dict[str, str | None] | None,
) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    format_version = report.get("format_version")
    if format_version == _FORMAT_VERSION:
        recorded_selection = report.get("required_selection_spin_basis")
        if report.get("v2_policy_hash") != policy_hash:
            raise ValueError(
                "existing canonicalization was checked against a different v2 data "
                f"policy (recorded {report.get('v2_policy_hash')!r}, "
                f"requested {policy_hash!r})"
            )
        if report.get("spin_prior_allow_list") != spin_prior_allow_list:
            raise ValueError(
                "existing canonicalization used a different spin-prior allow-list"
            )
    elif format_version in _LEGACY_FORMAT_VERSIONS:
        if policy_hash is not None or spin_prior_allow_list is not None:
            raise ValueError(
                f"existing {format_version} canonicalization predates the v2 data "
                "policy and cannot vouch for it; canonicalize into a fresh directory"
            )
        recorded_selection = (
            report.get("required_selection_spin_basis")
            if format_version.endswith("-1.1")
            else report.get("required_spin_basis")
        )
    else:
        raise ValueError("existing canonicalization report has unsupported format")
    if report.get("required_spin_basis") != required_spin_basis:
        raise ValueError("existing canonicalization spin-basis requirement differs")
    if recorded_selection != required_selection_spin_basis:
        raise ValueError(
            "existing canonicalization selection spin-basis requirement differs"
        )
    if report.get("source_pe") != source_pe:
        raise ValueError("existing canonicalization used a different PE export")
    if report.get("source_selection") != source_selection:
        raise ValueError(
            "existing canonicalization used a different selection export"
        )

    for key in ("canonical_pe", "canonical_selection"):
        entry = dict(report[key])
        path = Path(str(entry["path"]))
        if not path.is_file():
            raise ValueError(
                f"canonicalization report references missing output {path}"
            )
        actual = _file_entry(path)
        if actual != entry:
            raise ValueError(
                f"canonical output checksum/size/path mismatch for {path}"
            )
    return report


def canonicalize_gwcat_v2_pair(
    pe_export: str | Path,
    selection_export: str | Path,
    output_dir: str | Path,
    *,
    required_spin_basis: str,
    required_selection_spin_basis: str | None = None,
    policy: GwcatV2DataPolicy | None = None,
    spin_prior_allow_list=None,
) -> dict[str, object]:
    """Convert one reviewed gwcat-v2 pair without reconstructing denominators.

    ``required_selection_spin_basis`` defaults to ``required_spin_basis``. The
    only permitted difference is a reference selection basis paired with its PE
    basis (``chieff_reference`` with ``chieff``), whose ceiling equality is
    verified by the adapter and recorded in the report.

    The selection's sky declaration decides the basis of both halves (a
    sky-marginal selection gives a sky-marginal pair). A ``chieff_reference``
    selection has its declared out-of-reference sentinel rows dropped
    (BUILD_PLAN OD-9) and the count recorded.

    ``policy`` (a :class:`GwcatV2DataPolicy`) runs every v2 check and supplies
    the OD-7 spin-prior allow-list; its hash is recorded, and a rerun must
    name the same policy. Without a policy, ``spin_prior_allow_list`` may be
    given directly; with neither, any non-``own_analytic`` spin prior in a
    reference pair is refused.
    """
    if policy is not None and spin_prior_allow_list is not None:
        raise ValueError(
            "pass the spin-prior allow-list through the v2 policy, not separately"
        )
    policy_hash = None if policy is None else policy.policy_hash
    allow_list = (
        dict(policy.spin_prior_allow_list)
        if policy is not None
        else load_spin_prior_allow_list(spin_prior_allow_list)
    )
    if allow_list is not None:
        allow_list = dict(sorted(allow_list.items()))
    pe_export = Path(pe_export).resolve()
    selection_export = Path(selection_export).resolve()
    output_dir = Path(output_dir).resolve()
    if not pe_export.is_file():
        raise FileNotFoundError(pe_export)
    if not selection_export.is_file():
        raise FileNotFoundError(selection_export)
    if required_spin_basis not in _PE_SPIN_BASES:
        raise ValueError(
            "required_spin_basis must be chieff, chieff_chip, or component"
        )
    required_selection = _required_selection_spin_basis(
        required_spin_basis, required_selection_spin_basis
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pe_output = output_dir / "pe.h5"
    selection_output = output_dir / "selection.h5"
    report_path = output_dir / "canonicalization_report.json"

    source_pe = _file_entry(pe_export)
    source_selection = _file_entry(selection_export)

    existing = [
        pe_output.exists(),
        selection_output.exists(),
        report_path.exists(),
    ]
    if any(existing):
        if not all(existing):
            raise ValueError(
                "canonicalization output directory contains a partial prior run"
            )
        return _validate_existing_outputs(
            report_path,
            source_pe=source_pe,
            source_selection=source_selection,
            required_spin_basis=required_spin_basis,
            required_selection_spin_basis=required_selection,
            policy_hash=policy_hash,
            spin_prior_allow_list=allow_list,
        )

    selection = load_selection(selection_export)
    posterior = load_pe(
        pe_export, sky_marginal=bool(selection.metadata.get("sky_marginal"))
    )
    pe_spin = str(posterior.metadata.get("spin_basis", ""))
    selection_spin = str(selection.metadata.get("spin_basis", ""))
    if pe_spin != required_spin_basis or selection_spin != required_selection:
        raise ValueError(
            "gwcat export spin basis does not match explicit requirement: "
            f"required={required_spin_basis!r}, "
            f"required_selection={required_selection!r}, "
            f"pe={pe_spin!r}, selection={selection_spin!r}"
        )
    policy_checks = None
    if policy is not None:
        policy_checks = evaluate_gwcat_v2_policy(posterior, selection, policy)
        reference_pairing = policy_checks["reference_pairing"]
    else:
        reference_pairing = validate_reference_pairing(
            posterior, selection, spin_prior_allow_list=allow_list
        )
    validate_pair(posterior, selection)
    selection_contract = (
        "gwcat exported estimator-ready pdraw, stored as "
        "log_draw_density with no second ndraw/time/campaign factor"
    )
    if reference_pairing is not None:
        selection_contract += (
            f"; {required_selection}: component draw reweighted to the declared "
            "isotropic uniform-magnitude reference with a_ref="
            f"{reference_pairing['spin_reference_amax']} (equal to every PE "
            "chi_eff prior ceiling); its "
            f"{reference_pairing['n_selection_zero_weight_rows_removed']} declared "
            "zero-weight rows outside the reference support were removed"
        )

    posterior.to_hdf5(pe_output)
    selection.to_hdf5(selection_output)

    report = {
        "format_version": _FORMAT_VERSION,
        "required_spin_basis": required_spin_basis,
        "required_selection_spin_basis": required_selection,
        "selection_reference_pairing": reference_pairing,
        "source_pe": source_pe,
        "source_selection": source_selection,
        "canonical_pe": _file_entry(pe_output),
        "canonical_selection": _file_entry(selection_output),
        "coordinate_basis": posterior.basis.to_dict(),
        "coordinate_basis_identity": posterior.basis.identity,
        "event_names": list(posterior.event_names),
        "n_events": int(posterior.n_events),
        "n_pe_samples": int(posterior.n_samples_total),
        "n_selected_injections": int(selection.n_selected),
        "selection_mode": selection.mode.value,
        "estimator_semantics": selection.estimator_semantics,
        "pe_adapter_metadata": dict(posterior.metadata),
        "selection_adapter_metadata": dict(selection.metadata),
        "denominator_contract": {
            "pe": "gwcat exported p_pe, stored as log_ref_density",
            "selection": selection_contract,
        },
        "sky_marginal": bool(selection.metadata.get("sky_marginal")),
        "v2_policy": None if policy is None else policy.to_dict(),
        "v2_policy_hash": policy_hash,
        "spin_prior_allow_list": allow_list,
        "v2_policy_checks": policy_checks,
    }
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, default=_json_default)
    )
    return json.loads(report_path.read_text())
