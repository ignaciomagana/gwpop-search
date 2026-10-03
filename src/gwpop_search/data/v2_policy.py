"""Declared data policy for the GWTC-5 v2 gwcat pair (BUILD_PLAN Sections 4.4, 10-12).

A :class:`GwcatV2DataPolicy` names every operator decision the adapter must
enforce on a v2 PE/selection pair, so canonicalisation can refuse a pair that
violates any of them and the canonical report can record which policy (by
hash) the data passed:

* OD-7: the spin-prior allow-list (event -> approved non-``own_analytic``
  kind); the non-``own_analytic`` events must equal it exactly (G6).
* GW-41 / Section 11: exact chi_eff and distance priors on both halves.
* OD-6 / G17: ``z_max`` recorded identically on both halves and equal to the
  policy value; exported samples inside it; events with more than
  ``z_fraction_threshold`` of their PE mass above ``z_max`` must be on the
  Section 11 allow-list.
* PR-A: a sky-marginal, cumulative-mixture selection with derived per-component
  exposure bookkeeping.
* OD-9: the reference-basis sentinel rows are dropped (always done by the
  adapter; the policy requires a reference pair so that it applies).

The policy is data, not code: it is loaded from JSON, and its ``sources``
field records where each list came from (path and sha256).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .adapters.gwcat_v2 import (
    load_spin_prior_allow_list,
    validate_exact_priors,
    validate_reference_pairing,
)
from .posterior import PosteriorCatalog
from .schema import DataContractError
from .selection import SelectionCatalog

POLICY_FORMAT_VERSION = "gwpop-search-gwcat-v2-data-policy-1.0"
_C_KM_S = 299792.458


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class GwcatV2DataPolicy:
    """Operator-declared checks for one v2 gwcat pair (see module docstring)."""

    policy_id: str
    z_max: float
    spin_prior_allow_list: Mapping[str, str | None]
    z_allow_list: Mapping[str, str] = field(default_factory=dict)
    z_fraction_threshold: float = 0.01
    #: Per-event fraction of the RAW PE pool above z_max at the population
    #: cosmology, measured from the raw files by the provenance gate (G17).
    #: The export records only the count it cut, not the raw pool size, so
    #: without these the check falls back to a conservative upper bound.
    z_fraction_above_zmax: Mapping[str, float] | None = None
    require_exact_priors: bool = True
    require_sky_marginal: bool = True
    require_cumulative_mixture: bool = True
    require_reference_pair: bool = True
    require_exact_spin_allow_list: bool = True
    population_cosmology: Mapping[str, float] = field(
        default_factory=lambda: {"H0": 67.74, "Om0": 0.3089}
    )
    status: str = "staging"
    sources: Mapping[str, object] = field(default_factory=dict)
    format_version: str = POLICY_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != POLICY_FORMAT_VERSION:
            raise DataContractError(
                f"unsupported v2 data policy format {self.format_version!r}; "
                f"expected {POLICY_FORMAT_VERSION!r}"
            )
        if not str(self.policy_id).strip():
            raise DataContractError("policy_id must be non-empty")
        if not np.isfinite(self.z_max) or self.z_max <= 0:
            raise DataContractError(f"z_max={self.z_max!r} must be finite and positive")
        if not 0.0 <= float(self.z_fraction_threshold) < 1.0:
            raise DataContractError("z_fraction_threshold must lie in [0, 1)")
        allow = load_spin_prior_allow_list(dict(self.spin_prior_allow_list))
        object.__setattr__(self, "spin_prior_allow_list", allow or {})
        object.__setattr__(
            self, "z_allow_list", {str(k): str(v) for k, v in self.z_allow_list.items()}
        )
        if self.z_fraction_above_zmax is not None:
            fractions = {str(k): float(v) for k, v in self.z_fraction_above_zmax.items()}
            bad = {k: v for k, v in fractions.items() if not 0.0 <= v <= 1.0}
            if bad:
                raise DataContractError(f"z fractions outside [0, 1]: {bad}")
            object.__setattr__(self, "z_fraction_above_zmax", fractions)
        cosmo = {str(k): float(v) for k, v in self.population_cosmology.items()}
        if set(cosmo) != {"H0", "Om0"}:
            raise DataContractError("population_cosmology must give exactly H0 and Om0")
        object.__setattr__(self, "population_cosmology", cosmo)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "policy_id": self.policy_id,
            "status": self.status,
            "z_max": float(self.z_max),
            "spin_prior_allow_list": dict(sorted(self.spin_prior_allow_list.items())),
            "z_allow_list": dict(sorted(self.z_allow_list.items())),
            "z_fraction_threshold": float(self.z_fraction_threshold),
            "z_fraction_above_zmax": (
                None
                if self.z_fraction_above_zmax is None
                else dict(sorted(self.z_fraction_above_zmax.items()))
            ),
            "require_exact_priors": bool(self.require_exact_priors),
            "require_sky_marginal": bool(self.require_sky_marginal),
            "require_cumulative_mixture": bool(self.require_cumulative_mixture),
            "require_reference_pair": bool(self.require_reference_pair),
            "require_exact_spin_allow_list": bool(self.require_exact_spin_allow_list),
            "population_cosmology": dict(sorted(self.population_cosmology.items())),
            "sources": dict(self.sources),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> GwcatV2DataPolicy:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise DataContractError(f"unknown v2 data policy key(s) {unknown}")
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, path: str | Path) -> GwcatV2DataPolicy:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> str:
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n")
        return str(path)

    @property
    def policy_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


def z_of_luminosity_distance(
    d_l: np.ndarray, *, H0: float, Om0: float, z_grid_max: float = 5.0, n_grid: int = 200001
) -> np.ndarray:
    """Flat-LambdaCDM redshift of luminosity distances (Mpc), numpy only.

    Trapezoid comoving distance on a fine grid, inverted by interpolation; the
    relative error is below 1e-9 for z < 5, ample for a support diagnostic.
    """
    z = np.linspace(0.0, z_grid_max, n_grid)
    inv_e = 1.0 / np.sqrt(Om0 * (1.0 + z) ** 3 + (1.0 - Om0))
    dc = np.concatenate(([0.0], np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * np.diff(z))))
    dl_grid = (1.0 + z) * dc * (_C_KM_S / H0)
    d_l = np.asarray(d_l, dtype=float)
    if np.any(d_l > dl_grid[-1]):
        raise DataContractError("luminosity distance beyond the z grid")
    return np.interp(d_l, dl_grid, z)


def validate_redshift_support(
    pe: PosteriorCatalog,
    selection: SelectionCatalog,
    *,
    z_max: float,
    z_allow_list: Mapping[str, str],
    z_fraction_threshold: float = 0.01,
    z_fraction_above_zmax: Mapping[str, float] | None = None,
    population_cosmology: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """G17 at the adapter: one z_max on both halves, and the approved z allow-list.

    * Both exports record ``z_max`` equal to the policy value, and every exported
      PE sample and selection row lies at ``z <= z_max`` (their own z column).
    * Events with more than ``z_fraction_threshold`` of PE mass above ``z_max``
      must be on ``z_allow_list`` (BUILD_PLAN Section 11). The export records
      the number of samples it cut (``n_samples_cut_by_z_max``) but not the raw
      pool size, so the fraction is known only up to the upper bound
      ``cut / (cut + n_unique + n_spin_dropped)``. When the gate's measured
      per-event fractions are supplied they decide; otherwise an event whose
      upper bound exceeds the threshold and is not allow-listed is refused.
    * Every allow-listed event must be in the catalog and must have had
      samples cut (a stale entry is refused).
    * Diagnostic: exported PE samples whose z at the population cosmology
      exceeds ``z_max`` (they get zero population weight in the model).
    """
    provenance = dict(pe.metadata.get("prior_provenance") or {})
    pe_zmax = provenance.get("z_max")
    sel_zmax = selection.metadata.get("z_max")
    for side, value in (("PE", pe_zmax), ("selection", sel_zmax)):
        if value is None or not np.isfinite(float(value)):
            raise DataContractError(
                f"{side} export records no finite z_max; v2 requires z_max={z_max}"
            )
        if not np.isclose(float(value), float(z_max), rtol=0.0, atol=1e-12):
            raise DataContractError(
                f"{side} export z_max={value} != policy z_max={z_max}"
            )
    for side, catalog in (("PE", pe), ("selection", selection)):
        if "z" not in catalog.samples:
            raise DataContractError(f"{side} export has no redshift column")
        zmax_seen = float(np.max(catalog.samples["z"]))
        if zmax_seen > float(z_max):
            raise DataContractError(
                f"{side} export has samples at z={zmax_seen} > z_max={z_max}"
            )
    names = list(pe.event_names)
    for key in (
        "n_samples_cut_by_z_max",
        "n_unique_samples_per_event",
        "n_dropped_spin_above_ceiling_per_event",
    ):
        if key not in provenance:
            raise DataContractError(f"PE export records no {key}; cannot check G17")
    cut = np.asarray(provenance["n_samples_cut_by_z_max"], dtype=float)
    unique = np.asarray(provenance["n_unique_samples_per_event"], dtype=float)
    dropped = np.asarray(provenance["n_dropped_spin_above_ceiling_per_event"], dtype=float)
    upper = np.where(cut > 0, cut / np.maximum(cut + unique + dropped, 1.0), 0.0)

    allow = {str(k): str(v) for k, v in z_allow_list.items()}
    stale = sorted(set(allow) - set(names))
    if stale:
        raise DataContractError(f"z allow-list names events not in the catalog: {stale}")
    uncut = [n for n, c in zip(names, cut) if n in allow and c <= 0]
    if uncut:
        raise DataContractError(
            f"z allow-listed event(s) {uncut} had no samples above z_max cut"
        )

    measured: dict[str, float] | None = None
    if z_fraction_above_zmax is not None:
        measured = {str(k): float(v) for k, v in z_fraction_above_zmax.items()}
        missing = [n for n in names if n not in measured]
        extra = sorted(set(measured) - set(names))
        if missing or extra:
            raise DataContractError(
                f"measured z fractions must cover exactly the catalog; missing "
                f"{missing[:5]}, extra {extra[:5]}"
            )
        frac = np.asarray([measured[n] for n in names])
        basis = "measured"
    else:
        frac = upper
        basis = "export_upper_bound"
    over = [(n, float(f)) for n, f in zip(names, frac) if f > z_fraction_threshold]
    refused = [(n, f) for n, f in over if n not in allow]
    if refused:
        hint = (
            ""
            if measured is not None
            else " (fractions are the export's upper bound; supply the gate's "
            "measured per-event fractions to refine them)"
        )
        raise DataContractError(
            f"{len(refused)} event(s) have more than {z_fraction_threshold:.2%} of PE "
            f"mass above z_max={z_max} and are not on the z allow-list: "
            f"{refused[:10]}{hint}"
        )

    diag: dict[str, object] = {}
    if population_cosmology is not None:
        z_pop = z_of_luminosity_distance(
            pe.samples["luminosity_distance"],
            H0=float(population_cosmology["H0"]),
            Om0=float(population_cosmology["Om0"]),
        )
        above = z_pop > float(z_max)
        per_event = {
            pe.event_names[i]: int(np.sum(above[pe.event_slice(i)]))
            for i in range(pe.n_events)
            if np.any(above[pe.event_slice(i)])
        }
        diag = {
            "population_cosmology": dict(population_cosmology),
            "n_exported_pe_samples_above_zmax_at_population_cosmology": int(above.sum()),
            "per_event_exported_samples_above_zmax_at_population_cosmology": per_event,
        }
    return {
        "z_max": float(z_max),
        "pe_z_max_attr": float(pe_zmax),
        "selection_z_max_attr": float(sel_zmax),
        "max_exported_pe_z": float(np.max(pe.samples["z"])),
        "max_selection_z": float(np.max(selection.samples["z"])),
        "n_events_with_samples_cut_by_z_max": int(np.sum(cut > 0)),
        "fraction_basis": basis,
        "events_above_threshold": over,
        "z_allow_list": dict(sorted(allow.items())),
        "allow_listed_events_below_threshold": sorted(
            n for n in allow if dict(zip(names, frac))[n] <= z_fraction_threshold
        ),
        **diag,
    }


def evaluate_gwcat_v2_policy(
    pe: PosteriorCatalog,
    selection: SelectionCatalog,
    policy: GwcatV2DataPolicy,
) -> dict[str, object]:
    """Run every policy check on a loaded pair; raise on the first failure."""
    checks: dict[str, object] = {}
    selection_basis = str(selection.metadata.get("spin_basis", ""))
    if policy.require_reference_pair and "spin_reference" not in selection.metadata:
        raise DataContractError(
            f"v2 policy requires a chieff_reference selection; got {selection_basis!r}"
        )
    pairing = validate_reference_pairing(
        pe,
        selection,
        spin_prior_allow_list=dict(policy.spin_prior_allow_list),
        require_exact_allow_list=policy.require_exact_spin_allow_list,
    )
    if pairing is not None and "spin_prior_sources" not in pairing:
        raise DataContractError(
            "v2 policy requires GW-40c per-event spin prior source kinds on the PE"
        )
    checks["reference_pairing"] = pairing
    if policy.require_exact_priors:
        checks["exact_priors"] = validate_exact_priors(pe, selection)
    sky_marginal = bool(selection.metadata.get("sky_marginal")) and bool(
        pe.metadata.get("sky_marginal")
    )
    if policy.require_sky_marginal and not sky_marginal:
        raise DataContractError(
            "v2 policy requires a sky-marginal pair (selection sky_marginalized / "
            "sky_position_available=False)"
        )
    checks["sky"] = {
        "sky_marginal": bool(selection.metadata.get("sky_marginal")),
        "reason": selection.metadata.get("sky_mode_reason"),
        "pe_sky_columns_dropped": pe.metadata.get("sky_columns_dropped", []),
        "basis_identity": pe.basis.identity,
    }
    mixture = selection.metadata.get("cumulative_mixture")
    if policy.require_cumulative_mixture:
        if mixture is None:
            raise DataContractError("v2 policy requires a cumulative-mixture selection")
        if not dict(mixture).get("bookkeeping_derived"):
            raise DataContractError(
                "v2 policy requires derived per-component exposure bookkeeping "
                "(N_per_component / T_per_component_s)"
            )
    checks["cumulative_mixture"] = mixture
    checks["redshift_support"] = validate_redshift_support(
        pe,
        selection,
        z_max=policy.z_max,
        z_allow_list=policy.z_allow_list,
        z_fraction_threshold=policy.z_fraction_threshold,
        z_fraction_above_zmax=policy.z_fraction_above_zmax,
        population_cosmology=policy.population_cosmology,
    )
    return checks
