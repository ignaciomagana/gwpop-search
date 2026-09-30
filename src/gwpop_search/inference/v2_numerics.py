"""v2 (GWTC-5 BBH atom search) numerical configuration: taper, seeds, fidelity.

DRAFT: every value here is a pre-registration draft for operator approval at
freeze; nothing in this module has been frozen.

Numerics decided for v2 (plan ``scalable-stargazing-shamir``, "Numerics"):

* Variance guard *inside* the likelihood: the smooth taper of
  :mod:`gwpop_search.hbi.taper` at ``sigma^2_lnL = 1`` (GWTC-5,
  arXiv:2605.27226 Sec. III; Callister & Farr 2024, PRX 14, 021005). Claimed
  edges are re-run with the taper at 2 (sensitivity; D3).
* One dynesty run per model: nlive 500, bound ``multi``, sample ``rslice``,
  dlogz 0.1 (``F3`` rung, one repeat).
* A second seed only for decision-relevant edges (:class:`SecondSeedRule`):
  ``|ln BF|`` within ``2 sigma`` of ``+-3``, or any claimed edge.
* Pilot seed-scatter rule (:class:`PilotSeedScatterRule`): if the measured
  scatter of ``ln Z`` over the pilot's root seeds exceeds 1.5 x dynesty's own
  error estimate, fall back to 2 seeds for every model.
* Binding (D1): every nested-sampling and evidence check binds; the
  importance-sampling checks (ESS, maximum weights, ``Var[ln L]``) are
  reported but non-binding, because the taper (not a post-hoc gate) now
  handles an unreliable Monte-Carlo estimate; the posterior mass inside the
  taper region is always reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gwpop_search.hbi import HBIConfig
from gwpop_search.hbi.taper import CALLISTER_FARR_EXPONENT, VarianceTaper

from .dynesty_backend import DynestyConfig
from .evidence_campaign import EvidenceCampaignConfig
from .fidelity import (
    REQUIRED_BOUND,
    REQUIRED_SAMPLE,
    FidelityRunConfig,
    NumericalCriteria,
    fidelity_config_sha256,
    fidelity_run_config_to_dict,
)

V2_NUMERICS_FORMAT_VERSION = "gwpop-search-v2-numerics-draft-1.0"
V2_TAPER_THRESHOLD = 1.0
V2_TAPER_SENSITIVITY_THRESHOLD = 2.0
V2_NLIVE = 500
V2_DLOGZ = 0.1

# Importance-sampling check families: reported, non-binding in v2 (see module doc).
V2_NON_BINDING_FAMILIES = (
    "importance.min_event_ess",
    "importance.selection_ess",
    "importance.max_event_weight_fraction",
    "importance.selection_max_weight_fraction",
    "importance.shape_log_likelihood_variance",
)


def _positive(name: str, value) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive; got {value}")
    return value


@dataclass(frozen=True)
class SecondSeedRule:
    """Pre-declared rule: which edges get a second dynesty seed.

    An edge (child vs parent model) is decision-relevant when its
    ``ln BF`` lies within ``n_sigma * sigma_total`` of either decision
    threshold ``+-decision_threshold`` (the D2 claim / DISFAVOURED boundary,
    3), i.e. ``| |ln BF| - 3 | <= 2 sigma_total``, or when the edge is
    claimed. ``sigma_total`` is the edge's total ``ln BF`` uncertainty as used
    in D2 (Monte-Carlo and nested-sampling errors combined).
    """

    decision_threshold: float = 3.0
    n_sigma: float = 2.0
    always_for_claimed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision_threshold", _positive("decision_threshold", self.decision_threshold)
        )
        object.__setattr__(self, "n_sigma", _positive("n_sigma", self.n_sigma))
        if not isinstance(self.always_for_claimed, bool):
            raise TypeError("always_for_claimed must be a bool")

    def assess(self, ln_bf: float, sigma_total: float, *, claimed: bool = False) -> dict[str, object]:
        ln_bf = float(ln_bf)
        sigma = float(sigma_total)
        if not math.isfinite(ln_bf):
            raise ValueError("ln_bf must be finite")
        if not (math.isfinite(sigma) and sigma >= 0.0):
            raise ValueError("sigma_total must be finite and non-negative")
        distance = abs(abs(ln_bf) - self.decision_threshold)
        near = distance <= self.n_sigma * sigma
        claimed_rule = bool(claimed) and self.always_for_claimed
        return {
            "ln_bf": ln_bf,
            "sigma_total": sigma,
            "distance_to_threshold": distance,
            "window": self.n_sigma * sigma,
            "near_threshold": bool(near),
            "claimed": bool(claimed),
            "second_seed": bool(near or claimed_rule),
            "reason": (
                "claimed edge"
                if claimed_rule
                else (
                    f"|ln BF| within {self.n_sigma:g} sigma of +-{self.decision_threshold:g}"
                    if near
                    else "not decision-relevant"
                )
            ),
        }

    def requires_second_seed(
        self, ln_bf: float, sigma_total: float, *, claimed: bool = False
    ) -> bool:
        return bool(self.assess(ln_bf, sigma_total, claimed=claimed)["second_seed"])


@dataclass(frozen=True)
class PilotSeedScatterRule:
    """Pilot validation of the one-seed design.

    The pilot runs the root model with ``n_pilot_seeds`` seeds. The measured
    scatter is the sample standard deviation (ddof 1) of their ``ln Z``;
    dynesty's estimate is the root-mean-square of the runs' ``logzerr``. If
    ``scatter > max_scatter_ratio * estimate`` every model falls back to
    ``fallback_seeds`` seeds.
    """

    max_scatter_ratio: float = 1.5
    n_pilot_seeds: int = 3
    fallback_seeds: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_scatter_ratio", _positive("max_scatter_ratio", self.max_scatter_ratio)
        )
        for name, minimum in (("n_pilot_seeds", 2), ("fallback_seeds", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            object.__setattr__(self, name, int(value))

    def assess(self, log_evidences: Sequence[float], log_evidence_errors: Sequence[float]) -> dict:
        lnz = np.asarray(log_evidences, dtype=float)
        err = np.asarray(log_evidence_errors, dtype=float)
        if lnz.ndim != 1 or lnz.shape != err.shape:
            raise ValueError("log_evidences and log_evidence_errors must be 1-D of equal length")
        if lnz.size < self.n_pilot_seeds:
            raise ValueError(
                f"the pilot rule needs {self.n_pilot_seeds} seeds; got {lnz.size}"
            )
        if not (np.all(np.isfinite(lnz)) and np.all(np.isfinite(err)) and np.all(err > 0)):
            raise ValueError("log evidences must be finite and errors finite and positive")
        scatter = float(np.std(lnz, ddof=1))
        estimate = float(np.sqrt(np.mean(err**2)))
        ratio = scatter / estimate
        fallback = ratio > self.max_scatter_ratio
        return {
            "n_seeds": int(lnz.size),
            "measured_scatter": scatter,
            "dynesty_estimate_rms_logzerr": estimate,
            "ratio": ratio,
            "max_scatter_ratio": float(self.max_scatter_ratio),
            "fallback_to_two_seeds_everywhere": bool(fallback),
            "seeds_per_model": int(self.fallback_seeds if fallback else 1),
        }


@dataclass(frozen=True)
class SeedPolicy:
    """Seeds per model: one, plus the second-seed and pilot fallback rules."""

    seeds_per_model: int = 1
    second_seed: SecondSeedRule = field(default_factory=SecondSeedRule)
    pilot: PilotSeedScatterRule = field(default_factory=PilotSeedScatterRule)

    def seeds_for_edge(
        self,
        ln_bf: float,
        sigma_total: float,
        *,
        claimed: bool = False,
        pilot_fallback: bool = False,
    ) -> int:
        """Seeds each end of an edge needs (the pilot fallback overrides everything)."""
        base = self.pilot.fallback_seeds if pilot_fallback else self.seeds_per_model
        if self.second_seed.requires_second_seed(ln_bf, sigma_total, claimed=claimed):
            return max(base, 2)
        return base

    def to_dict(self) -> dict[str, object]:
        return {
            "seeds_per_model": int(self.seeds_per_model),
            "second_seed": asdict(self.second_seed),
            "pilot": asdict(self.pilot),
        }


def v2_variance_taper(threshold: float = V2_TAPER_THRESHOLD) -> VarianceTaper:
    """The v2 smooth taper (Callister & Farr p = 30) at ``threshold`` (1; 2 for D3)."""
    return VarianceTaper(kind="smooth", threshold=threshold, exponent=CALLISTER_FARR_EXPONENT)


def v2_criteria(*, repeats: int) -> NumericalCriteria:
    """v2 criteria: nested-sampling/evidence binding; importance reported, non-binding.

    ``repeats >= 2`` (the second-seed rung) adds the cross-run gates.
    DRAFT thresholds: Kish ESS 250 per run (half of v1's F3 value, as nlive
    halves); evidence error 0.5 (the D2 sigma budget); importance thresholds
    kept at v1 F3 values for the advisory record.
    """
    multi = int(repeats) >= 2
    return NumericalCriteria(
        max_cross_run_r_hat=1.05 if multi else None,
        min_kish_ess_per_run=250.0,
        require_dlogz_termination=True,
        require_selection_support=True,
        max_evidence_error=0.50,
        max_evidence_repeat_std=None,
        max_repeat_consistency_z=3.0 if multi else None,
        min_event_ess=10.0,
        min_selection_ess=100.0,
        max_event_weight_fraction=0.35,
        max_selection_weight_fraction=0.15,
        max_shape_log_likelihood_variance=V2_TAPER_THRESHOLD,
        binding={name: False for name in V2_NON_BINDING_FAMILIES},
        max_posterior_taper_mass=None,
    )


def v2_evidence(*, repeats: int) -> EvidenceCampaignConfig:
    return EvidenceCampaignConfig(
        repeats=int(repeats),
        dynesty=DynestyConfig(
            nlive=V2_NLIVE, bound=REQUIRED_BOUND, sample=REQUIRED_SAMPLE, dlogz=V2_DLOGZ
        ),
        slices_multiplier=2,
    )


def v2_fidelity_run_config(threshold: float = V2_TAPER_THRESHOLD) -> FidelityRunConfig:
    """DRAFT v2 fidelity configuration.

    ``F3``: one run per model (nlive 500, ``multi``/``rslice``, dlogz 0.1).
    ``F4``: the second-seed rung for decision-relevant edges (2 runs, same
    settings, cross-run gates). The likelihood carries the smooth variance
    taper at ``threshold``.
    """
    return FidelityRunConfig(
        f3_evidence=v2_evidence(repeats=1),
        f4_evidence=v2_evidence(repeats=2),
        f3_criteria=v2_criteria(repeats=1),
        f4_criteria=v2_criteria(repeats=2),
        hbi=HBIConfig(selection_chunk_size=None, variance_taper=v2_variance_taper(threshold)),
    )


def v2_campaign_numerics(
    *,
    fidelity_files: Mapping[str, str] | None = None,
    policy: SeedPolicy | None = None,
) -> dict[str, object]:
    """DRAFT campaign-level numerics (seed rules, taper, sensitivity rerun)."""
    policy = SeedPolicy() if policy is None else policy
    primary = v2_fidelity_run_config(V2_TAPER_THRESHOLD)
    sensitivity = v2_fidelity_run_config(V2_TAPER_SENSITIVITY_THRESHOLD)
    return {
        "format_version": V2_NUMERICS_FORMAT_VERSION,
        "status": "DRAFT - pre-registration draft; operator approves at freeze",
        "likelihood": {
            "rate_treatment": "shape (rate-marginalised, p(R) ~ 1/R)",
            "variance": "sigma^2 = sum_i Var[ln I_i] + N^2 Var[xi]/xi^2",
            "taper": primary.hbi.variance_taper.to_dict(),
            "taper_form": "ln T = -ln(1 + (sigma^2/threshold)^p), p = 30",
            "taper_sources": [
                "GWTC-5.0 populations, arXiv:2605.27226, Sec. III: maximum variance 1, "
                "'sharp or smoothly-tapered cutoff [Callister & Farr 2024, e.g.]'",
                "Callister & Farr 2024, PRX 14, 021005; tcallister/autoregressive-bbh-inference "
                "@53573e9 code/autoregressive_mass_models.py: "
                "log(1/(1+(N_eff/(4 N_obs))**-30))",
                "gwpopulation @b3a34f9 hyperpe.py: sharp cut ln L - inf*(max < variance)",
            ],
            "sensitivity_taper": sensitivity.hbi.variance_taper.to_dict(),
            "sensitivity_applies_to": "claimed edges (D3: the taper-at-2 rerun keeps the sign)",
        },
        "fidelity": {
            "primary": {
                "file": (fidelity_files or {}).get("primary"),
                "sha256": fidelity_config_sha256(primary),
            },
            "taper_sensitivity": {
                "file": (fidelity_files or {}).get("taper_sensitivity"),
                "sha256": fidelity_config_sha256(sensitivity),
            },
            "rungs": {
                "F0": "prior-support scan + JAX/NumPy parity (every node)",
                "F3": "one dynesty run per model, nlive 500, dlogz 0.1, multi/rslice",
                "F4": (
                    "second-seed rung (2 runs) for decision-relevant edges only; repeat 0 "
                    "has the same seed and configuration as the F3 run, so an orchestrator "
                    "can reuse it (reuse NOT implemented here)"
                ),
            },
        },
        "seed_policy": policy.to_dict(),
        "second_seed_rule": (
            "second seed iff | |ln BF| - 3 | <= 2 sigma_total, or the edge is claimed"
        ),
        "pilot_seed_scatter_rule": (
            "pilot root with 3 seeds: if std(ln Z, ddof=1) > 1.5 x rms(logzerr) then 2 seeds "
            "for every model"
        ),
        "binding": {
            "binding": "every nested-sampling and evidence check (D1)",
            "non_binding_reported": list(V2_NON_BINDING_FAMILIES),
            "reported": "posterior mass inside the taper region (per run and pooled)",
        },
        "nulls": "none (trials factor = number of atoms tried, disclosed with every claim)",
    }


def write_v2_draft_configs(out_dir: str | Path) -> dict[str, str]:
    """Write the DRAFT v2 fidelity (primary + taper-2 sensitivity) and campaign JSONs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "primary": "fidelity_v2_DRAFT.json",
        "taper_sensitivity": "fidelity_v2_taper2_DRAFT.json",
    }
    for key, threshold in (
        ("primary", V2_TAPER_THRESHOLD),
        ("taper_sensitivity", V2_TAPER_SENSITIVITY_THRESHOLD),
    ):
        payload = fidelity_run_config_to_dict(v2_fidelity_run_config(threshold))
        (out / files[key]).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    campaign = v2_campaign_numerics(fidelity_files=files)
    (out / "campaign_v2_numerics_DRAFT.json").write_text(
        json.dumps(campaign, sort_keys=True, indent=2) + "\n"
    )
    return {**files, "campaign": "campaign_v2_numerics_DRAFT.json"}
