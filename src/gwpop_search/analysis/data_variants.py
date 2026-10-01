"""Importance reweighting of a posterior to a data variant (BUILD_PLAN OD-3b).

The v2 primary analysis is ALWAYS the 259-event catalog with the cumulative
O1-O4b injection set. Three pre-declared variants are reported **only by
importance reweighting** of the primary posterior (operator decision OD-3b,
2026-09-30):

* ``o3o4b_249`` -- the 249 O3-O4b events with the O1/O2 injection rows dropped
  from the same file (rows dropped only; ``N``, ``T`` and the per-row weights
  are never renormalised, BUILD_PLAN Sec. 3.4);
* ``snr9`` / ``snr11`` -- the O1/O2 semianalytic-SNR detection threshold at 9
  and 11 instead of 10 (accepted approximation A1), i.e. a different
  selection file with the same events.

For a posterior sample ``{Lambda_k, W_k}`` of the primary likelihood ``L`` and a
variant likelihood ``L'`` with the same hyperprior::

    r_k = L'(Lambda_k) / L(Lambda_k),      W'_k ∝ W_k r_k,
    ln Z' - ln Z = ln sum_k W_k r_k

where ``ln L = sum_i ln ell_i - N ln A`` is the shape likelihood of
:mod:`.terms` (and, when the sampled likelihood carries the v2 variance taper,
``+ ln taper(sigma^2_lnL)`` with the variance of each dataset; pass
``log_taper``). The variant posterior and the evidence shift are exact in the
limit of many draws; their reliability is measured by

* the Kish ESS of ``W'`` (and of ``W`` for reference), and
* the Pareto ``k-hat`` of the ratios ``r`` over equal-weight draws of ``W``
  (PSIS; Vehtari et al. 2024).

**Rerun rule (flag only).** A variant is flagged ``rerun_recommended`` when
``ESS' < 1000`` or the result is borderline: ``ESS' < 1500``, ``k-hat > 0.5``
(or undefined), or -- for a supplied edge -- the variant ``ln BF`` changes
sign or lies within ``borderline_margin`` of the ``|ln BF| = 3`` decision
threshold. The thresholds 1500 / 0.5 / 1 nat are DRAFT (operator approves at
freeze). **This module never launches a rerun:** a full rerun of a variant is
gated on the operator (OD-3b); the flag is the whole output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from ._common import (
    AnalysisInputError,
    WeightedPosterior,
    as_float,
    hbi_config_from_identity,
    json_ready,
    kish_ess,
    model_hash_of,
    require_identity_matches,
    weighted_quantile,
)
from .terms import CatalogWeightEvaluator, pad_catalog

VARIANT_FORMAT = "gwpop-search-data-variant-reweighting-1.0"
RERUN_POLICY = (
    "operator-gated (BUILD_PLAN Sec. 10 OD-3b): reweighting only; this tool never launches "
    "a rerun, it only flags one"
)
#: the pre-declared variants of the v2 primary analysis (OD-3b)
PREDECLARED_VARIANTS = ("o3o4b_249", "snr9", "snr11")
#: first GPS-free O3 date in event names (O3a started 2019-04-01)
O3_START_YYMMDD = 190401
_EVENT_DATE = re.compile(r"^GW(\d{6})")


@dataclass(frozen=True)
class ReweightCriteria:
    """Thresholds of the rerun flag (``min_ess`` is the plan's 1000; the rest DRAFT)."""

    min_ess: float = 1000.0
    borderline_ess: float = 1500.0
    khat_unreliable: float = 0.7
    khat_borderline: float = 0.5
    decision_threshold: float = 3.0
    borderline_margin: float = 1.0
    n_khat_draws: int = 4000

    def __post_init__(self) -> None:
        for name in ("min_ess", "borderline_ess", "khat_unreliable", "khat_borderline",
                     "decision_threshold", "borderline_margin"):
            as_float(name, getattr(self, name), nonnegative=True)
        if self.borderline_ess < self.min_ess:
            raise ValueError("borderline_ess must be >= min_ess")
        if self.khat_borderline > self.khat_unreliable:
            raise ValueError("khat_borderline must be <= khat_unreliable")
        if int(self.n_khat_draws) < 10:
            raise ValueError("n_khat_draws must be >= 10")

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class DataVariant:
    """A variant dataset: its PE catalog and selection, and how it was derived."""

    variant_id: str
    posterior: object
    selection: object
    description: str = ""
    derivation: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variant_id or not re.match(r"^[A-Za-z0-9_.-]+$", self.variant_id):
            raise ValueError("variant_id must contain only letters, numbers, _, . or -")


def event_run_is_pre_o3(name: str) -> bool:
    """True for O1/O2 events (date in the ``GWYYMMDD`` name before 2019-04-01)."""
    match = _EVENT_DATE.match(str(name))
    if match is None:
        raise ValueError(f"cannot read an observing date from event name {name!r}")
    return int(match.group(1)) < O3_START_YYMMDD


def o1_o2_event_names(names: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(n) for n in names if event_run_is_pre_o3(n))


def subset_selection_rows(selection, keep, *, reason: str):
    """Keep only the rows ``keep`` (boolean mask); campaigns, ``N`` and ``T`` unchanged.

    This is the only legal run subset of the v2 selection (BUILD_PLAN Sec. 3.4:
    "a run subset means dropping rows only"): the per-campaign ``n_draw`` and
    observing time (raw-draw) or the per-row estimator weights
    (estimator-ready) stay exactly as in the primary file.
    """
    from gwpop_search.data import SelectionCatalog

    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (selection.n_selected,):
        raise ValueError(f"row mask must have shape ({selection.n_selected},); got {keep.shape}")
    if not keep.any():
        raise ValueError("row subset removes every selection row")
    return SelectionCatalog(
        samples={k: np.asarray(v)[keep] for k, v in selection.samples.items()},
        log_draw_density=np.asarray(selection.log_draw_density)[keep],
        campaign_id=np.asarray(selection.campaign_id)[keep],
        campaigns=selection.campaigns,
        basis=selection.basis,
        mode=selection.mode,
        estimator_semantics=selection.estimator_semantics,
        metadata={
            **dict(selection.metadata),
            "row_subset": {
                "source_n_selected": int(selection.n_selected),
                "selected_n": int(keep.sum()),
                "reason": str(reason),
                "renormalised": False,
            },
        },
    )


def make_data_variant(
    variant_id: str,
    posterior,
    selection,
    *,
    drop_events: Sequence[str] = (),
    drop_o1o2_events: bool = False,
    selection_override=None,
    selection_row_mask=None,
    description: str = "",
) -> DataVariant:
    """Build a variant from the primary catalogs (events dropped, selection replaced/subset)."""
    from gwpop_search.data.subset import drop_posterior_events

    drop = list(dict.fromkeys(str(n) for n in drop_events))
    if drop_o1o2_events:
        drop.extend(n for n in o1_o2_event_names(posterior.event_names) if n not in drop)
    variant_posterior = (
        drop_posterior_events(posterior, drop, reason=f"data variant {variant_id}") if drop else posterior
    )
    if selection_override is not None and selection_row_mask is not None:
        raise ValueError("pass selection_override or selection_row_mask, not both")
    if selection_override is not None:
        variant_selection = selection_override
        sel_how = "replaced"
    elif selection_row_mask is not None:
        variant_selection = subset_selection_rows(selection, selection_row_mask, reason=f"data variant {variant_id}")
        sel_how = "row_subset"
    else:
        variant_selection = selection
        sel_how = "unchanged"
    return DataVariant(
        variant_id=variant_id,
        posterior=variant_posterior,
        selection=variant_selection,
        description=description,
        derivation={
            "dropped_events": drop,
            "n_events": int(variant_posterior.n_events),
            "selection": sel_how,
            "n_selection_rows": int(variant_selection.n_selected),
        },
    )


def _taper_declared(identity: Mapping[str, object] | None) -> bool:
    """Whether the sampled likelihood's identity declares a variance taper.

    The v2 taper is ``identity["hbi_config"]["variance_taper"]``
    (:class:`gwpop_search.hbi.types.HBIConfig`); any other key naming a taper
    outside the data block is also treated as a declaration (conservative).
    """

    def walk(node) -> bool:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if "taper" in str(key).lower() and value not in (None, False):
                    return True
                if walk(value):
                    return True
        elif isinstance(node, (list, tuple)):
            return any(walk(item) for item in node)
        return False

    if not identity:
        return False
    hbi = identity.get("hbi_config") if isinstance(identity, Mapping) else None
    if isinstance(hbi, Mapping) and hbi.get("variance_taper") not in (None, False):
        return True
    return walk({k: v for k, v in identity.items() if k != "data"})


def _log_likelihood_and_variance(sample, posterior, selection, model, hbi_config, *, batch_size, backend, require_support):
    # untapered ln L and sigma^2_lnL; the caller adds ln T(sigma^2) of each dataset
    treatment = "weights_only" if getattr(hbi_config, "variance_taper", None) is not None else None
    catalog = pad_catalog(posterior, selection, model, hbi_config=hbi_config, taper_treatment=treatment)
    evaluator = CatalogWeightEvaluator(catalog, model, sample.names, batch_size=batch_size, backend=backend)
    # with zero weights the evaluator does not refuse unsupported points: a variant
    # may legitimately give a primary-posterior point zero likelihood (r = 0)
    W = sample.weights if require_support else np.zeros_like(sample.weights)
    out = evaluator.moments(sample.points, W)
    return np.asarray(out["log_likelihood"], dtype=np.float64), np.asarray(out["variance"], dtype=np.float64)


def _summaries(sample: WeightedPosterior, weights: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for k, name in enumerate(sample.names):
        x = sample.points[:, k]
        mean = float(np.sum(weights * x))
        std = float(math.sqrt(max(float(np.sum(weights * (x - mean) ** 2)), 0.0)))
        q = weighted_quantile(x, weights, [0.05, 0.5, 0.95])
        out[name] = {"mean": mean, "std": std, "q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2])}
    return out


def reweight_to_variant(
    sample: WeightedPosterior,
    posterior,
    selection,
    population_model,
    variant: DataVariant,
    *,
    hbi_config=None,
    log_taper: Callable[[np.ndarray], np.ndarray] | None = None,
    criteria: ReweightCriteria | None = None,
    label: str | None = None,
    verify_identity: bool = True,
    batch_size: int = 16,
    backend: str = "jax",
    seed: int = 0,
) -> dict[str, object]:
    """Reweight one model's primary posterior to ``variant`` (see the module docstring).

    Returns a JSON-ready dict with the evidence shift, the variant posterior
    summaries, the ESS/k-hat diagnostics and the rerun flag. Never runs a
    sampler.
    """
    from gwpop_search.inference.dynesty_backend import equal_weight_resample

    from .psis_loo import psis_smooth

    criteria = ReweightCriteria() if criteria is None else criteria
    identity = sample.likelihood_identity
    if hbi_config is None:
        from gwpop_search.hbi import HBIConfig

        hbi_config = (
            hbi_config_from_identity(identity) if identity is not None else HBIConfig(selection_chunk_size=None)
        )
    if verify_identity:
        require_identity_matches(
            identity, posterior, selection, population_model, sample.names, hbi_config,
            what=f"data-variant reweighting of {label or 'model'}",
        )
    configured = getattr(hbi_config, "variance_taper", None)
    if log_taper is None and configured is not None:
        # the sampled likelihood's own taper, each dataset with its own sigma^2_lnL
        log_taper = configured.log_taper
    if _taper_declared(identity) and log_taper is None:
        raise AnalysisInputError(
            "the sampled likelihood declares a variance taper; pass log_taper so that the primary "
            "and variant likelihoods carry the same taper (with their own sigma^2_lnL)"
        )
    ll_p, var_p = _log_likelihood_and_variance(
        sample, posterior, selection, population_model, hbi_config,
        batch_size=batch_size, backend=backend, require_support=True,
    )
    ll_v, var_v = _log_likelihood_and_variance(
        sample, variant.posterior, variant.selection, population_model, hbi_config,
        batch_size=batch_size, backend=backend, require_support=False,
    )
    taper_p = taper_v = None
    if log_taper is not None:
        taper_p = np.asarray(log_taper(var_p), dtype=np.float64)
        taper_v = np.asarray(log_taper(var_v), dtype=np.float64)
        if taper_p.shape != ll_p.shape or taper_v.shape != ll_v.shape:
            raise ValueError("log_taper must return one value per posterior point")
        if np.any(np.isnan(taper_p)) or np.any(np.isnan(taper_v)) or np.any(taper_p > 1e-12) or np.any(taper_v > 1e-12):
            raise ValueError("log_taper must return finite-or--inf values <= 0")
        ll_p = ll_p + taper_p
        ll_v = ll_v + taper_v
    if not np.all(np.isfinite(ll_p)):
        raise AnalysisInputError("the primary likelihood is not finite at every positive-weight posterior point")
    log_r = np.where(np.isfinite(ll_v), ll_v - ll_p, -np.inf)
    log_W = np.log(sample.weights)
    log_num = log_W + log_r
    if not np.any(np.isfinite(log_num)):
        raise AnalysisInputError(f"variant {variant.variant_id}: every posterior point has zero variant likelihood")
    delta_ln_z = float(logsumexp(log_num))
    w_new = np.exp(log_num - delta_ln_z)
    ess_new = kish_ess(w_new)
    ess_root = kish_ess(sample.weights)
    # MC standard error of ln sum W r (self-normalised IS, W treated as fixed)
    r_rel = np.exp(log_r - delta_ln_z)
    se_ln_z = float(math.sqrt(float(np.sum(sample.weights**2 * (r_rel - 1.0) ** 2))))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5EED]))
    # k-hat on an equal-weight resample no larger than the primary Kish ESS:
    # a larger resample fills the tail with duplicates and biases the Pareto fit
    n_khat = int(max(10, min(int(criteria.n_khat_draws), math.floor(ess_root))))
    idx = equal_weight_resample(np.arange(sample.n_points), sample.weights, n_khat, rng)
    finite_r = log_r[idx][np.isfinite(log_r[idx])]
    if finite_r.size and float(np.ptp(finite_r)) <= 1e-12 and finite_r.size == idx.size:
        khat = 0.0  # constant ratios: the reweighting is exact
    else:
        khat = float(psis_smooth(log_r[idx]).khat)

    reasons = []
    if ess_new < criteria.min_ess:
        reasons.append("ess_below_min")
    elif ess_new < criteria.borderline_ess:
        reasons.append("ess_borderline")
    if not math.isfinite(khat):
        reasons.append("khat_undefined")
    elif khat > criteria.khat_unreliable:
        reasons.append("khat_unreliable")
    elif khat > criteria.khat_borderline:
        reasons.append("khat_borderline")
    root = _summaries(sample, sample.weights)
    new = _summaries(sample, w_new)
    shifts = {
        name: (None if root[name]["std"] == 0.0 else (new[name]["mean"] - root[name]["mean"]) / root[name]["std"])
        for name in sample.names
    }
    payload = {
        "format_version": VARIANT_FORMAT,
        "label": str(label) if label is not None else "model",
        "model_hash": model_hash_of(identity),
        "variant_id": variant.variant_id,
        "description": variant.description,
        "derivation": dict(variant.derivation),
        "n_events_primary": int(posterior.n_events),
        "n_events_variant": int(variant.posterior.n_events),
        "n_posterior_points": int(sample.n_points),
        "identity_verified": bool(verify_identity),
        "taper_applied": log_taper is not None,
        "taper": None if configured is None else configured.to_dict(),
        "delta_log_evidence": delta_ln_z,
        "delta_log_evidence_mc_se": se_ln_z,
        "ess_primary": ess_root,
        "ess_variant": ess_new,
        "ess_fraction": ess_new / ess_root,
        "pareto_khat": khat,
        "pareto_khat_resample_size": n_khat,
        "primary_ess_below_min_ess": bool(ess_root < criteria.min_ess),
        "n_zero_variant_likelihood": int(np.sum(~np.isfinite(log_r))),
        "posterior_primary": root,
        "posterior_variant": new,
        "posterior_mean_shift_in_primary_sigma": shifts,
        "variance_lnL_primary_median": float(weighted_quantile(var_p, sample.weights, 0.5)),
        "variance_lnL_variant_median": float(weighted_quantile(var_v, w_new, 0.5)),
        "criteria": criteria.to_dict(),
        "rerun_reasons": reasons,
        "rerun_recommended": bool(reasons),
        "status": "rerun_recommended_operator_gated" if reasons else "reweighting_reliable",
        "rerun_policy": RERUN_POLICY,
    }
    return json_ready(payload)


def variant_edge_log_bayes_factor(
    child: Mapping[str, object],
    parent: Mapping[str, object],
    log_bayes_factor: float,
    *,
    criteria: ReweightCriteria | None = None,
) -> dict[str, object]:
    """``ln BF'`` of an edge under a variant from the two models' reweightings.

    ``ln BF' = ln BF + (ln Z'_c - ln Z_c) - (ln Z'_p - ln Z_p)``. The edge is
    flagged when either model's reweighting is flagged, the sign changes, or
    ``| |ln BF'| - 3 | < borderline_margin``. Flag only (operator-gated rerun).
    """
    criteria = ReweightCriteria() if criteria is None else criteria
    if child.get("variant_id") != parent.get("variant_id"):
        raise AnalysisInputError("child and parent reweightings describe different variants")
    lnbf = float(log_bayes_factor)
    new = lnbf + float(child["delta_log_evidence"]) - float(parent["delta_log_evidence"])
    se = math.hypot(float(child["delta_log_evidence_mc_se"]), float(parent["delta_log_evidence_mc_se"]))
    reasons = []
    if child.get("rerun_recommended"):
        reasons.append("child_reweighting_flagged")
    if parent.get("rerun_recommended"):
        reasons.append("parent_reweighting_flagged")
    if np.sign(new) != np.sign(lnbf):
        reasons.append("sign_change")
    if abs(abs(new) - criteria.decision_threshold) < criteria.borderline_margin:
        reasons.append("near_decision_threshold")
    return json_ready({
        "variant_id": child.get("variant_id"),
        "child_hash": child.get("model_hash"),
        "parent_hash": parent.get("model_hash"),
        "log_bayes_factor_primary": lnbf,
        "log_bayes_factor_variant": new,
        "reweighting_mc_se": se,
        "rerun_reasons": reasons,
        "rerun_recommended": bool(reasons),
        "rerun_policy": RERUN_POLICY,
    })
