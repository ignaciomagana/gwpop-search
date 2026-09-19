"""Model-comparison report: Bayes factors with a full error budget, posterior
model probabilities, structural masses and claim criteria.

Formulas (MODEL_COMPARISON_MATH.md Sec. 9.1):

* ``ln BF_ab = ln Zbar_a - ln Zbar_b`` (no additive term, nested or not);
  ``ln O_ab = ln BF_ab + ln p(M_a) - ln p(M_b)``.
* Error budget of an edge (formula (6))::

      sigma^2(ln BF_ab) = sigma_NS,a^2 + sigma_NS,b^2 + sigma^2_MC,ab

  with the nested-sampling error of each model's mean evidence
  (``ns_error_mode="formula6"``: ``max(repeat std, mean logzerr,
  kappa_hat sqrt(H/nlive)) / sqrt(R)``; ``"conservative"``: the per-run
  ``max(repeat std, max logzerr)`` of the evidence summaries, not divided by
  ``sqrt(R)``) and the *joint* Monte-Carlo-likelihood error ``sigma_MC,ab``
  of :mod:`gwpop_search.analysis.edge_mc_error` (common PE samples and
  injections; never the per-model MC errors in quadrature). The predicted MC
  bias is reported and gated, never subtracted.
* Posterior model probabilities over the **deduplicated** model set
  (statistically identical nodes, i.e. same structure and same priors on the
  hyperparameters actually used, are merged first: one evidence, one prior
  weight; finding F1). Uncertainty by Monte-Carlo propagation (formula (7))::

      ln Z_k^(s) = ln Zbar_k + sigma_NS,k xi_k^(s) + eta_k^(s),
      xi ~ N(0, 1) iid,  eta ~ N(0, Sigma_MC),  Sigma_MC[a, b] = Cov_MC[ln Zhat_a, ln Zhat_b].

* Structural masses ``P(S|D)`` with prior masses ``P(S)`` and
  ``BF_S = [P(S|D)/(1-P(S|D))] / [P(S)/(1-P(S))]`` (formula (5)), per atom and
  per axis.
* Claim criteria (Sec. 9.5), per edge and its atom ``S``; each criterion is
  ``pass``/``fail``/``missing``/``not_applicable``/``incomplete`` and a claim
  requires every applicable criterion to pass:

  1. numerics: both models pass their production gates, G-MC1..3 per model and
     G-MC4/5 for the edge (G-MC4 gates the same ``sigma_MC`` the budget uses,
     :func:`_budget_sigma_mc`), and every model was scored at one fidelity rung
     (D4's calibrated statistic must not mix F3 and F4 evidences);
  2. strength: ``ln BF - 2 sigma >= 3`` and model-averaged ``ln BF_S >= 3``;
  3. prior robustness: every prior-width variant keeps ``ln BF >= 1`` with an
     unchanged sign, and ``P(S|D) >= 0.75`` for every model-prior penalty;
  4. SDDR agreement where eligible. An SDDR-eligible edge whose cross-check
     could not be computed (``not_estimable``: the VW support condition fails,
     too few draws near the null, no estimable density and no NS bound) is
     ``missing``, not ``not_applicable``: it has no independent confirmation.
     Only an ``evidence_only`` edge is genuinely not applicable;
  5. search-level null calibration ``p = (b+1)/(m+1) <= alpha`` (pre-registered
     ``alpha = 0.01``) for every null truth; with ``1/(m+1) > alpha`` the
     calibration is resolution-limited and cannot pass. The calibration must
     also describe *this* report: its statistic has to be one the report
     computes and its ``observed`` has to reproduce the report's own value
     (``null_observed_sigmas``/``null_observed_atol``), and it cannot pass while
     some effective model has no evidence, because the statistic is then a
     maximum over a subset of the graph the nulls were calibrated on;
  6. event-drop, leave-one-out and nearby-baseline results do not reverse the sign.

Every criterion distinguishes ``fail`` (evaluated, the data did not meet it)
from ``missing``/``incomplete`` (not evaluated). Both block a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from gwpop_search.grammar import ModelGraph

from ._common import AnalysisInputError, as_float, as_int, json_ready
from .structure import (
    atoms_of,
    edge_atom,
    find_alias_groups,
    normalized_log_prior,
    structural_bayes_factor,
    structural_masses,
    complexity_prior,
)

REPORT_FORMAT = "gwpop-search-model-comparison-report-1.0"
DEFAULT_LAMBDAS = (0.0, math.log(2.0), math.log(4.0))
STATUSES = ("pass", "fail", "missing", "not_applicable", "incomplete")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceSummary:
    """Per-model evidence (mean over ``R`` independent repeats).

    ``gates_passed`` is the model's production numerical-gate status (``None``
    when unknown). ``information``/``nlive`` enable the
    ``kappa_hat sqrt(H/nlive)`` term of formula (6). ``rung`` names the dynesty
    trajectory configuration the repeats were run at (``None`` when unknown);
    :func:`build_model_comparison` refuses to call a report homogeneous when the
    models carry more than one, because D4's calibrated statistic must come from
    a single procedure.
    """

    model_hash: str
    log_evidence: float
    estimates: tuple[float, ...] = ()
    reported_errors: tuple[float, ...] = ()
    conservative_error: float | None = None
    information: float | None = None
    nlive: int | None = None
    gates_passed: bool | None = None
    rung: str | None = None

    def __post_init__(self) -> None:
        as_float("log_evidence", self.log_evidence)
        estimates = tuple(float(x) for x in self.estimates)
        errors = tuple(float(x) for x in self.reported_errors)
        if errors and len(errors) != len(estimates):
            raise ValueError("reported_errors must match estimates")
        if not all(math.isfinite(x) for x in (*estimates, *errors)):
            raise ValueError("evidence estimates and errors must be finite")
        if estimates and abs(float(np.mean(estimates)) - self.log_evidence) > 1e-9 * max(1.0, abs(self.log_evidence)):
            raise ValueError(f"{self.model_hash}: log_evidence is not the mean of the estimates")
        object.__setattr__(self, "estimates", estimates)
        object.__setattr__(self, "reported_errors", errors)
        if self.conservative_error is not None:
            as_float("conservative_error", self.conservative_error, nonnegative=True)
        if self.nlive is not None:
            as_int("nlive", self.nlive, minimum=1)

    @property
    def n_repeats(self) -> int:
        return max(1, len(self.estimates))

    @property
    def repeat_std(self) -> float | None:
        return float(np.std(self.estimates, ddof=1)) if len(self.estimates) > 1 else None

    def ns_sigma_per_run(self, kappa_hat: float | None = None) -> float:
        """``max(repeat std, mean logzerr, kappa_hat sqrt(H/nlive))``."""
        candidates = []
        if self.repeat_std is not None:
            candidates.append(self.repeat_std)
        if self.reported_errors:
            candidates.append(float(np.mean(self.reported_errors)))
        if kappa_hat is not None and self.information is not None and self.nlive:
            candidates.append(float(kappa_hat) * math.sqrt(max(self.information, 0.0) / self.nlive))
        if not candidates:
            if self.conservative_error is None:
                raise AnalysisInputError(f"{self.model_hash}: no evidence error information")
            return float(self.conservative_error)
        return float(max(candidates))

    def ns_error(self, mode: str = "formula6", kappa_hat: float | None = None) -> float:
        if mode == "formula6":
            return self.ns_sigma_per_run(kappa_hat) / math.sqrt(self.n_repeats)
        if mode == "conservative":
            if self.conservative_error is not None:
                return float(self.conservative_error)
            repeat = self.repeat_std or 0.0
            return float(max(repeat, max(self.reported_errors, default=0.0)))
        raise ValueError("ns_error_mode must be 'formula6' or 'conservative'")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_hash": self.model_hash,
            "log_evidence": self.log_evidence,
            "estimates": list(self.estimates),
            "reported_errors": list(self.reported_errors),
            "conservative_error": self.conservative_error,
            "information": self.information,
            "nlive": self.nlive,
            "gates_passed": self.gates_passed,
            "rung": self.rung,
            "n_repeats": self.n_repeats,
            "repeat_std": self.repeat_std,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], model_hash: str | None = None) -> "EvidenceSummary":
        """Read an evidence summary (``log_evidence`` or ``log_evidence_mean``)."""
        payload = dict(payload)
        value = payload.get("log_evidence", payload.get("log_evidence_mean"))
        if value is None:
            raise AnalysisInputError("evidence summary lacks log_evidence/log_evidence_mean")
        return cls(
            model_hash=str(model_hash or payload["model_hash"]),
            log_evidence=float(value),
            estimates=tuple(payload.get("estimates") or ()),
            reported_errors=tuple(payload.get("reported_errors", payload.get("errors")) or ()),
            conservative_error=(
                None if payload.get("conservative_error") is None else float(payload["conservative_error"])
            ),
            information=None if payload.get("information") is None else float(payload["information"]),
            nlive=None if payload.get("nlive") is None else int(payload["nlive"]),
            gates_passed=None if payload.get("gates_passed") is None else bool(payload["gates_passed"]),
            rung=None if payload.get("rung") is None else str(payload["rung"]),
        )

    @classmethod
    def from_dynesty_results(
        cls, model_hash: str, results: Sequence, *, gates_passed: bool | None = None
    ) -> "EvidenceSummary":
        from ._common import trajectory_rung

        results = tuple(results)
        if not results:
            raise ValueError("no dynesty results")
        estimates = tuple(float(r.log_evidence) for r in results)
        errors = tuple(float(r.log_evidence_error) for r in results)
        nlives = {int(r.config.nlive) for r in results}
        rungs = {trajectory_rung(r.config) for r in results}
        repeat = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
        return cls(
            model_hash=str(model_hash),
            log_evidence=float(np.mean(estimates)),
            estimates=estimates,
            reported_errors=errors,
            conservative_error=float(max(repeat, max(errors))),
            information=float(np.mean([r.information for r in results])),
            nlive=nlives.pop() if len(nlives) == 1 else None,
            gates_passed=gates_passed,
            rung=rungs.pop() if len(rungs) == 1 else None,
        )


def merge_alias_evidence(items: Sequence[EvidenceSummary], canonical: str) -> tuple[EvidenceSummary, dict]:
    """Merge the evidences of statistically identical models (repeats pooled)."""
    items = tuple(items)
    if len(items) == 1:
        return replace(items[0], model_hash=canonical), {"merged": [items[0].model_hash], "consistent": True}
    estimates = tuple(x for item in items for x in (item.estimates or (item.log_evidence,)))
    errors = tuple(e for item in items for e in item.reported_errors)
    if len(errors) != len(estimates):
        errors = ()
    means = [item.log_evidence for item in items]
    sigmas = [item.ns_error("formula6") if (item.estimates or item.conservative_error is not None) else 0.0 for item in items]
    consistent = True
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if abs(means[i] - means[j]) > 3.0 * math.hypot(sigmas[i], sigmas[j]):
                consistent = False
    conservative = [item.conservative_error for item in items if item.conservative_error is not None]
    gates = [item.gates_passed for item in items]
    # an unrecorded rung is unknown, not a second rung
    rungs = {item.rung for item in items if item.rung is not None}
    merged = EvidenceSummary(
        model_hash=canonical,
        log_evidence=float(np.mean(estimates)),
        estimates=estimates,
        reported_errors=errors,
        conservative_error=max(conservative) if conservative else None,
        information=items[0].information,
        nlive=items[0].nlive if len({item.nlive for item in items}) == 1 else None,
        gates_passed=None if any(g is None for g in gates) else all(gates),
        rung=rungs.pop() if len(rungs) == 1 else (None if not rungs else "mixed"),
    )
    return merged, {
        "merged": [item.model_hash for item in items],
        "log_evidences": means,
        "consistent": consistent,
    }


@dataclass(frozen=True)
class NullCalibration:
    """Search-level null replays of one null truth: ``b`` exceedances in ``m``."""

    label: str
    statistic: str
    observed: float
    exceedances: int
    replays: int

    def __post_init__(self) -> None:
        as_int("replays", self.replays, minimum=1)
        b = as_int("exceedances", self.exceedances, minimum=0)
        if b > self.replays:
            raise ValueError("exceedances cannot exceed replays")

    @property
    def p_value(self) -> float:
        return (self.exceedances + 1.0) / (self.replays + 1.0)

    @property
    def resolution(self) -> float:
        return 1.0 / (self.replays + 1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "statistic": self.statistic,
            "observed": self.observed,
            "exceedances": self.exceedances,
            "replays": self.replays,
            "p_value": self.p_value,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NullCalibration":
        return cls(
            label=str(payload.get("label", "null")),
            statistic=str(payload.get("statistic", "max_edge_log_bayes_factor")),
            observed=float(payload["observed"]),
            exceedances=int(payload["exceedances"]),
            replays=int(payload["replays"]),
        )


@dataclass(frozen=True)
class ClaimCriteria:
    strength_threshold: float = 3.0
    strength_sigmas: float = 2.0
    structural_threshold: float = 3.0
    robustness_min_log_bf: float = 1.0
    robustness_min_posterior_mass: float = 0.75
    alpha: float = 0.01
    gmc_max_expected_variance: float = 1.0
    gmc_min_neff_fraction: float = 0.99
    gmc_max_c_pp: float = 1.0
    gmc_max_edge_sigma: float = 0.3
    gmc_max_edge_bias: float = 0.3
    #: a supplied null calibration's ``observed`` statistic must reproduce the
    #: report's own within ``null_observed_sigmas`` error bars of the edge that
    #: attains it, plus ``null_observed_atol`` nats (D4 runs the identical F3
    #: procedure, so the two are the same number up to that noise)
    null_observed_sigmas: float = 3.0
    null_observed_atol: float = 0.05

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            as_float(name, getattr(self, name), nonnegative=True)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ComparisonConfig:
    ns_error_mode: str = "formula6"
    kappa_hat: float | None = None
    n_propagation: int = 4000
    seed: int = 0
    model_prior_lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    criteria: ClaimCriteria = field(default_factory=ClaimCriteria)

    def __post_init__(self) -> None:
        if self.ns_error_mode not in {"formula6", "conservative"}:
            raise ValueError("ns_error_mode must be 'formula6' or 'conservative'")
        if self.kappa_hat is not None:
            as_float("kappa_hat", self.kappa_hat, positive=True)
        as_int("n_propagation", self.n_propagation, minimum=2)
        as_int("seed", self.seed, minimum=0)
        object.__setattr__(self, "model_prior_lambdas", tuple(float(x) for x in self.model_prior_lambdas))

    def to_dict(self) -> dict[str, object]:
        return {
            "ns_error_mode": self.ns_error_mode,
            "kappa_hat": self.kappa_hat,
            "n_propagation": self.n_propagation,
            "seed": self.seed,
            "model_prior_lambdas": list(self.model_prior_lambdas),
            "criteria": self.criteria.to_dict(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget_sigma_mc(mc: Mapping[str, object]) -> float:
    """``sigma_MC`` for the error budget, never below the plug-in estimate.

    ``EdgeMCError.variance`` is the *unbiased* U-statistic of the edge
    Monte-Carlo variance. Under common random numbers the true edge variance is
    tiny, so that estimator can land slightly below zero on finite samples, and
    ``EdgeMCError.sigma`` then reports 0 (the raw value stays in ``variance``
    with ``variance_negative`` set). Reporting 0 in the budget would drop a real
    contribution, so the budget falls back to the plug-in estimate, which is
    non-negative and biased upward by the ``s = s'`` diagonal. Measured on the
    two-regime toy (validation/analysis_estimators/mc_error_*.json): where the
    U-statistic went negative the plug-in tracked the realized variance to 2%
    (sigma ~ 0.036 nats), and where the U-statistic was positive it was the
    tighter of the two, so this only ever widens the interval.
    """
    variance = mc.get("variance")
    plug_in = mc.get("variance_plug_in")
    if variance is None:  # a supplied sigma (source='supplied') carries no variance
        return float(mc["sigma"])  # type: ignore[arg-type]
    best = float(variance)
    if best < 0.0 and plug_in is not None:
        best = max(best, float(plug_in))
    return math.sqrt(max(best, 0.0))


#: search statistic -> the edge quantity it maximizes over
_STATISTIC_EDGE_KEY = {
    "max_edge_log_bayes_factor": "log_bayes_factor",
    "max_edge_log_posterior_odds": "log_posterior_odds",
}


def _canonicalize_edge_map(mapping, canonical: Mapping[str, str]):
    """Re-key a ``(parent_hash, child_hash)`` map onto canonical model hashes.

    Both spellings are kept so a caller that already supplied canonical keys is
    unaffected; a graph-hash key that maps onto an alias head becomes reachable.
    """
    if mapping is None:
        return None
    out = dict(mapping)
    for (p, c), value in mapping.items():
        key = (canonical.get(p, p), canonical.get(c, c))
        out.setdefault(key, value)
    return out


def _null_observed_tolerance(statistic: str, edges: Sequence[Mapping[str, object]], criteria: ClaimCriteria) -> float:
    """How far a null campaign's ``observed`` may sit from the report's own value.

    D4 computes the calibrated statistic from the identical F3 procedure, so the
    two numbers differ only by the evidence noise of the edge that attains the
    maximum: ``null_observed_sigmas * sigma_total`` of that edge plus an absolute
    floor. Statistics with no edge quantity (e.g. ``T3_any_structure``) get the
    floor alone.
    """
    key = _STATISTIC_EDGE_KEY.get(statistic)
    sigma = 0.0
    if key is not None and edges:
        best = max(edges, key=lambda e: float(e[key]))
        sigma = float(best.get("sigma_total") or 0.0)
    return criteria.null_observed_sigmas * sigma + criteria.null_observed_atol


def kass_raftery(log_bf: float) -> str:
    """Kass & Raftery (1995) scale in nats of ``ln B`` (boundaries 1, 3, 5)."""
    magnitude = abs(float(log_bf))
    if magnitude < 1.0:
        label = "not worth more than a bare mention"
    elif magnitude < 3.0:
        label = "positive"
    elif magnitude < 5.0:
        label = "strong"
    else:
        label = "very strong"
    if magnitude < 1.0:
        return label
    return f"{label} ({'for child' if log_bf > 0 else 'for parent'})"


def _psd_factor(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """``F`` with ``F F^T = PSD part of matrix``; negative eigenvalues are clipped and reported."""
    sym = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(sym)
    clipped = np.clip(values, 0.0, None)
    info = {
        "min_eigenvalue": float(values.min()),
        "clipped_negative_eigenvalue_sum": float(-np.sum(values[values < 0.0])),
    }
    return vectors * np.sqrt(clipped)[None, :], info


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.16, 0.5, 0.84])
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "q16": float(q[0]), "q50": float(q[1]), "q84": float(q[2])}


def _criterion(status: str, **detail) -> dict[str, object]:
    if status not in STATUSES:
        raise ValueError(status)
    return {"status": status, **detail}


def _combine(criteria: Mapping[str, Mapping[str, object]]) -> str:
    statuses = [c["status"] for c in criteria.values()]
    if "fail" in statuses:
        return "not_claimed"
    if any(s in {"missing", "incomplete"} for s in statuses):
        return "incomplete"
    return "claimed"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_model_comparison(
    graph: ModelGraph,
    evidences: Mapping[str, EvidenceSummary],
    model_prior,
    *,
    mc_weights: Mapping[str, object] | None = None,
    edge_mc: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
    sddr: Mapping[str, Mapping[str, object]] | None = None,
    prior_sensitivity: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
    null_calibrations: Sequence[NullCalibration] | None = None,
    stress: Mapping[tuple[str, str], Sequence[float]] | None = None,
    nearby: Mapping[str, Sequence[float]] | None = None,
    loo: Mapping[str, object] | None = None,
    config: ComparisonConfig | None = None,
    alias_groups=None,
) -> dict[str, object]:
    """Assemble the report (see the module docstring for every quantity).

    ``evidences`` maps model hashes to :class:`EvidenceSummary`;
    ``mc_weights`` maps model hashes to
    :class:`~gwpop_search.analysis.edge_mc_error.ModelMCWeights`
    (edge ``sigma_MC``/bias and ``Sigma_MC`` are derived from them);
    ``edge_mc`` optionally supplies precomputed ``{"sigma": ..., "bias": ...}``
    keyed by ``(parent_hash, child_hash)``; ``sddr`` maps ``"parent->child"``
    (or, for graphs where they are unique, mutation ids) to
    :class:`~gwpop_search.analysis.sddr.SDDRCheck` dicts; ``prior_sensitivity`` maps ``(parent, child)`` to
    :func:`~gwpop_search.analysis.prior_sensitivity.edge_prior_sensitivity`
    outputs; ``stress`` maps ``(parent, child)`` to stressed ``ln BF`` values
    (event-drop suites); ``nearby`` maps mutation ids to nearby-baseline
    ``ln BF`` values; ``loo`` is a PSIS-LOO report.
    """
    from .edge_mc_error import edge_mc_error, mc_covariance_matrix

    config = ComparisonConfig() if config is None else config
    crit = config.criteria
    by_hash = graph.by_hash
    root = by_hash[graph.root_hash]
    unknown = sorted(set(evidences) - set(by_hash))
    if unknown:
        raise AnalysisInputError(f"evidence supplied for unknown model hash(es) {unknown}")
    for key, item in evidences.items():
        if item.model_hash != key:
            raise AnalysisInputError(f"evidence keyed {key} describes {item.model_hash}")

    # -- deduplication ------------------------------------------------------
    aliases = alias_groups or find_alias_groups(graph.nodes)
    canonical = aliases.canonical
    canon_models = [h for h in (m.model_hash for m in graph.nodes) if canonical[h] == h]
    merged: dict[str, EvidenceSummary] = {}
    merge_log = {}
    for head in canon_models:
        members = [h for h in aliases.groups[head] if h in evidences]
        if members:
            merged[head], merge_log[head] = merge_alias_evidence([evidences[h] for h in members], head)
    complete = all(head in merged for head in canon_models)
    specs = {h: by_hash[h] for h in canon_models}
    if mc_weights is not None:
        # weights of a merged alias group: the head's, else any member's
        resolved = {}
        for head in canon_models:
            for member in (head, *aliases.groups[head]):
                if member in mc_weights:
                    resolved[head] = mc_weights[member]
                    break
        mc_weights = resolved
    # The per-edge inputs are keyed by the hashes of the graph nodes they were
    # computed on; the claim evaluation works on canonical (alias-merged)
    # endpoints, so the keys are canonicalized here the same way ``mc_weights``
    # are. Without this, every edge of an aliased pair reports "missing"/"fail"
    # for reasons unrelated to the data (the depth-1 production graph has no
    # aliases; the depth-2 graph has 8 groups covering 22 of its 102 nodes).
    prior_sensitivity = _canonicalize_edge_map(prior_sensitivity, canonical)
    stress = _canonicalize_edge_map(stress, canonical)

    # -- fidelity rungs -----------------------------------------------------
    rungs = {h: merged[h].rung for h in merged}
    distinct_rungs = sorted({r for r in rungs.values() if r is not None})
    rung_homogeneous = len(distinct_rungs) <= 1

    # -- model priors and point probabilities -------------------------------
    log_prior_all = normalized_log_prior([specs[h] for h in canon_models], root, model_prior)
    ns_sigma = {h: merged[h].ns_error(config.ns_error_mode, config.kappa_hat) for h in merged}
    models_out = {}
    posterior = prior = None
    if complete:
        lw = np.asarray([merged[h].log_evidence + log_prior_all[h] for h in canon_models])
        post = np.exp(lw - logsumexp(lw))
        posterior = dict(zip(canon_models, post.tolist()))
        prior = {h: math.exp(log_prior_all[h]) for h in canon_models}
    for h in canon_models:
        row = {
            "model_hash": h,
            "aliases": list(aliases.groups[h]),
            "log_model_prior": log_prior_all[h],
            "atoms": sorted(atoms_of(root, specs[h])),
            "evidence": None if h not in merged else merged[h].to_dict(),
            "ns_error": ns_sigma.get(h),
            "posterior_probability": None if posterior is None else posterior[h],
        }
        if mc_weights is not None and h in mc_weights:
            row["mc"] = mc_weights[h].summary()
        models_out[h] = row

    # -- MC covariance ------------------------------------------------------
    mc_available = mc_weights is not None and all(h in mc_weights for h in merged)
    sigma_mc = None
    psd_info = None
    if mc_available and merged:
        keys = sorted(merged)
        sigma_mc_matrix = mc_covariance_matrix([mc_weights[h] for h in keys])
        sigma_mc = (keys, sigma_mc_matrix)

    # -- edges ----------------------------------------------------------------
    edges_out = []
    seen = set()
    for edge in graph.edges:
        p, c = canonical[edge.parent_hash], canonical[edge.child_hash]
        key = (p, c, edge.mutation_id)
        if p == c:
            edges_out.append({"parent_hash": edge.parent_hash, "child_hash": edge.child_hash,
                              "mutation_id": edge.mutation_id, "skipped": "parent and child are aliases"})
            continue
        if key in seen:
            continue
        seen.add(key)
        row: dict[str, object] = {
            "parent_hash": p,
            "child_hash": c,
            "mutation_id": edge.mutation_id,
            "graph_parent_hash": edge.parent_hash,
            "graph_child_hash": edge.child_hash,
            "atom": edge_atom(root, specs[p], specs[c]),
            "log_prior_odds": log_prior_all[c] - log_prior_all[p],
        }
        if p not in merged or c not in merged:
            row["log_bayes_factor"] = None
            row["reason"] = "missing evidence"
            edges_out.append(row)
            continue
        lnbf = merged[c].log_evidence - merged[p].log_evidence
        mc = None
        if edge_mc is not None and (p, c) in edge_mc:
            mc = {"sigma": float(edge_mc[(p, c)]["sigma"]), "bias": float(edge_mc[(p, c)]["bias"]),
                  "source": "supplied"}
        elif mc_weights is not None and p in mc_weights and c in mc_weights:
            e = edge_mc_error(mc_weights[c], mc_weights[p])
            mc = {**e.to_dict(), "source": "omega_bar"}
        sig_ns = math.hypot(ns_sigma[p], ns_sigma[c])
        sig_mc = None if mc is None else _budget_sigma_mc(mc)
        total = math.sqrt(sig_ns**2 + (sig_mc or 0.0) ** 2)
        row.update(
            {
                "log_bayes_factor": lnbf,
                "sigma_ns_parent": ns_sigma[p],
                "sigma_ns_child": ns_sigma[c],
                "sigma_ns": sig_ns,
                "sigma_mc": sig_mc,
                "mc": mc,
                "sigma_total": total,
                "mc_error_included": mc is not None,
                "log_posterior_odds": lnbf + log_prior_all[c] - log_prior_all[p],
                "kass_raftery": kass_raftery(lnbf),
            }
        )
        edges_out.append(row)

    # -- propagation ------------------------------------------------------------
    propagation = None
    structural = None
    variants = []
    if complete:
        rng = np.random.default_rng(config.seed)
        keys = list(canon_models)
        mean = np.asarray([merged[h].log_evidence for h in keys])
        ns = np.asarray([ns_sigma[h] for h in keys])
        S = config.n_propagation
        draws = mean[None, :] + ns[None, :] * rng.standard_normal((S, len(keys)))
        if sigma_mc is not None:
            order = [sigma_mc[0].index(h) for h in keys]
            matrix = sigma_mc[1][np.ix_(order, order)]
            factor, psd_info = _psd_factor(matrix)
            draws = draws + rng.standard_normal((S, len(keys))) @ factor.T
        lp = np.asarray([log_prior_all[h] for h in keys])
        lw = draws + lp[None, :]
        probs = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
        propagation = {
            "n_draws": S,
            "includes_mc_covariance": sigma_mc is not None,
            "sigma_mc_psd": psd_info,
            "models": {h: _quantiles(probs[:, k]) for k, h in enumerate(keys)},
        }
        for h in keys:
            models_out[h]["posterior_probability_propagated"] = propagation["models"][h]
        structural = {"atoms": {}, "axes": {}}
        for level in ("atom", "axis"):
            masses = structural_masses(root, specs, posterior, prior, level=level)
            for label, row in masses.items():
                idx = [keys.index(m) for m in row["members"]]
                draw_mass = probs[:, idx].sum(axis=1)
                pri = row["prior_mass"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    lbf = np.log(draw_mass) - np.log1p(-draw_mass) - math.log(pri) + math.log1p(-pri)
                row = dict(row)
                row["posterior_mass_propagated"] = _quantiles(draw_mass)
                finite = np.isfinite(lbf)
                row["log_bayes_factor_propagated"] = (
                    _quantiles(lbf[finite]) if finite.sum() > 1 else None
                )
                structural["atoms" if level == "atom" else "axes"][label] = row
        for penalty in config.model_prior_lambdas:
            log_prior_v = normalized_log_prior([specs[h] for h in keys], root, complexity_prior(penalty))
            lw_v = np.asarray([merged[h].log_evidence + log_prior_v[h] for h in keys])
            post_v = dict(zip(keys, np.exp(lw_v - logsumexp(lw_v)).tolist()))
            prior_v = {h: math.exp(log_prior_v[h]) for h in keys}
            variants.append(
                {
                    "penalty_per_axis": penalty,
                    "posterior_model_probabilities": post_v,
                    "atoms": structural_masses(root, specs, post_v, prior_v, level="atom"),
                }
            )

    # -- search statistics -------------------------------------------------------
    evaluated = [e for e in edges_out if e.get("log_bayes_factor") is not None]
    search = {
        "max_edge_log_bayes_factor": max((e["log_bayes_factor"] for e in evaluated), default=None),
        "max_edge_log_posterior_odds": max((e["log_posterior_odds"] for e in evaluated), default=None),
    }
    if complete and graph.root_hash in posterior:
        p_root = posterior[graph.root_hash]
        pr_root = math.exp(log_prior_all[graph.root_hash])
        if 0.0 < p_root < 1.0 and 0.0 < pr_root < 1.0:
            search["T3_any_structure"] = (
                math.log1p(-p_root) - math.log(p_root) - math.log1p(-pr_root) + math.log(pr_root)
            )

    # -- claims -------------------------------------------------------------------
    claims = []
    for row in evaluated:
        claims.append(
            _evaluate_claim(
                row,
                merged=merged,
                mc_weights=mc_weights,
                structural=structural,
                variants=variants,
                sddr=sddr,
                prior_sensitivity=prior_sensitivity,
                null_calibrations=null_calibrations,
                stress=stress,
                nearby=nearby,
                loo=loo,
                criteria=crit,
                search_statistics=search,
                evaluated_edges=evaluated,
                evidence_complete=complete,
                rung_homogeneous=rung_homogeneous,
                rungs=rungs,
            )
        )

    report = {
        "format_version": REPORT_FORMAT,
        "graph_root_hash": graph.root_hash,
        "model_prior_version": getattr(model_prior, "version", None),
        "config": config.to_dict(),
        "deduplication": {
            **aliases.to_dict(),
            "merged_evidence": {k: v for k, v in merge_log.items() if len(v["merged"]) > 1},
            "inconsistent_alias_evidence": sorted(k for k, v in merge_log.items() if not v["consistent"]),
        },
        "evidence_coverage": {
            "n_effective_models": len(canon_models),
            "n_with_evidence": len(merged),
            "complete": complete,
            "rungs": {h: rungs[h] for h in sorted(rungs)},
            "distinct_rungs": distinct_rungs,
            "rung_homogeneous": rung_homogeneous,
            "rung_note": None if rung_homogeneous else (
                "the models were scored at more than one dynesty trajectory configuration; "
                "D4 computes the calibrated statistic from a single rung (F3), so ln BF, the "
                "error budget and the search statistic must not mix rungs"
            ),
            "missing": [h for h in canon_models if h not in merged],
            "note": None if complete else (
                "posterior model probabilities and structural masses are undefined over the "
                "declared graph until every effective model has valid evidence"
            ),
        },
        "models": models_out,
        "edges": edges_out,
        "propagation": propagation,
        "structural": structural,
        "model_prior_variants": variants,
        "search_statistics": search,
        "null_calibrations": [n.to_dict() for n in null_calibrations or ()],
        "claims": claims,
    }
    return json_ready(report)


def _evaluate_claim(
    row, *, merged, mc_weights, structural, variants, sddr, prior_sensitivity,
    null_calibrations, stress, nearby, loo, criteria: ClaimCriteria,
    search_statistics=None, evaluated_edges=(), evidence_complete=True,
    rung_homogeneous=True, rungs=None,
):
    p, c = row["parent_hash"], row["child_hash"]
    atom = row["atom"]
    lnbf = float(row["log_bayes_factor"])
    out: dict[str, dict] = {}

    # 1. numerics
    gates = [merged[p].gates_passed, merged[c].gates_passed]
    detail: dict[str, object] = {"production_gates": gates}
    status = "pass"
    if any(g is False for g in gates):
        status = "fail"
    elif any(g is None for g in gates):
        status = "missing"
    if mc_weights is not None and p in mc_weights and c in mc_weights:
        gmc = {}
        for key in (p, c):
            w = mc_weights[key]
            gmc[key] = {
                "G-MC1_expected_variance": w.expected_variance,
                "G-MC1": w.expected_variance <= criteria.gmc_max_expected_variance,
                "G-MC2_neff_fraction": w.selection_neff_ok_fraction,
                "G-MC2": w.selection_neff_ok_fraction >= criteria.gmc_min_neff_fraction,
                "G-MC3_c_pp": w.c_pp(),
                "G-MC3": w.c_pp() <= criteria.gmc_max_c_pp,
            }
        mc = row.get("mc") or {}
        # G-MC4 gates the same sigma_MC the error budget uses: when the unbiased
        # U-statistic lands below zero, EdgeMCError.sigma reports 0 and gating on
        # it would pass on a value known to be wrong, while sigma_total carries
        # the (non-negative) plug-in. See _budget_sigma_mc.
        sigma_mc = None if not mc else _budget_sigma_mc(mc)
        edge_ok = {
            "G-MC4_sigma": sigma_mc,
            "G-MC4_sigma_u_statistic": mc.get("sigma"),
            "G-MC4": sigma_mc is not None and sigma_mc <= criteria.gmc_max_edge_sigma,
            "G-MC5_bias": mc.get("bias"),
            "G-MC5": mc.get("bias") is not None and abs(mc["bias"]) <= criteria.gmc_max_edge_bias,
        }
        detail["gmc_models"] = gmc
        detail["gmc_edge"] = edge_ok
        passed = all(v["G-MC1"] and v["G-MC2"] and v["G-MC3"] for v in gmc.values()) and edge_ok["G-MC4"] and edge_ok["G-MC5"]
        if not passed:
            status = "fail"
    elif status != "fail":
        status = "missing" if status == "pass" else status
        detail["gmc"] = "Monte-Carlo weights not supplied"
    # D4 computes the calibrated statistic from one fidelity rung; mixing rungs
    # across models makes ln BF, the error budget and the search statistic a
    # mixture of two procedures, so the edge is not a valid numerical comparison.
    detail["fidelity_rungs"] = None if rungs is None else {"parent": rungs.get(p), "child": rungs.get(c)}
    detail["rung_homogeneous"] = bool(rung_homogeneous)
    if not rung_homogeneous:
        detail["rung_note"] = (
            "the evidences of this graph come from more than one dynesty trajectory "
            "configuration (fidelity rung); ln BF and the search statistic mix procedures"
        )
        status = "fail"
    out["numerics"] = _criterion(status, **detail)

    # 2. strength
    lower = lnbf - criteria.strength_sigmas * float(row["sigma_total"])
    atom_row = None if structural is None else structural["atoms"].get(atom)
    struct_lbf = None if atom_row is None else atom_row.get("log_bayes_factor")
    strength = {
        "log_bayes_factor": lnbf,
        "sigma_total": row["sigma_total"],
        "mc_error_included": row["mc_error_included"],
        "lower": lower,
        "structural_log_bayes_factor": struct_lbf,
    }
    if lower < criteria.strength_threshold:
        s = "fail"
    elif struct_lbf is None:
        s = "missing"
    elif struct_lbf < criteria.structural_threshold:
        s = "fail"
    elif not row["mc_error_included"]:
        s = "incomplete"
    else:
        s = "pass"
    out["strength"] = _criterion(s, **strength)

    # 3. prior robustness
    #
    # The two sub-checks are kept apart: an absent input is "missing" (the
    # criterion was not evaluated) and a present value that violates a threshold
    # is "fail" (it was evaluated and the data did not meet it). Both block a
    # claim through :func:`_combine`, but reporting a missing input as a
    # scientific failure misdescribes the report.
    detail = {}
    status = "pass"
    sens = None if prior_sensitivity is None else prior_sensitivity.get((p, c))
    if sens is None:
        status = "missing"
        detail["width_variants"] = None
    else:
        values = [v.get("log_bayes_factor") for v in sens.get("variants", [])]
        detail["width_variants"] = sens.get("variants", [])
        if not values:
            status = "missing"
        elif any(v is None for v in values):
            # a widened prior whose posterior is not contained, or a narrowing
            # whose reweighting ESS was too small to trust: not evaluated
            status = "incomplete"
        if any(v is not None and (v < criteria.robustness_min_log_bf or np.sign(v) != np.sign(lnbf)) for v in values):
            status = "fail"
    masses = []
    missing_mass = False
    for variant in variants:
        m = variant["atoms"].get(atom, {}).get("posterior_mass")
        masses.append({"penalty_per_axis": variant["penalty_per_axis"], "posterior_mass": m})
        if m is None:
            # the atom carries no structural mass in this variant (e.g. a
            # ``not:`` removal label, which structural_masses never keys): the
            # check could not be made, it did not fail
            missing_mass = True
        elif m < criteria.robustness_min_posterior_mass:
            status = "fail"
    if not variants:
        status = "missing" if status == "pass" else status
    elif missing_mass:
        detail["model_prior_variants_note"] = f"atom {atom!r} carries no structural mass in the variants"
        if status == "pass":
            status = "missing"
    detail["model_prior_variants"] = masses
    out["prior_robustness"] = _criterion(status, **detail)

    # 4. SDDR
    check = None
    if sddr is not None:
        for key in (f"{row['graph_parent_hash']}->{row['graph_child_hash']}", f"{p}->{c}", row["mutation_id"]):
            if key in sddr:
                check = sddr[key]
                break
    if check is None:
        out["sddr"] = _criterion("missing", reason="no SDDR cross-check supplied")
    else:
        st = check.get("status")
        # ``not_estimable`` is a *method* failure of an SDDR-eligible edge (the
        # VW factor's support condition fails, too few posterior draws near the
        # null, or the density is not estimable with no NS bound to compare) --
        # not "this edge has no SDDR route". Scoring it ``not_applicable`` would
        # let an eligible edge be claimed with no cross-check at all, which
        # ``mass.family.pl_two_peak`` (vw_required, vw_first_form_valid false)
        # hits deterministically on the production graph. Only an edge the
        # grammar classifies ``evidence_only`` is genuinely not applicable.
        eligible = check.get("classification") in {"exact", "approximate"}
        mapping = {"agree": "pass", "bound_consistent": "pass", "disagree": "fail",
                   "bound_violated": "fail", "computed": "incomplete"}
        if st in {"not_applicable", "not_estimable"}:
            sddr_status = "missing" if eligible else "not_applicable"
        else:
            sddr_status = mapping.get(st, "missing")
        sddr_detail: dict[str, object] = {
            "check_status": st,
            "classification": check.get("classification"),
        }
        reason = check.get("reason") or (check.get("details") or {}).get("reason")
        if reason is not None:
            sddr_detail["reason"] = reason
        if sddr_status == "missing" and st in {"not_applicable", "not_estimable"}:
            sddr_detail["note"] = (
                "SDDR-eligible edge whose cross-check could not be computed: the edge has no "
                "independent confirmation and rests on the evidence route alone"
            )
        if "method_systematic" in check:
            # how wide the agreement band was, and where that width came from:
            # "formula8" (the bare test), "supplied" or "measured" (the post-hoc
            # floor of SDDR_METHOD_SYSTEMATIC, which is not a validated systematic)
            sddr_detail["method_systematic"] = check.get("method_systematic")
            sddr_detail["method_systematic_source"] = check.get("method_systematic_source")
        for key in ("difference", "tolerance", "log_bf_child_over_parent_sddr",
                    "log_bf_child_over_parent_ns"):
            if key in check:
                sddr_detail[key] = check.get(key)
        out["sddr"] = _criterion(sddr_status, **sddr_detail)

    # 5. search-level null calibration
    #
    # D4 computes the calibrated statistic from the identical F3 procedure, so a
    # supplied calibration must describe *this* report: its statistic has to be
    # one the report computes and its observed value has to agree with the
    # report's own. Otherwise a stale campaign, one calibrated for a different
    # statistic, or one run on a different graph satisfies the criterion in
    # silence.
    if not null_calibrations:
        out["null_calibration"] = _criterion("missing", reason="no null replays supplied")
    else:
        rows, status = [], "pass"
        stats = dict(search_statistics or {})
        for null in null_calibrations:
            limited = null.resolution > criteria.alpha
            ok = null.p_value <= criteria.alpha
            computed = stats.get(null.statistic)
            tol = None if computed is None else _null_observed_tolerance(
                null.statistic, evaluated_edges, criteria
            )
            matches = None if computed is None else bool(abs(null.observed - float(computed)) <= tol)
            rows.append({
                **null.to_dict(),
                "resolution_limited": limited,
                "at_resolution_floor": null.exceedances == 0,
                "report_statistic": computed,
                "observed_tolerance": tol,
                "matches_report_statistic": matches,
                "passed": bool(ok and not limited and matches),
            })
            if limited or not ok or matches is False:
                status = "fail"
            elif matches is None and status == "pass":
                # the report does not compute this statistic, so the calibration
                # cannot be tied to it
                status = "missing"
        if status == "pass" and not evidence_complete:
            # the observed statistic is a max over the subset of edges that have
            # evidence, not over the graph the nulls were calibrated on
            status = "incomplete"
        out["null_calibration"] = _criterion(
            status, alpha=criteria.alpha, nulls=rows,
            evidence_coverage_complete=bool(evidence_complete),
        )

    # 6. stress / LOO / nearby
    detail, status = {}, "pass"
    stressed = None if stress is None else stress.get((p, c))
    near = None if nearby is None else nearby.get(row["mutation_id"])
    loo_edge = None
    if loo is not None:
        # a PSIS-LOO report keys its edges by the graph's hashes, like the
        # prior-sensitivity and stress inputs; accept either spelling
        keys = {(p, c), (row["graph_parent_hash"], row["graph_child_hash"])}
        for e in loo.get("edges", []):
            if (e.get("parent_hash"), e.get("child_hash")) in keys:
                loo_edge = e
    if stressed is None and near is None and loo_edge is None:
        status = "missing"
    for label, values in (("event_drop", stressed), ("nearby_baseline", near)):
        if values is None:
            detail[label] = None
            status = "incomplete" if status == "pass" else status
            continue
        reversed_ = [float(v) for v in values if np.sign(float(v)) != np.sign(lnbf)]
        detail[label] = {"values": [float(v) for v in values], "sign_reversals": reversed_}
        if reversed_:
            status = "fail"
    if loo_edge is None:
        detail["leave_one_out"] = None
        status = "incomplete" if status == "pass" else status
    else:
        detail["leave_one_out"] = {
            "any_sign_reversal": loo_edge.get("any_sign_reversal"),
            "n_flagged_events": loo_edge.get("n_flagged_events"),
            "max_abs_delta_log_bayes_factor": loo_edge.get("max_abs_delta_log_bayes_factor"),
        }
        if loo_edge.get("any_sign_reversal"):
            status = "fail"
        elif loo_edge.get("n_flagged_events"):
            status = "incomplete" if status == "pass" else status
    out["stress"] = _criterion(status, **detail)

    return {
        "parent_hash": p,
        "child_hash": c,
        "mutation_id": row["mutation_id"],
        "atom": atom,
        "direction": "child" if lnbf > 0 else "parent",
        "criteria": out,
        "status": _combine(out),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value, digits=3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return str(v)
    return f"{v:.{digits}f}"


def _fmt_p(value) -> str:
    """Probabilities: three significant digits so that small masses stay visible."""
    if value is None:
        return "–"
    v = float(value)
    return f"{v:.3g}" if math.isfinite(v) else str(v)


def render_markdown(report: Mapping[str, object], *, labels: Mapping[str, str] | None = None) -> str:
    """Markdown rendering of :func:`build_model_comparison` output."""
    labels = dict(labels or {})

    def name(h):
        return labels.get(h, str(h)[:12])

    lines = ["# Model comparison", ""]
    cov = report["evidence_coverage"]
    dedup = report["deduplication"]
    lines += [
        f"- Graph root: `{str(report['graph_root_hash'])[:16]}`; model prior: `{report['model_prior_version']}`",
        f"- Effective models: {dedup['n_effective_models']} (graph nodes: {dedup['n_nodes']}; "
        f"alias groups: {len(dedup['alias_groups'])})",
        f"- Evidence coverage: {cov['n_with_evidence']}/{cov['n_effective_models']}"
        + ("" if cov["complete"] else " (incomplete: model probabilities undefined)"),
        "- Fidelity rung: "
        + (f"`{cov['distinct_rungs'][0]}`" if cov.get("distinct_rungs") else "not recorded")
        + ("" if cov.get("rung_homogeneous", True) else
           f" — **MIXED across models**: {cov['distinct_rungs']}"),
        f"- NS error mode: `{report['config']['ns_error_mode']}`",
        "",
        "## Models",
        "",
        "| model | ln Z | σ_NS | p(M) prior | p(M\\|D) | p(M\\|D) 16–84% | E_P[V] | C_PP |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h, row in report["models"].items():
        ev = row.get("evidence") or {}
        prop = row.get("posterior_probability_propagated") or {}
        mc = row.get("mc") or {}
        lines.append(
            f"| {name(h)} | {_fmt(ev.get('log_evidence'))} | {_fmt(row.get('ns_error'))} | "
            f"{_fmt_p(math.exp(row['log_model_prior']))} | {_fmt_p(row.get('posterior_probability'))} | "
            f"{_fmt_p(prop.get('q16'))}–{_fmt_p(prop.get('q84'))} | {_fmt(mc.get('expected_variance'))} | "
            f"{_fmt(mc.get('c_pp'))} |"
        )
    lines += [
        "",
        "## Edges",
        "",
        "| mutation | ln BF | σ_NS | σ_MC | σ_total | MC bias | ln odds | Kass–Raftery |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in report["edges"]:
        if e.get("skipped") or e.get("log_bayes_factor") is None:
            continue
        mc = e.get("mc") or {}
        lines.append(
            f"| {e['mutation_id']} | {_fmt(e['log_bayes_factor'])} | {_fmt(e['sigma_ns'])} | "
            f"{_fmt(e['sigma_mc'])} | {_fmt(e['sigma_total'])} | {_fmt(mc.get('bias'))} | "
            f"{_fmt(e['log_posterior_odds'])} | {e['kass_raftery']} |"
        )
    structural = report.get("structural")
    if structural:
        lines += [
            "",
            "## Structural atoms",
            "",
            "| atom | P(S) | P(S\\|D) | P(S\\|D) 16–84% | ln BF_S |",
            "|---|---|---|---|---|",
        ]
        for label, row in structural["atoms"].items():
            prop = row.get("posterior_mass_propagated") or {}
            lines.append(
                f"| {label} | {_fmt_p(row['prior_mass'])} | {_fmt_p(row['posterior_mass'])} | "
                f"{_fmt_p(prop.get('q16'))}–{_fmt_p(prop.get('q84'))} | {_fmt(row.get('log_bayes_factor'))} |"
            )
    lines += ["", "## Claims", "", "| mutation | status | numerics | strength | prior | SDDR | null | stress |",
              "|---|---|---|---|---|---|---|---|"]
    for claim in report["claims"]:
        crit = claim["criteria"]
        lines.append(
            f"| {claim['mutation_id']} | **{claim['status']}** | {crit['numerics']['status']} | "
            f"{crit['strength']['status']} | {crit['prior_robustness']['status']} | "
            f"{crit['sddr']['status']} | {crit['null_calibration']['status']} | {crit['stress']['status']} |"
        )
    search = report.get("search_statistics") or {}
    lines += [
        "",
        "## Search statistics",
        "",
        f"- max edge ln BF: {_fmt(search.get('max_edge_log_bayes_factor'))}",
        f"- max edge ln posterior odds: {_fmt(search.get('max_edge_log_posterior_odds'))}",
        f"- T3 (any structure vs root, prior-odds corrected): {_fmt(search.get('T3_any_structure'))}",
    ]
    # whether each supplied calibration describes this report (criterion 5)
    checked = {}
    for claim in report["claims"]:
        for entry in (claim["criteria"]["null_calibration"].get("nulls") or []):
            checked[(entry["label"], entry["statistic"])] = entry
    for null in report.get("null_calibrations") or []:
        entry = checked.get((null["label"], null["statistic"]), {})
        matches = entry.get("matches_report_statistic")
        if matches is None:
            note = " — statistic not computed by this report"
        elif matches:
            note = ""
        else:
            note = (f" — **observed {_fmt(null['observed'])} does not match this report's "
                    f"{_fmt(entry.get('report_statistic'))}**")
        lines.append(
            f"- null `{null['label']}` ({null['statistic']}): p = {null['p_value']:.4g} "
            f"({null['exceedances']}/{null['replays']}; resolution {null['resolution']:.4g}){note}"
        )
    return "\n".join(lines) + "\n"


def evidence_summaries_from_results(
    grouped: Mapping[str, Sequence], *, gates: Mapping[str, bool] | None = None
) -> dict[str, EvidenceSummary]:
    return {
        h: EvidenceSummary.from_dynesty_results(h, results, gates_passed=None if gates is None else gates.get(h))
        for h, results in grouped.items()
    }


def stress_values_from_summaries(summaries: Iterable[Mapping[str, object]]) -> dict[tuple[str, str], list[float]]:
    """Stressed edge ln BF values from event-stress suite summaries."""
    out: dict[tuple[str, str], list[float]] = {}
    for summary in summaries:
        for scenario in summary.get("scenarios", []):
            for row in (scenario.get("edge_bayes_factors") or {}).values():
                key = (str(row["parent_hash"]), str(row["child_hash"]))
                out.setdefault(key, []).append(float(row["log_bayes_factor"]))
    return out


def nearby_values_from_summaries(summaries: Iterable[Mapping[str, object]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for summary in summaries:
        for scenario in summary.get("scenarios", []):
            for mutation, value in (scenario.get("mutation_max_log_bayes_factors") or {}).items():
                out.setdefault(str(mutation), []).append(float(value))
    return out


__all__ = [
    "ClaimCriteria",
    "ComparisonConfig",
    "EvidenceSummary",
    "NullCalibration",
    "REPORT_FORMAT",
    "build_model_comparison",
    "evidence_summaries_from_results",
    "kass_raftery",
    "merge_alias_evidence",
    "nearby_values_from_summaries",
    "render_markdown",
    "stress_values_from_summaries",
    "structural_bayes_factor",
]
