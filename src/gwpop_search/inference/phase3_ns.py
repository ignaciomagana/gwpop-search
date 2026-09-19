"""Phase-3 v2: multi-catalog synthetic recovery on dynesty nested sampling.

The NumPyro-NUTS Phase-3 recovery (``inference.recovery`` and
``inference.campaign``, formats ``gwpop-search-phase3-recovery-1.0`` and
``gwpop-search-phase3-campaign-1.0``) was rejected on 2026-09-19 and is kept
unchanged for the historical record. This module replaces it. Nothing about
the likelihood, the population model, the hyperpriors or the data semantics
changes: every catalog is fitted with ``R`` independent static dynesty runs of
the standardized HBI shape likelihood through
:func:`gwpop_search.inference.dynesty_backend.run_dynesty_population`, and the
numerical gates are nested-sampling diagnostics.

Campaign
--------
``n_runs`` catalogs. Catalog ``i`` uses the v1 seed pair
``recovery_seed_pairs(n_runs, root_seed)[i] = (data_seed, sampler_seed)``, so
the catalog data seeds of the rejected NUTS campaign are reused exactly. Its
``R`` repeats use independent sampler seeds
``_derived_seed(sampler_seed, "dynesty-repeat", r)``. Every repeat has its own
run directory (manifest, dynesty checkpoint, result); a repeat resumes from its
checkpoint after a kill and a completed repeat is reused untouched (its files
are never rewritten).

The campaign plan (``gwpop-search-phase3-campaign-plan-2.0``) pins the seeds,
the survey, the dynesty configuration (every field except the I/O cadence
``checkpoint_every``), the HBI configuration, the model, the priors, the truth,
the diagnostic settings and the acceptance criteria. A root whose plan differs
is refused.

Per-catalog summary (``gwpop-search-phase3-recovery-2.0``)
---------------------------------------------------------
* Pooled posterior: the equal-weight mixture of the ``R`` separately
  normalized runs, i.e. the concatenation of each run's
  ``num_posterior_samples`` equal-weight draws (never ``merge_runs``).
  Quantiles 5/50/95, mean, std, truth-in-90%-interval and the standardized
  offset ``(median - truth) / std``.
* Truth-rank diagnostics of the survey (:func:`pe_truth_quantiles`).
* Per run: log evidence, reported error, information, niter, ncall,
  efficiency, Kish ESS, elapsed time, evaluation counters and dlogz
  termination.
* Evidence repeats: mean, std (ddof 1), maximum reported error, the
  conservative error ``max(std, max error)`` and the maximum pairwise
  ``|lnZ_r - lnZ_s| / sqrt(err_r^2 + err_s^2)``.
* Convergence: rank-normalized split R-hat (Vehtari et al. 2021) per
  parameter with the runs as chains; each chain is ``M = min(max draws, floor
  of the smallest Kish ESS)`` equal-weight draws of its run (systematic
  resampling followed by a random permutation).
* Importance diagnostics with the backend's jitted diagnostics and the runs'
  own HBI configuration (verified against every run's likelihood identity) at
  the coordinate-wise median of the pooled posterior, over ``n`` seeded pooled
  draws (quantiles, ``inverted_cdf``) and, reported only, at the truth.

Acceptance (:class:`NSRecoveryAcceptanceCriteria`)
--------------------------------------------------
Each catalog must pass every sampler, evidence and importance check; the
importance thresholds are enforced at the posterior-median point, at the median
over the pooled draws and at the tail (the ``tail_ess_quantile`` quantile of
ESS-type metrics, the ``tail_weight_quantile`` quantile of weights and
``Var(log L)``). Ensemble coverage and standardized offsets are reported, not
thresholded.

Fingerprints
------------
:func:`ns_run_fingerprints` hashes the completed-run artifacts
(``manifest.json``, ``result.npz``, ``result.npz.json``) of every dynesty run
directory below a path; checkpoints are excluded. A resume must leave every
previously fingerprinted file unchanged (:func:`compare_ns_fingerprints`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
import json
import math
import numbers
from pathlib import Path
from typing import ClassVar, Iterable, Mapping, Sequence

import numpy as np

from .campaign import _derived_seed, file_sha256, recovery_seed_pairs
from .dynesty_backend import (
    DynestyConfig,
    DynestyResult,
    _atomic_write_text,
    _json_ready,
    _json_sha256,
    _manifest_differences,
    _without_chunk_size,
    build_importance_diagnostics,
    run_dynesty_population,
)
from .priors import BASELINE_SYNTHETIC_PRIORS, serialize_prior_map

PLAN_FORMAT_VERSION = "gwpop-search-phase3-campaign-plan-2.0"
RECOVERY_SUMMARY_FORMAT_VERSION = "gwpop-search-phase3-recovery-2.0"
CAMPAIGN_SUMMARY_FORMAT_VERSION = "gwpop-search-phase3-campaign-2.0"
NS_FINGERPRINT_FORMAT_VERSION = "gwpop-search-ns-run-fingerprints-1.0"

PHASE3_ROOT_SEED = 20260917
DEFAULT_NS_REPEATS = 4
DEFAULT_IMPORTANCE_DRAWS = 512
DEFAULT_RHAT_DRAWS_PER_RUN = 2000
DEFAULT_IMPORTANCE_BATCH_SIZE = 64

# Quantiles of every importance statistic over the pooled posterior draws.
IMPORTANCE_QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
QUANTILE_METHOD = "inverted_cdf"

# Importance metrics gated by the criteria: (summary key, direction, threshold field).
_GATED_IMPORTANCE = (
    ("min_event_ess", "ge", "min_event_ess"),
    ("selection_ess", "ge", "min_selection_ess"),
    ("max_event_weight_fraction", "le", "max_event_weight_fraction"),
    ("selection_max_weight_fraction", "le", "max_selection_weight_fraction"),
    ("shape_log_likelihood_variance", "le", "max_shape_log_likelihood_variance"),
)

RUN_DIR_PREFIX = "run_"
REPEATS_DIRNAME = "repeats"
REPEAT_DIR_PREFIX = "repeat_"
RECOVERY_SUMMARY_NAME = "recovery_summary.json"
POOLED_POSTERIOR_NAME = "posterior_pooled.npz"
CAMPAIGN_PLAN_NAME = "campaign_plan.json"
CAMPAIGN_SUMMARY_NAME = "campaign_summary.json"

# Completed-run artifacts of one dynesty run directory (checkpoints excluded).
_FINGERPRINTED_RUN_FILES = ("manifest.json", "result.npz", "result.npz.json")
# Plan files that a resume must never rewrite.
_FINGERPRINTED_PLAN_FILES = (CAMPAIGN_PLAN_NAME, "evidence_check_plan.json")


# ---------------------------------------------------------------------------
# Defaults (orchestrator decisions D1, D5)
# ---------------------------------------------------------------------------


def slices_for_ndim(ndim: int) -> int:
    """Default rslice ``slices`` for ``ndim`` parameters: ``2 * (3 + ndim)``."""
    ndim = int(ndim)
    if ndim < 1:
        raise ValueError("ndim must be positive")
    return 2 * (3 + ndim)


def default_phase3_survey_config():
    """Survey v2 recommended for the Phase-3 re-run (docs/phase3_recovery.md)."""
    from .synthetic import (
        INJECTION_DRAW_POPULATION_PROXY,
        OBSERVATION_MODEL_NOISY,
        RECOMMENDED_POPULATION_PROXY_N_INJECTIONS,
        SyntheticSurveyConfig,
    )

    return SyntheticSurveyConfig(
        n_events=48,
        posterior_samples_per_event=1024,
        n_injections=RECOMMENDED_POPULATION_PROXY_N_INJECTIONS,
        injection_draw=INJECTION_DRAW_POPULATION_PROXY,
        observation_model=OBSERVATION_MODEL_NOISY,
    )


def default_phase3_dynesty_config(ndim: int | None = None, **overrides) -> DynestyConfig:
    """Static dynesty settings of the Phase-3 v2 campaign.

    ``nlive=1000``, ``bound="multi"``, ``sample="rslice"``,
    ``slices=2*(3+ndim)`` (26 for the 10-parameter baseline), ``dlogz=0.1``,
    ``batch_size=64``; every other field keeps the dynesty default.
    """
    ndim = len(BASELINE_SYNTHETIC_PRIORS) if ndim is None else int(ndim)
    payload = {
        "nlive": 1000,
        "bound": "multi",
        "sample": "rslice",
        "slices": slices_for_ndim(ndim),
        "dlogz": 0.1,
        "batch_size": 64,
    }
    payload.update(overrides)
    return DynestyConfig(**payload)


def phase3_hbi_config():
    """HBI configuration of every dynesty evaluation: one selection chunk (D5)."""
    from gwpop_search.hbi import HBIConfig

    return HBIConfig(selection_chunk_size=None)


def repeat_seeds(sampler_seed: int, repeats: int) -> tuple[int, ...]:
    """Independent dynesty seeds of a catalog's repeats (sampler seed, repeat index)."""
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return tuple(_derived_seed(int(sampler_seed), "dynesty-repeat", r) for r in range(repeats))


def plan_dynesty_config(config: DynestyConfig) -> dict[str, object]:
    """What a plan pins of a dynesty configuration: every field but ``checkpoint_every``.

    ``checkpoint_every`` is the I/O cadence; it changes neither the sampling
    trajectory nor the result, so a campaign may be resumed with another value.
    """
    if not isinstance(config, DynestyConfig):
        raise TypeError("config must be a DynestyConfig")
    payload = config.to_dict()
    payload.pop("checkpoint_every")
    return payload


def _hbi_config_payload(hbi_config) -> dict[str, object]:
    from .numpyro import _hbi_config_dict

    return _hbi_config_dict(hbi_config)


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _strict_json(value):
    """``value`` with non-finite floats written as ``null`` (strict JSON).

    Summaries are reports: a non-finite diagnostic is recorded as ``null`` and
    every check treats ``null`` as a failure.
    """
    value = _json_ready(value)
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload) -> None:
    _atomic_write_text(
        Path(path), json.dumps(_strict_json(payload), sort_keys=True, indent=2, allow_nan=False)
    )


# ---------------------------------------------------------------------------
# Rank-normalized split R-hat (Vehtari, Gelman, Simpson, Carpenter & Buerkner 2021)
# ---------------------------------------------------------------------------


def _z_scale(values: np.ndarray) -> np.ndarray:
    """Rank-normalize with average ranks and Blom's offset ``(r - 3/8) / (S + 1/4)``."""
    from scipy.stats import norm, rankdata

    values = np.asarray(values, dtype=float)
    ranks = rankdata(values, method="average", axis=None).reshape(values.shape)
    return norm.ppf((ranks - 0.375) / (values.size + 0.25))


def _classic_rhat(chains: np.ndarray) -> float:
    """``sqrt((B/W + n - 1) / n)`` with ``B = n Var(chain means)`` and ``W = mean chain var``."""
    n = chains.shape[1]
    between = n * np.var(np.mean(chains, axis=1), ddof=1)
    within = np.mean(np.var(chains, axis=1, ddof=1))
    if not (np.isfinite(within) and within > 0.0 and np.isfinite(between)):
        return float("nan")
    return float(np.sqrt((between / within + n - 1.0) / n))


def rank_normalized_split_rhat(chains) -> dict[str, float]:
    """Rank-normalized split R-hat of ``chains`` with shape ``[n_chains, n_draws]``.

    Each chain is split into its first and last ``n_draws // 2`` draws. ``bulk``
    is the classic R-hat of the rank-normalized split chains, ``tail`` the same
    for the folded draws ``|x - median|``, and ``rank = max(bulk, tail)`` (the
    Vehtari et al. 2021 recommendation, as ``arviz.rhat(method="rank")``). A
    parameter without within-chain variance gives NaN.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError(f"chains must have shape [n_chains, n_draws]; got {chains.shape}")
    n_chains, n_draws = chains.shape
    if n_chains < 2 or n_draws < 4:
        raise ValueError("rank-normalized split R-hat needs >= 2 chains of >= 4 draws")
    if not np.all(np.isfinite(chains)):
        raise ValueError("chains contain non-finite values")
    half = n_draws // 2
    split = np.vstack((chains[:, :half], chains[:, -half:]))
    bulk = _classic_rhat(_z_scale(split))
    tail = _classic_rhat(_z_scale(np.abs(split - np.median(split))))
    rank = float("nan") if (math.isnan(bulk) or math.isnan(tail)) else max(bulk, tail)
    return {"bulk": bulk, "tail": tail, "rank": rank}


# ---------------------------------------------------------------------------
# Nested-sampling fit summaries (R runs of one model on one catalog)
# ---------------------------------------------------------------------------


def _common_names(results: Sequence[DynestyResult]) -> tuple[str, ...]:
    if not results:
        raise ValueError("at least one dynesty result is required")
    names = tuple(results[0].names)
    for result in results[1:]:
        if tuple(result.names) != names:
            raise ValueError("dynesty results have different parameter names")
    return names


def data_identity_digest(results: Sequence[DynestyResult]) -> dict[str, object]:
    """PE/selection digests of the catalog every run sampled (they must agree)."""
    identities = [dict(result.likelihood_identity or {}).get("data") for result in results]
    if not identities or any(item is None for item in identities):
        raise ValueError("every run must carry the likelihood identity of its data")
    if any(item != identities[0] for item in identities[1:]):
        raise ValueError("the runs sampled different catalogs")
    data = dict(identities[0])
    return {
        "pe_sha256": data["pe_sha256"],
        "selection_sha256": data["selection_sha256"],
        "n_events": int(data["n_events"]),
        "n_pe_samples": int(data["n_pe_samples"]),
        "n_selected": int(data["n_selected"]),
    }


def pooled_posterior_draws(results: Sequence[DynestyResult]) -> np.ndarray:
    """Equal-weight mixture of separately normalized runs, ``[R * n, ndim]``.

    Each run contributes its own ``num_posterior_samples`` equal-weight draws;
    the runs must have the same number of draws so the mixture weights are
    equal.
    """
    _common_names(results)
    sizes = {int(np.asarray(result.posterior_samples).shape[0]) for result in results}
    if len(sizes) != 1:
        raise ValueError(
            f"runs have different numbers of equal-weight draws {sorted(sizes)}; the "
            "pooled posterior is an equal-weight mixture of the runs"
        )
    return np.concatenate([np.asarray(r.posterior_samples, dtype=float) for r in results])


def cross_run_rhat(
    results: Sequence[DynestyResult],
    *,
    max_draws_per_run: int = DEFAULT_RHAT_DRAWS_PER_RUN,
) -> dict[str, object]:
    """Rank-normalized split R-hat per parameter with the runs as chains.

    Every run is resampled to ``M = max(4, min(max_draws_per_run, floor(min
    Kish ESS)))`` equal-weight draws (``DynestyResult.with_num_posterior_samples``:
    seeded systematic resampling followed by a random permutation, so the two
    halves of a chain are exchangeable).
    """
    results = list(results)
    names = _common_names(results)
    if len(results) < 2:
        raise ValueError("cross-run R-hat needs at least two runs")
    max_draws_per_run = int(max_draws_per_run)
    if max_draws_per_run < 4:
        raise ValueError("max_draws_per_run must be >= 4")
    min_kish = min(float(result.kish_ess) for result in results)
    draws = int(max(4, min(max_draws_per_run, math.floor(min_kish))))
    chains = np.stack(
        [
            np.asarray(result.with_num_posterior_samples(draws).posterior_samples, dtype=float)
            for result in results
        ]
    )
    per_parameter = {
        name: rank_normalized_split_rhat(chains[:, :, k]) for k, name in enumerate(names)
    }
    ranks = [item["rank"] for item in per_parameter.values()]
    max_rhat = None if any(math.isnan(value) for value in ranks) else float(max(ranks))
    return {
        "method": "rank-normalized split R-hat (Vehtari et al. 2021), runs as chains",
        "n_chains": len(results),
        "draws_per_run": draws,
        "per_parameter": {
            name: {key: _finite_or_none(value) for key, value in item.items()}
            for name, item in per_parameter.items()
        },
        "max_r_hat": max_rhat,
    }


def evidence_repeat_statistics(results: Sequence[DynestyResult]) -> dict[str, object]:
    """Repeat statistics of the log evidences of independent runs of one model."""
    results = list(results)
    _common_names(results)
    estimates = np.asarray([float(r.log_evidence) for r in results])
    errors = np.asarray([float(r.log_evidence_error) for r in results])
    information = np.asarray([float(r.information) for r in results])
    nlive = np.asarray([int(r.config.nlive) for r in results], dtype=float)
    n = len(results)
    repeat_std = float(np.std(estimates, ddof=1)) if n > 1 else None
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            scale = math.hypot(errors[a], errors[b])
            diff = abs(estimates[a] - estimates[b])
            z = diff / scale if scale > 0.0 else (0.0 if diff == 0.0 else float("inf"))
            pairs.append({"a": a, "b": b, "z": _finite_or_none(z)})
    finite_z = [item["z"] for item in pairs]
    max_z = None if (not pairs or any(z is None for z in finite_z)) else float(max(finite_z))
    max_error = float(np.max(errors))
    predicted = np.sqrt(np.maximum(information, 0.0) / nlive)
    return {
        "n_repeats": n,
        "estimates": estimates.tolist(),
        "reported_errors": errors.tolist(),
        "mean": float(np.mean(estimates)),
        "repeat_std": repeat_std,
        "mean_reported_error": float(np.mean(errors)),
        "max_reported_error": max_error,
        "conservative_error": (
            None if repeat_std is None else float(max(repeat_std, max_error))
        ),
        "pairwise_z": pairs,
        "max_pairwise_z": max_z,
        "predicted_error_sqrt_h_over_nlive": [float(v) for v in predicted],
    }


def _run_record(result: DynestyResult, *, repeat: int, run_dir: str | None) -> dict[str, object]:
    diagnostics = dict(result.diagnostics)
    converged = diagnostics.get("converged")
    return {
        "repeat": int(repeat),
        "seed": int(result.seed),
        "run_dir": run_dir,
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_error),
        "information": float(result.information),
        "predicted_log_evidence_error": float(
            math.sqrt(max(float(result.information), 0.0) / int(result.config.nlive))
        ),
        "niter": int(result.niter),
        "ncall": int(result.ncall),
        "efficiency": float(result.efficiency),
        "kish_ess": float(result.kish_ess),
        "elapsed_seconds": float(result.elapsed_seconds),
        "n_likelihood_evaluations": result.n_likelihood_evaluations,
        "n_selection_unsupported": result.n_selection_unsupported,
        "converged": None if converged is None else bool(converged),
        "final_delta_logz": _finite_or_none(diagnostics.get("final_delta_logz")),
        "n_zero_likelihood_points": diagnostics.get("n_zero_likelihood_points"),
        "manifest_sha256": (result.provenance.get("run") or {}).get("manifest_sha256"),
    }


def _importance_statistics(batch) -> dict[str, np.ndarray]:
    stats = batch.summary_statistics()
    return {
        "log_likelihood": np.asarray(stats["log_likelihood"], dtype=float),
        "min_event_ess": np.asarray(stats["min_event_ess"], dtype=float),
        "min_event_ess_fraction": np.min(np.asarray(batch.event_ess_fraction, float), axis=1),
        "max_event_weight_fraction": np.asarray(stats["max_event_max_weight"], dtype=float),
        "selection_ess": np.asarray(stats["selection_ess"], dtype=float),
        "selection_ess_fraction": np.asarray(stats["selection_ess_fraction"], dtype=float),
        "selection_max_weight_fraction": np.asarray(stats["selection_max_weight"], dtype=float),
        "event_variance_total": np.asarray(stats["event_variance_total"], dtype=float),
        "selection_variance_term": np.asarray(stats["selection_variance_term"], dtype=float),
        "shape_log_likelihood_variance": np.asarray(
            stats["shape_log_likelihood_variance"], dtype=float
        ),
    }


def _point_statistics(stats: Mapping[str, np.ndarray], row: int, batch) -> dict[str, object]:
    point: dict[str, object] = {key: _finite_or_none(value[row]) for key, value in stats.items()}
    point["worst_event"] = batch.event_names[int(np.argmin(np.asarray(batch.event_ess)[row]))]
    return point


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    payload: dict[str, object] = {
        f"q{q:g}": _finite_or_none(np.quantile(values, q, method=QUANTILE_METHOD))
        for q in IMPORTANCE_QUANTILES
    }
    payload["min"] = _finite_or_none(np.min(values))
    payload["max"] = _finite_or_none(np.max(values))
    payload["mean"] = _finite_or_none(np.mean(values)) if nonfinite == 0 else None
    payload["n_nonfinite"] = nonfinite
    return payload


def pooled_importance_diagnostics(
    results: Sequence[DynestyResult],
    posterior,
    selection,
    population_model,
    *,
    hbi_config,
    n_draws: int = DEFAULT_IMPORTANCE_DRAWS,
    seed: int,
    truth: Mapping[str, float] | None = None,
    batch_size: int = DEFAULT_IMPORTANCE_BATCH_SIZE,
) -> dict[str, object]:
    """Importance diagnostics at the pooled posterior median and over pooled draws.

    Uses :func:`build_importance_diagnostics` with ``hbi_config``; the
    diagnosed likelihood must equal the one every run sampled (names, HBI
    configuration other than the selection chunk size, model and data digests;
    ``ValueError`` otherwise). The point is the coordinate-wise median of the
    pooled equal-weight draws; the ``n_draws`` draws are a seeded subset
    (without replacement) of them. ``truth``, when given, is evaluated too and
    reported only.
    """
    results = list(results)
    names = _common_names(results)
    n_draws = int(n_draws)
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    fn = build_importance_diagnostics(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=int(batch_size),
    )
    requested = _without_chunk_size(fn.likelihood_identity())
    for index, result in enumerate(results):
        stored = result.likelihood_identity
        if stored is None:
            raise ValueError(f"run {index} carries no likelihood identity")
        diffs = _manifest_differences(_without_chunk_size(stored), requested)
        if diffs:
            raise ValueError(
                f"run {index} sampled a different likelihood than the diagnostics evaluate; "
                f"differing keys: {diffs}"
            )
    draws = pooled_posterior_draws(results)
    rng = np.random.default_rng(int(seed))
    k = min(n_draws, draws.shape[0])
    chosen = draws[np.sort(rng.choice(draws.shape[0], size=k, replace=False))]
    median = np.median(draws, axis=0)
    rows = [median[None, :], chosen]
    if truth is not None:
        missing = sorted(set(names) - set(truth))
        if missing:
            raise ValueError(f"truth is missing hyperparameter(s) {missing}")
        rows.append(np.asarray([[float(truth[name]) for name in names]]))
    batch = fn(np.vstack(rows))
    stats = _importance_statistics(batch)
    payload: dict[str, object] = {
        "hbi_config": _hbi_config_payload(hbi_config),
        "n_draws": int(k),
        "seed": int(seed),
        "quantile_method": QUANTILE_METHOD,
        "posterior_median_hyperparameters": {
            name: float(median[i]) for i, name in enumerate(names)
        },
        "posterior_median": _point_statistics(stats, 0, batch),
        "over_posterior": {key: _distribution(value[1 : 1 + k]) for key, value in stats.items()},
    }
    if truth is not None:
        payload["truth"] = _point_statistics(stats, 1 + k, batch)
    return payload


def summarize_ns_fit(
    results: Sequence[DynestyResult],
    posterior,
    selection,
    population_model,
    *,
    hbi_config,
    importance_draws: int = DEFAULT_IMPORTANCE_DRAWS,
    importance_seed: int,
    rhat_draws_per_run: int = DEFAULT_RHAT_DRAWS_PER_RUN,
    truth: Mapping[str, float] | None = None,
    run_dirs: Sequence[str] | None = None,
) -> dict[str, object]:
    """Sampler, evidence and importance summary of ``R`` runs of one model."""
    results = list(results)
    names = _common_names(results)
    if run_dirs is not None and len(run_dirs) != len(results):
        raise ValueError("run_dirs must have one entry per result")
    runs = [
        _run_record(result, repeat=r, run_dir=None if run_dirs is None else str(run_dirs[r]))
        for r, result in enumerate(results)
    ]
    unsupported = [run["n_selection_unsupported"] for run in runs]
    converged = [run["converged"] for run in runs]
    return {
        "names": list(names),
        "n_repeats": len(results),
        "data_identity": data_identity_digest(results),
        "runs": runs,
        "totals": {
            "min_kish_ess": float(min(run["kish_ess"] for run in runs)),
            # Structural: the backend raises PopulationDensityError on any NaN/+inf
            # likelihood value, so a completed run has none.
            "n_nan_or_posinf_evaluations": 0,
            "n_selection_unsupported": (
                None if any(v is None for v in unsupported) else int(sum(unsupported))
            ),
            "all_converged": bool(all(value is True for value in converged)),
            "total_elapsed_seconds": float(sum(run["elapsed_seconds"] for run in runs)),
            "total_likelihood_evaluations": (
                None
                if any(run["n_likelihood_evaluations"] is None for run in runs)
                else int(sum(run["n_likelihood_evaluations"] for run in runs))
            ),
        },
        "evidence": evidence_repeat_statistics(results),
        "convergence": (
            cross_run_rhat(results, max_draws_per_run=rhat_draws_per_run)
            if len(results) >= 2
            else {"max_r_hat": None, "per_parameter": {}, "draws_per_run": None}
        ),
        "importance_diagnostics": pooled_importance_diagnostics(
            results,
            posterior,
            selection,
            population_model,
            hbi_config=hbi_config,
            n_draws=importance_draws,
            seed=importance_seed,
            truth=truth,
        ),
    }


# ---------------------------------------------------------------------------
# Acceptance criteria v2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NSRecoveryAcceptanceCriteria:
    """Numerical acceptance thresholds for nested-sampling fits (Phase-3 v2).

    Defaults are the Phase-3 v2 values: cross-run R-hat <= 1.01, Kish ESS >=
    1000 per run, evidence repeat std <= 0.2, every reported log-evidence error
    <= 0.2, pairwise repeat z <= 3.0, and the unchanged v1 importance thresholds
    (min event ESS 20, selection ESS 200, max event weight 0.25, max selection
    weight 0.10, Var(log L) 1.0) enforced at the posterior-median point, at the
    median over the pooled draws and at the tail (``tail_ess_quantile`` of the
    ESS-type metrics, ``tail_weight_quantile`` of the weights and the
    variance). Every run must also terminate by ``dlogz`` (not by a budget) and
    have no selection-unsupported evaluations (NaN/+inf evaluations raise in
    the backend and never produce a result). ``min_repeats`` applies per
    catalog and ``min_runs`` (catalogs) per campaign. :meth:`f3_level` gives the
    F3 gates of the dynesty ladder (orchestrator decision D3).
    """

    FORMAT_VERSION: ClassVar[str] = "gwpop-search-ns-acceptance-criteria-2.0"

    max_r_hat: float = 1.01
    min_kish_ess_per_run: float = 1000.0
    max_evidence_repeat_std: float = 0.2
    max_log_evidence_error: float = 0.2
    max_pairwise_repeat_z: float = 3.0
    min_event_ess: float = 20.0
    min_selection_ess: float = 200.0
    max_event_weight_fraction: float = 0.25
    max_selection_weight_fraction: float = 0.10
    max_shape_log_likelihood_variance: float = 1.0
    tail_ess_quantile: float = 0.1
    tail_weight_quantile: float = 0.9
    max_selection_unsupported_evaluations: int = 0
    require_dlogz_termination: bool = True
    min_repeats: int = 4
    min_runs: int = 4

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        for item in dataclass_fields(self):
            value = getattr(self, item.name)
            if item.name == "require_dlogz_termination":
                if not isinstance(value, bool):
                    raise TypeError("require_dlogz_termination must be a bool")
            elif item.name in ("max_selection_unsupported_evaluations", "min_repeats", "min_runs"):
                if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                    raise TypeError(f"{item.name} must be an integer; got {value!r}")
                set_(self, item.name, int(value))
            else:
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise TypeError(f"{item.name} must be a real number; got {value!r}")
                if not math.isfinite(float(value)):
                    raise ValueError(f"{item.name} must be finite")
                set_(self, item.name, float(value))
        if self.max_r_hat <= 1.0:
            raise ValueError("max_r_hat must be greater than 1")
        for name in (
            "min_kish_ess_per_run",
            "max_evidence_repeat_std",
            "max_log_evidence_error",
            "max_pairwise_repeat_z",
            "min_event_ess",
            "min_selection_ess",
            "max_shape_log_likelihood_variance",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_event_weight_fraction", "max_selection_weight_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.tail_ess_quantile not in IMPORTANCE_QUANTILES:
            raise ValueError(f"tail_ess_quantile must be one of {IMPORTANCE_QUANTILES}")
        if self.tail_weight_quantile not in IMPORTANCE_QUANTILES:
            raise ValueError(f"tail_weight_quantile must be one of {IMPORTANCE_QUANTILES}")
        if not self.tail_ess_quantile <= 0.5 <= self.tail_weight_quantile:
            raise ValueError(
                "the tail must be at least as strict as the median: "
                "tail_ess_quantile <= 0.5 <= tail_weight_quantile"
            )
        if self.max_selection_unsupported_evaluations < 0:
            raise ValueError("max_selection_unsupported_evaluations cannot be negative")
        if self.min_repeats < 2:
            raise ValueError("min_repeats must be >= 2 (cross-run R-hat and repeat scatter)")
        if self.min_runs <= 0:
            raise ValueError("min_runs must be positive")

    @classmethod
    def f3_level(cls, **overrides) -> "NSRecoveryAcceptanceCriteria":
        """F3 gates (D3): R-hat 1.05, Kish 500, evidence 0.5/0.5, z 3.5, 10/100/0.35/0.15/2.0."""
        payload = {
            "max_r_hat": 1.05,
            "min_kish_ess_per_run": 500.0,
            "max_evidence_repeat_std": 0.5,
            "max_log_evidence_error": 0.5,
            "max_pairwise_repeat_z": 3.5,
            "min_event_ess": 10.0,
            "min_selection_ess": 100.0,
            "max_event_weight_fraction": 0.35,
            "max_selection_weight_fraction": 0.15,
            "max_shape_log_likelihood_variance": 2.0,
            "min_repeats": 2,
            "min_runs": 1,
        }
        payload.update(overrides)
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"format_version": self.FORMAT_VERSION}
        for item in dataclass_fields(self):
            payload[item.name] = getattr(self, item.name)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "NSRecoveryAcceptanceCriteria":
        values = dict(payload)
        version = values.pop("format_version", None)
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"unsupported acceptance-criteria format {version!r}")
        known = {item.name for item in dataclass_fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown acceptance-criteria field(s) {unknown}")
        return cls(**values)


def _check(name: str, value, limit, *, comparison: str, **extra) -> dict[str, object]:
    if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
        passed = False
    elif comparison == "le":
        passed = float(value) <= float(limit)
    elif comparison == "ge":
        passed = float(value) >= float(limit)
    else:  # pragma: no cover - internal misuse
        raise ValueError(comparison)
    record = {
        "name": name,
        "value": None if value is None or isinstance(value, bool) else _finite_or_none(value),
        "comparison": comparison,
        "limit": float(limit),
        "passed": bool(passed),
    }
    record.update(extra)
    return record


def _flag_check(name: str, value) -> dict[str, object]:
    return {
        "name": name,
        "value": None if value is None else bool(value),
        "comparison": "is_true",
        "limit": True,
        "passed": value is True,
    }


def assess_ns_fit(
    fit: Mapping[str, object],
    criteria: NSRecoveryAcceptanceCriteria,
) -> dict[str, object]:
    """Apply ``criteria`` to one :func:`summarize_ns_fit` summary."""
    if not isinstance(criteria, NSRecoveryAcceptanceCriteria):
        raise TypeError("criteria must be NSRecoveryAcceptanceCriteria")
    evidence = dict(fit["evidence"])
    convergence = dict(fit["convergence"])
    totals = dict(fit["totals"])
    importance = dict(fit["importance_diagnostics"])
    point = dict(importance["posterior_median"])
    over = {key: dict(value) for key, value in dict(importance["over_posterior"]).items()}

    checks = [
        _check("n_repeats", fit["n_repeats"], criteria.min_repeats, comparison="ge"),
        _check("max_cross_run_r_hat", convergence.get("max_r_hat"), criteria.max_r_hat,
               comparison="le"),
        _check("min_kish_ess_per_run", totals.get("min_kish_ess"),
               criteria.min_kish_ess_per_run, comparison="ge"),
        _check("evidence_repeat_std", evidence.get("repeat_std"),
               criteria.max_evidence_repeat_std, comparison="le"),
        _check("max_log_evidence_error", evidence.get("max_reported_error"),
               criteria.max_log_evidence_error, comparison="le"),
        _check("max_pairwise_repeat_z", evidence.get("max_pairwise_z"),
               criteria.max_pairwise_repeat_z, comparison="le"),
        _check("n_nan_or_posinf_evaluations", totals.get("n_nan_or_posinf_evaluations"), 0,
               comparison="le"),
        _check("n_selection_unsupported", totals.get("n_selection_unsupported"),
               criteria.max_selection_unsupported_evaluations, comparison="le"),
    ]
    if criteria.require_dlogz_termination:
        checks.append(_flag_check("all_runs_terminated_by_dlogz", totals.get("all_converged")))
    for key, comparison, field_name in _GATED_IMPORTANCE:
        limit = getattr(criteria, field_name)
        distribution = over.get(key, {})
        tail_q = criteria.tail_ess_quantile if comparison == "ge" else criteria.tail_weight_quantile
        checks.append(_check(f"point.{key}", point.get(key), limit, comparison=comparison))
        checks.append(
            _check(f"draw_median.{key}", distribution.get("q0.5"), limit,
                   comparison=comparison, quantile=0.5)
        )
        checks.append(
            _check(f"tail.{key}", distribution.get(f"q{tail_q:g}"), limit,
                   comparison=comparison, quantile=float(tail_q))
        )
    return {"passed": bool(all(item["passed"] for item in checks)), "checks": checks}


# ---------------------------------------------------------------------------
# Per-catalog recovery
# ---------------------------------------------------------------------------


def recovery_posterior_summary(draws: np.ndarray, names, truth: Mapping[str, float]):
    """Pooled posterior quantiles, moments, truth coverage and standardized offset."""
    from .recovery import posterior_summary

    draws = np.asarray(draws, dtype=float)
    samples = {name: draws[:, k] for k, name in enumerate(names)}
    summary = posterior_summary(samples, truth)
    for name, entry in summary.items():
        std = float(entry["std"])
        if "truth" in entry and std > 0.0:
            entry["standardized_offset"] = (float(entry["median"]) - float(entry["truth"])) / std
    return summary


def truth_rank_diagnostics(dataset) -> dict[str, object]:
    """Quantile of each event's true parameter within its PE (:func:`pe_truth_quantiles`).

    For DAG-consistent PE (``noisy_observation``) the quantile is uniform
    across events for a coordinate that detection does not depend on and whose
    PE-prior edges are far from the posterior (chi_eff); zero-noise PE centred
    on the truth puts it near 0.5. The KS p-value against U(0, 1) is advisory.
    Detection depends on the other coordinates, so uniformity is not expected
    there (reported for inspection only).
    """
    from scipy.stats import kstest

    from .synthetic import pe_truth_quantiles

    payload: dict[str, object] = {
        "observation_model": dataset.config.observation_model,
        "uniform_expected": ["chi_eff"],
    }
    for name in ("chi_eff", "q", "m1_detector", "luminosity_distance"):
        if name not in dataset.posterior.samples or name not in dataset.event_truths:
            continue
        quantiles = pe_truth_quantiles(dataset.posterior, dataset.event_truths, name)
        test = kstest(quantiles, "uniform")
        payload[name] = {
            "mean": float(np.mean(quantiles)),
            "std": float(np.std(quantiles)),
            "ks_statistic": float(test.statistic),
            "ks_pvalue": float(test.pvalue),
            "quantiles": [float(v) for v in quantiles],
        }
    return payload


def _generate_catalog(data_seed: int, survey_config):
    from gwpop_search.models import GwcatChiEffBBHModel

    from .synthetic import generate_baseline_synthetic_dataset

    model = GwcatChiEffBBHModel()
    dataset = generate_baseline_synthetic_dataset(
        seed=int(data_seed), model=model, config=survey_config
    )
    return dataset, model


def _save_pooled_posterior(path: Path, draws: np.ndarray, names, n_repeats: int) -> None:
    per_run = draws.shape[0] // n_repeats
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            names=np.asarray(list(names)),
            samples=np.asarray(draws, dtype=np.float64),
            run_index=np.repeat(np.arange(n_repeats, dtype=np.int64), per_run),
        )
    tmp.replace(path)


def run_ns_synthetic_recovery(
    run_dir: str | Path,
    *,
    data_seed: int,
    sampler_seed: int,
    repeats: int = DEFAULT_NS_REPEATS,
    survey_config=None,
    dynesty_config: DynestyConfig | None = None,
    hbi_config=None,
    importance_draws: int = DEFAULT_IMPORTANCE_DRAWS,
    rhat_draws_per_run: int = DEFAULT_RHAT_DRAWS_PER_RUN,
) -> dict[str, object]:
    """Generate one Phase-3 catalog and run/resume/reuse its ``R`` dynesty repeats.

    Repeat ``r`` lives in ``run_dir/repeats/repeat_{r:03d}`` and uses the seed
    ``repeat_seeds(sampler_seed, repeats)[r]``. The catalog is regenerated
    deterministically from ``data_seed`` (every repeat manifest pins the data
    digests, so a different catalog is refused). Writes and returns
    ``recovery_summary.json`` (format ``gwpop-search-phase3-recovery-2.0``) and
    the pooled equal-weight draws ``posterior_pooled.npz``.
    """
    from .synthetic import require_population_proxy_coverage

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    survey_config = default_phase3_survey_config() if survey_config is None else survey_config
    dynesty_config = default_phase3_dynesty_config() if dynesty_config is None else dynesty_config
    hbi_config = phase3_hbi_config() if hbi_config is None else hbi_config
    seeds = repeat_seeds(sampler_seed, repeats)
    priors = dict(BASELINE_SYNTHETIC_PRIORS)

    dataset, model = _generate_catalog(data_seed, survey_config)
    require_population_proxy_coverage(
        dataset.selection,
        priors=priors,
        context="Phase-3 v2 dynesty recovery hyperprior",
    )

    results: list[DynestyResult] = []
    run_dirs: list[str] = []
    for r, seed in enumerate(seeds):
        relative = f"{REPEATS_DIRNAME}/{REPEAT_DIR_PREFIX}{r:03d}"
        results.append(
            run_dynesty_population(
                dataset.posterior,
                dataset.selection,
                model,
                priors,
                seed=int(seed),
                config=dynesty_config,
                hbi_config=hbi_config,
                run_dir=run_dir / relative,
            )
        )
        run_dirs.append(relative)

    names = _common_names(results)
    truth = {name: float(value) for name, value in dataset.truth_hyperparameters.items()}
    draws = pooled_posterior_draws(results)
    fit = summarize_ns_fit(
        results,
        dataset.posterior,
        dataset.selection,
        model,
        hbi_config=hbi_config,
        importance_draws=importance_draws,
        importance_seed=_derived_seed(int(sampler_seed), "importance-draws", 0),
        rhat_draws_per_run=rhat_draws_per_run,
        truth=truth,
        run_dirs=run_dirs,
    )
    summary = {
        "format_version": RECOVERY_SUMMARY_FORMAT_VERSION,
        "data_seed": int(data_seed),
        "sampler_seed": int(sampler_seed),
        "repeat_seeds": [int(seed) for seed in seeds],
        "survey_config": survey_config.to_dict(),
        "dynesty_config": plan_dynesty_config(dynesty_config),
        "hbi_config": _hbi_config_payload(hbi_config),
        "model": model.to_config(),
        "priors": serialize_prior_map(priors),
        "truth_hyperparameters": truth,
        "posterior": recovery_posterior_summary(draws, names, truth),
        "pooled_posterior": {
            "method": "equal-weight mixture of separately normalized runs",
            "n_draws": int(draws.shape[0]),
            "draws_per_run": int(draws.shape[0] // len(results)),
            "file": POOLED_POSTERIOR_NAME,
        },
        "truth_rank_diagnostics": truth_rank_diagnostics(dataset),
        "fit": fit,
        "diagnostics": {
            "n_events": int(dataset.posterior.n_events),
            "n_pe_samples": int(dataset.posterior.n_samples_total),
            "n_selected_injections": int(dataset.selection.n_selected),
            "n_draw_injections": int(dataset.selection.campaigns[0].n_draw),
        },
    }
    _save_pooled_posterior(run_dir / POOLED_POSTERIOR_NAME, draws, names, len(results))
    _atomic_write_json(run_dir / RECOVERY_SUMMARY_NAME, summary)
    return summary


def assess_ns_recovery_summary(
    summary: Mapping[str, object],
    criteria: NSRecoveryAcceptanceCriteria | None = None,
) -> dict[str, object]:
    """Apply ``criteria`` to one ``gwpop-search-phase3-recovery-2.0`` summary."""
    criteria = NSRecoveryAcceptanceCriteria() if criteria is None else criteria
    if summary.get("format_version") != RECOVERY_SUMMARY_FORMAT_VERSION:
        raise ValueError(
            f"expected a {RECOVERY_SUMMARY_FORMAT_VERSION} summary; got "
            f"{summary.get('format_version')!r}"
        )
    assessment = assess_ns_fit(dict(summary["fit"]), criteria)
    return {
        "data_seed": int(summary["data_seed"]),
        "sampler_seed": int(summary["sampler_seed"]),
        "passed": assessment["passed"],
        "checks": assessment["checks"],
    }


def aggregate_ns_recovery_summaries(
    summaries: Iterable[Mapping[str, object]],
    criteria: NSRecoveryAcceptanceCriteria | None = None,
    *,
    plan_sha256: str | None = None,
) -> dict[str, object]:
    """Campaign summary 2.0: per-catalog assessments, coverage and offsets."""
    criteria = NSRecoveryAcceptanceCriteria() if criteria is None else criteria
    summaries = [dict(item) for item in summaries]
    assessments = [assess_ns_recovery_summary(item, criteria) for item in summaries]

    parameter_names: set[str] = set()
    for item in summaries:
        parameter_names.update(dict(item["posterior"]).keys())
    coverage: dict[str, dict[str, object]] = {}
    for name in sorted(parameter_names):
        indicators: list[bool] = []
        offsets: list[float] = []
        for item in summaries:
            stats = dict(dict(item["posterior"]).get(name, {}))
            if "truth_in_90pct_interval" in stats:
                indicators.append(bool(stats["truth_in_90pct_interval"]))
            if stats.get("standardized_offset") is not None:
                offsets.append(float(stats["standardized_offset"]))
        coverage[name] = {
            "n_runs": len(indicators),
            "central_90pct_coverage_fraction": (
                float(np.mean(indicators)) if indicators else None
            ),
            "median_standardized_offset": float(np.median(offsets)) if offsets else None,
            "standardized_offsets": offsets,
        }
    evidence = []
    for item in summaries:
        repeats = dict(dict(item["fit"]).get("evidence", {}))
        evidence.append(
            {
                "data_seed": int(item["data_seed"]),
                "log_evidence_mean": repeats.get("mean"),
                "repeat_std": repeats.get("repeat_std"),
                "max_reported_error": repeats.get("max_reported_error"),
                "max_pairwise_z": repeats.get("max_pairwise_z"),
            }
        )
    n_pass = sum(bool(item["passed"]) for item in assessments)
    enough_runs = len(summaries) >= criteria.min_runs
    all_pass = bool(summaries) and n_pass == len(summaries)
    return {
        "format_version": CAMPAIGN_SUMMARY_FORMAT_VERSION,
        "criteria": criteria.to_dict(),
        "plan_sha256": plan_sha256,
        "n_runs": len(summaries),
        "n_numerical_pass": n_pass,
        "n_numerical_fail": len(summaries) - n_pass,
        "enough_runs": bool(enough_runs),
        "all_numerical_pass": bool(all_pass),
        "phase3_numerical_gate_passed": bool(enough_runs and all_pass),
        # Reported, not thresholded: a handful of catalogs is not a calibrated
        # coverage measurement.
        "coverage": coverage,
        "evidence": evidence,
        "run_assessments": assessments,
    }


def load_ns_recovery_summaries(root: str | Path) -> list[dict[str, object]]:
    """All ``run_*/recovery_summary.json`` below ``root`` (format 2.0 only)."""
    summaries = []
    for path in sorted(Path(root).glob(f"{RUN_DIR_PREFIX}*/{RECOVERY_SUMMARY_NAME}")):
        payload = json.loads(path.read_text())
        if payload.get("format_version") != RECOVERY_SUMMARY_FORMAT_VERSION:
            raise ValueError(
                f"{path} has format {payload.get('format_version')!r}; this is not a "
                f"{RECOVERY_SUMMARY_FORMAT_VERSION} campaign root"
            )
        summaries.append(payload)
    return summaries


def _load_plan(root: Path) -> dict[str, object] | None:
    path = root / CAMPAIGN_PLAN_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def assess_ns_recovery_campaign(
    root: str | Path,
    criteria: NSRecoveryAcceptanceCriteria | None = None,
) -> dict[str, object]:
    """Assess completed catalogs and write ``campaign_summary.json`` (format 2.0).

    Without explicit ``criteria`` the criteria recorded in the campaign plan
    are used (the defaults when the root has no plan). Summaries of catalogs
    that are not part of the plan are refused.
    """
    root = Path(root)
    plan = _load_plan(root)
    if plan is not None and plan.get("format_version") != PLAN_FORMAT_VERSION:
        raise ValueError(
            f"{root / CAMPAIGN_PLAN_NAME} has format {plan.get('format_version')!r}; "
            f"expected {PLAN_FORMAT_VERSION}"
        )
    if criteria is None:
        criteria = (
            NSRecoveryAcceptanceCriteria.from_dict(plan["criteria"])
            if plan is not None
            else NSRecoveryAcceptanceCriteria()
        )
    summaries = load_ns_recovery_summaries(root)
    if plan is not None:
        planned = {
            (int(pair["data_seed"]), int(pair["sampler_seed"])) for pair in plan["seed_pairs"]
        }
        for item in summaries:
            key = (int(item["data_seed"]), int(item["sampler_seed"]))
            if key not in planned:
                raise ValueError(f"catalog {key} is not part of the campaign plan")
    result = aggregate_ns_recovery_summaries(
        summaries,
        criteria=criteria,
        plan_sha256=None if plan is None else _json_sha256(plan),
    )
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / CAMPAIGN_SUMMARY_NAME, result)
    return result


# ---------------------------------------------------------------------------
# Campaign plan and driver
# ---------------------------------------------------------------------------


def build_ns_campaign_plan(
    *,
    n_runs: int,
    root_seed: int,
    repeats: int,
    survey_config,
    dynesty_config: DynestyConfig,
    hbi_config,
    criteria: NSRecoveryAcceptanceCriteria,
    importance_draws: int = DEFAULT_IMPORTANCE_DRAWS,
    rhat_draws_per_run: int = DEFAULT_RHAT_DRAWS_PER_RUN,
) -> dict[str, object]:
    """The ``gwpop-search-phase3-campaign-plan-2.0`` payload (JSON-normalized)."""
    from gwpop_search.models import DEFAULT_BASELINE_HYPERPARAMETERS, GwcatChiEffBBHModel

    n_runs = int(n_runs)
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    repeats = int(repeats)
    if repeats < 2:
        raise ValueError("repeats must be >= 2 (cross-run R-hat and repeat scatter)")
    if int(importance_draws) < 1:
        raise ValueError("importance_draws must be positive")
    if int(rhat_draws_per_run) < 4:
        raise ValueError("rhat_draws_per_run must be >= 4")
    if not isinstance(criteria, NSRecoveryAcceptanceCriteria):
        raise TypeError("criteria must be NSRecoveryAcceptanceCriteria")
    pairs = recovery_seed_pairs(n_runs, root_seed=int(root_seed))
    plan = {
        "format_version": PLAN_FORMAT_VERSION,
        "root_seed": int(root_seed),
        "n_runs": n_runs,
        "repeats": repeats,
        "seed_pairs": [
            {
                "data_seed": int(data_seed),
                "sampler_seed": int(sampler_seed),
                "repeat_seeds": [int(s) for s in repeat_seeds(sampler_seed, repeats)],
            }
            for data_seed, sampler_seed in pairs
        ],
        "survey_config": survey_config.to_dict(),
        "dynesty_config": plan_dynesty_config(dynesty_config),
        "hbi_config": _hbi_config_payload(hbi_config),
        "model": GwcatChiEffBBHModel().to_config(),
        "priors": serialize_prior_map(BASELINE_SYNTHETIC_PRIORS),
        "truth_hyperparameters": {
            name: float(value) for name, value in DEFAULT_BASELINE_HYPERPARAMETERS.items()
        },
        "diagnostics": {
            "importance_draws": int(importance_draws),
            "rhat_draws_per_run": int(rhat_draws_per_run),
            "importance_quantiles": list(IMPORTANCE_QUANTILES),
            "quantile_method": QUANTILE_METHOD,
        },
        "criteria": criteria.to_dict(),
    }
    return json.loads(json.dumps(_json_ready(plan), sort_keys=True))


def write_or_verify_plan(path: str | Path, plan: Mapping[str, object]) -> None:
    """Write ``plan`` or refuse (``ValueError``) when an existing plan differs."""
    path = Path(path)
    plan = json.loads(json.dumps(_json_ready(dict(plan)), sort_keys=True))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != plan:
            diffs = _manifest_differences(existing, plan)
            raise ValueError(
                f"{path} does not match the requested campaign; refusing to resume a "
                f"different plan (differing keys: {diffs})"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, plan)


def run_ns_recovery_campaign(
    root: str | Path,
    *,
    n_runs: int = 4,
    root_seed: int = PHASE3_ROOT_SEED,
    repeats: int = DEFAULT_NS_REPEATS,
    survey_config=None,
    dynesty_config: DynestyConfig | None = None,
    hbi_config=None,
    criteria: NSRecoveryAcceptanceCriteria | None = None,
    importance_draws: int = DEFAULT_IMPORTANCE_DRAWS,
    rhat_draws_per_run: int = DEFAULT_RHAT_DRAWS_PER_RUN,
) -> dict[str, object]:
    """Run/resume the deterministic Phase-3 v2 campaign and assess it.

    Writes or verifies ``campaign_plan.json`` first (a different plan is
    refused before any computation), then runs every catalog in
    ``run_{i:03d}`` (completed repeats are reused untouched, interrupted ones
    resume from their dynesty checkpoint) and writes ``campaign_summary.json``.
    """
    root = Path(root)
    survey_config = default_phase3_survey_config() if survey_config is None else survey_config
    dynesty_config = default_phase3_dynesty_config() if dynesty_config is None else dynesty_config
    hbi_config = phase3_hbi_config() if hbi_config is None else hbi_config
    criteria = NSRecoveryAcceptanceCriteria() if criteria is None else criteria
    plan = build_ns_campaign_plan(
        n_runs=n_runs,
        root_seed=root_seed,
        repeats=repeats,
        survey_config=survey_config,
        dynesty_config=dynesty_config,
        hbi_config=hbi_config,
        criteria=criteria,
        importance_draws=importance_draws,
        rhat_draws_per_run=rhat_draws_per_run,
    )
    write_or_verify_plan(root / CAMPAIGN_PLAN_NAME, plan)
    for index, pair in enumerate(plan["seed_pairs"]):
        run_ns_synthetic_recovery(
            root / f"{RUN_DIR_PREFIX}{index:03d}",
            data_seed=int(pair["data_seed"]),
            sampler_seed=int(pair["sampler_seed"]),
            repeats=repeats,
            survey_config=survey_config,
            dynesty_config=dynesty_config,
            hbi_config=hbi_config,
            importance_draws=importance_draws,
            rhat_draws_per_run=rhat_draws_per_run,
        )
    return assess_ns_recovery_campaign(root, criteria=criteria)


# ---------------------------------------------------------------------------
# Fingerprints (resume integrity)
# ---------------------------------------------------------------------------


def ns_run_fingerprints(path: str | Path) -> dict[str, str]:
    """SHA-256 of every completed dynesty run artifact at or below ``path``.

    A dynesty run directory is one holding ``manifest.json``; its
    ``manifest.json``, ``result.npz`` and ``result.npz.json`` are hashed when
    present (``checkpoint.pkl`` is excluded: it changes while a run is in
    progress). Plan files at ``path`` itself (``campaign_plan.json``,
    ``evidence_check_plan.json``) are hashed too. Keys are POSIX paths relative
    to ``path``.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")
    fingerprints: dict[str, str] = {}
    for manifest in sorted(path.rglob("manifest.json")):
        run_dir = manifest.parent
        for name in _FINGERPRINTED_RUN_FILES:
            item = run_dir / name
            if item.is_file():
                fingerprints[item.relative_to(path).as_posix()] = file_sha256(item)
    for name in _FINGERPRINTED_PLAN_FILES:
        item = path / name
        if item.is_file():
            fingerprints[name] = file_sha256(item)
    return dict(sorted(fingerprints.items()))


def ns_fingerprint_report(path: str | Path) -> dict[str, object]:
    """The JSON document written by ``gwpop-search fingerprint-ns-run``."""
    return {
        "format_version": NS_FINGERPRINT_FORMAT_VERSION,
        "run_dir": str(Path(path)),
        "fingerprints": ns_run_fingerprints(path),
    }


def compare_ns_fingerprints(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, object]:
    """Every file fingerprinted ``before`` must keep its hash ``after``; new files may appear."""
    changed = sorted(key for key in before if key in after and before[key] != after[key])
    missing = sorted(key for key in before if key not in after)
    added = sorted(key for key in after if key not in before)
    unchanged = sorted(key for key in before if key in after and before[key] == after[key])
    return {
        "unchanged": unchanged,
        "changed": changed,
        "missing": missing,
        "added": added,
        "passed": not changed and not missing,
    }
