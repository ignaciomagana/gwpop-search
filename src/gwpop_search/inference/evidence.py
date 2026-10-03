"""Bayesian evidences from repeated, independent dynesty nested-sampling runs.

Every evidence and hyperposterior of the production ladder comes from the
static dynesty sampler of :mod:`gwpop_search.inference.dynesty_backend`
(bound ``multi``, sample ``rslice``); the JAXNS/NumPyro path is gone.

Run contract
------------
:func:`run_hbi_evidence` performs one resumable run of the standardized,
rate-marginalized HBI shape likelihood ``log L = sum_i log ell_i - N log A``
under the model's declared hyperpriors. ``-inf`` (zero population support)
stays ``-inf``: dynesty's plateau/initial-volume bookkeeping integrates it,
nothing is floored. NaN/``+inf`` raise
:class:`gwpop_search.hbi.PopulationDensityError`. A run directory holds
``manifest.json``, ``checkpoint.pkl`` and ``result.npz`` (+ ``.json``); an
existing complete result is reused only for an identical manifest (data
digests, priors, dynesty trajectory configuration, seed, code, software and
runtime identity), a checkpoint is resumed, anything else is refused.

The sampling *parameterization* is part of the run identity. The identity
parameterization is exactly the backend's
:func:`~gwpop_search.inference.dynesty_backend.run_dynesty_population`. The
``ordered_exchangeable_pairs`` parameterization
(:mod:`gwpop_search.inference.label_switching`) samples the same prior and
likelihood through a bijective order-statistic transform; the evidence is
identical and the posterior is on canonical labels. Its manifest is the
backend manifest plus a ``parameterization`` entry.

dynesty's "could not find a single point with a valid log-likelihood" after
1000 initialization attempts -- and its stall with fewer finite points than it
needs (``InsufficientFiniteSupportError``; dynesty itself would loop without
limit) -- is re-raised as :class:`NoFiniteSupportError` (a typed numerical
failure: the model has, numerically, too little support on the data);
everything else propagates.

Evidence error: each run's ``log_evidence_error`` is dynesty's ``logzerr``
plus, in quadrature, the relative error of its initial kept-volume estimate
(``sqrt(1/k - 1/(nlive N))``, zero without ``-inf`` regions; see
``dynesty_backend.initial_volume_uncertainty``). Under the v2 sharp variance
cut ``ln Z`` is the evidence over the full prior (the LVK
``log_bayes_factor_scaled`` convention, not bilby's retained-prior
``log_bayes_factor``).

Repeat summary
--------------
``conservative_error = max(repeat std (ddof=1), max reported logzerr)``:
dynesty's ``logzerr`` (~ sqrt(H / nlive)) excludes implementation error of
the constrained-prior sampler, plateau-fraction noise and Monte-Carlo
likelihood error, so independent repeats are mandatory. Repeats are never
merged (``merge_runs``/``jitter_run``/``resample_run`` are biased on plateau
runs); their evidences are averaged and their posteriors pooled as an
equal-weight mixture of separately normalized runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import numpy as np

from .dynesty_backend import (
    DirtyCodeWarning,
    DynestyConfig,
    DynestyResult,
    InsufficientFiniteSupportError,
    SelectionSupportWarning,
    build_batched_log_likelihood,
    build_dynesty_manifest,
    dynesty_result_exists,
    load_dynesty_result,
    run_dynesty,
    run_dynesty_population,
    save_dynesty_result,
)
from .label_switching import IDENTITY, ModelParameterization
from .ns_diagnostics import max_pairwise_z, run_termination
from .priors import PriorSpec

SAMPLER_BACKEND = "dynesty"
SUPPORTED_DYNESTY_VERSIONS = ("3.1.0",)
_NO_FINITE_SUPPORT_MESSAGE = "could not find a single point"


class EvidenceBackendUnavailableError(ImportError):
    """The dynesty evidence backend is not installed."""


class NoFiniteSupportError(RuntimeError):
    """dynesty found no (or too few) hyperparameters with finite likelihood in 1000 x nlive prior draws."""


class LegacyEvidenceArtifactError(ValueError):
    """A JAXNS/NumPyro-era evidence artifact or configuration was presented to the dynesty path."""


def dynesty_version() -> str:
    """Installed dynesty version (package metadata; does not import dynesty)."""
    try:
        return str(importlib.metadata.version("dynesty"))
    except importlib.metadata.PackageNotFoundError as exc:
        raise EvidenceBackendUnavailableError(
            "the dynesty evidence backend requires the nested extra: "
            "pip install 'gwpop-search[nested]'"
        ) from exc


def sampler_backend_identity() -> dict[str, str]:
    """``{"name": "dynesty", "version": <installed>}`` for manifests and pins."""
    return {"name": SAMPLER_BACKEND, "version": dynesty_version()}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _json_sha256(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _differences(existing: Mapping, requested: Mapping, prefix: str = "") -> list[str]:
    diffs = []
    for key in sorted(set(existing) | set(requested)):
        a, b = existing.get(key, "<missing>"), requested.get(key, "<missing>")
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            diffs.extend(_differences(a, b, prefix=f"{prefix}{key}."))
        elif a != b:
            diffs.append(f"{prefix}{key}")
    return diffs


def _atomic_write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    os.replace(tmp, path)


def _warn_selection_unsupported(result: DynestyResult) -> None:
    if result.n_selection_unsupported:
        warnings.warn(
            SelectionSupportWarning(
                f"{result.n_selection_unsupported} likelihood evaluation(s) had population "
                "support on every event but none on the selection injections (log A = -inf); "
                "that prior volume is missing from the evidence"
            ),
            stacklevel=3,
        )


def _run_reparameterized_population(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: DynestyConfig,
    hbi_config,
    run_dir: Path,
    parameterization: ModelParameterization,
) -> DynestyResult:
    """``run_dynesty_population`` with a non-identity (evidence-preserving) transform.

    Same files, identity checks and reuse semantics as the backend's
    population runner; the manifest additionally pins the parameterization.
    """
    manifest = build_dynesty_manifest(
        posterior,
        selection,
        population_model,
        priors,
        seed=seed,
        config=config,
        hbi_config=hbi_config,
    )
    manifest["parameterization"] = json.loads(json.dumps(parameterization.to_dict()))
    manifest_sha256 = _json_sha256(manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError(
                f"{manifest_path} does not match the requested run; differing keys: "
                f"{_differences(existing, manifest)}"
            )
    else:
        _atomic_write_json(manifest_path, manifest)
    code = manifest["code"]
    if code.get("git_dirty"):
        warnings.warn(
            DirtyCodeWarning(
                "gwpop_search sources have uncommitted changes: git commit "
                f"{code['git_commit']} does not contain the code of this run, which is "
                f"identified by source_sha256 {code['source_sha256']}. Commit before "
                "production runs."
            ),
            stacklevel=3,
        )

    names, transform = parameterization.prior_transform(priors)
    result_path = run_dir / "result.npz"
    if dynesty_result_exists(result_path):
        result = load_dynesty_result(result_path)
        stored_run = result.provenance.get("run") or {}
        if (
            result.names != names
            or result.seed != int(seed)
            or result.config.identity_dict() != config.identity_dict()
            or stored_run.get("manifest_sha256") != manifest_sha256
        ):
            raise ValueError(f"{result_path} is inconsistent with {manifest_path}")
        if result.config.num_posterior_samples != config.num_posterior_samples:
            result = result.with_num_posterior_samples(config.num_posterior_samples)
        result = replace(result, config=config)
        _warn_selection_unsupported(result)
        return result

    checkpoint = run_dir / "checkpoint.pkl"
    loglike = build_batched_log_likelihood(
        posterior,
        selection,
        population_model,
        names,
        hbi_config=hbi_config,
        batch_size=config.batch_size,
    )
    result = run_dynesty(
        loglike,
        transform,
        len(names),
        seed=seed,
        config=config,
        checkpoint_file=checkpoint,
        resume=checkpoint.exists(),
        names=names,
        identity={"manifest_sha256": manifest_sha256},
    )
    save_dynesty_result(result_path, result)
    return result


def run_hbi_evidence(
    posterior,
    selection,
    population_model,
    priors: Mapping[str, PriorSpec],
    *,
    seed: int,
    config: DynestyConfig,
    hbi_config,
    run_dir: str | Path,
    parameterization: ModelParameterization = IDENTITY,
) -> DynestyResult:
    """One resumable dynesty run of the HBI shape likelihood (evidence + posterior).

    ``config`` must be fully resolved (e.g. ``slices`` for the model's
    dimension). Raises :class:`NoFiniteSupportError` when dynesty cannot
    initialize because no prior draw has finite likelihood.
    """
    if not isinstance(config, DynestyConfig):
        raise TypeError("config must be a DynestyConfig")
    if not isinstance(parameterization, ModelParameterization):
        raise TypeError("parameterization must be a ModelParameterization")
    run_dir = Path(run_dir)
    try:
        if parameterization.is_identity:
            return run_dynesty_population(
                posterior,
                selection,
                population_model,
                priors,
                seed=seed,
                config=config,
                hbi_config=hbi_config,
                run_dir=run_dir,
            )
        return _run_reparameterized_population(
            posterior,
            selection,
            population_model,
            priors,
            seed=seed,
            config=config,
            hbi_config=hbi_config,
            run_dir=run_dir,
            parameterization=parameterization,
        )
    except RuntimeError as exc:
        if isinstance(exc, InsufficientFiniteSupportError) or _NO_FINITE_SUPPORT_MESSAGE in str(exc):
            raise NoFiniteSupportError(
                f"dynesty found no hyperparameter with finite log-likelihood in its "
                f"initialization draws (run {run_dir}): {exc}"
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Repeats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceRepeatSummary:
    """Independent-repeat evidence summary.

    ``repeat_std`` uses ``ddof=1`` (0 for a single run); ``max_pairwise_z`` is
    ``max_{r<s} |lnZ_r - lnZ_s| / sqrt(err_r^2 + err_s^2)`` (``None`` for one
    run); ``information`` holds each run's KL divergence ``H`` (nats) when
    available.
    """

    log_evidence_mean: float
    repeat_std: float
    mean_reported_error: float
    max_reported_error: float
    n_repeats: int
    estimates: tuple[float, ...]
    errors: tuple[float, ...]
    information: tuple[float | None, ...] = ()
    max_pairwise_z: float | None = None

    @property
    def conservative_error(self) -> float:
        return float(max(self.repeat_std, self.max_reported_error))

    def to_dict(self) -> dict[str, object]:
        finite_h = [value for value in self.information if value is not None]
        return {
            "log_evidence_mean": float(self.log_evidence_mean),
            "repeat_std": float(self.repeat_std),
            "mean_reported_error": float(self.mean_reported_error),
            "max_reported_error": float(self.max_reported_error),
            "conservative_error": float(self.conservative_error),
            "n_repeats": int(self.n_repeats),
            "estimates": [float(value) for value in self.estimates],
            "errors": [float(value) for value in self.errors],
            "information": [None if value is None else float(value) for value in self.information],
            "information_mean": float(np.mean(finite_h)) if finite_h else None,
            "max_pairwise_z": (
                None if self.max_pairwise_z is None else float(self.max_pairwise_z)
            ),
        }


def summarize_evidence_repeats(results: Sequence[object]) -> EvidenceRepeatSummary:
    """Summarize independent runs exposing ``log_evidence``/``log_evidence_error``."""
    results = tuple(results)
    if not results:
        raise ValueError("at least one evidence result is required")
    estimates = tuple(float(item.log_evidence) for item in results)
    errors = tuple(float(item.log_evidence_error) for item in results)
    if not all(math.isfinite(value) for value in estimates):
        raise ValueError(f"evidence estimates must be finite; got {estimates}")
    if not all(math.isfinite(value) and value >= 0.0 for value in errors):
        raise ValueError(f"reported evidence errors must be finite and >= 0; got {errors}")
    information = tuple(
        None if getattr(item, "information", None) is None else float(item.information)
        for item in results
    )
    return EvidenceRepeatSummary(
        log_evidence_mean=float(np.mean(estimates)),
        repeat_std=float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        mean_reported_error=float(np.mean(errors)),
        max_reported_error=float(np.max(errors)),
        n_repeats=len(results),
        estimates=estimates,
        errors=errors,
        information=information,
        max_pairwise_z=max_pairwise_z(estimates, errors),
    )


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def run_diagnostics(result: DynestyResult, *, repeat: int | None = None) -> dict[str, object]:
    """Per-run nested-sampling diagnostics recorded with every evidence."""
    diagnostics = dict(result.diagnostics or {})
    config = result.config
    information = float(result.information)
    return {
        "repeat": None if repeat is None else int(repeat),
        "seed": int(result.seed),
        "log_evidence": float(result.log_evidence),
        "log_evidence_error": float(result.log_evidence_error),
        "information": information,
        "predicted_error": (
            float(math.sqrt(information / config.nlive)) if information >= 0.0 else None
        ),
        "niter": int(result.niter),
        "ncall": int(result.ncall),
        "efficiency_percent": float(result.efficiency),
        "kish_ess": float(result.kish_ess),
        "n_likelihood_evaluations": (
            None
            if result.n_likelihood_evaluations is None
            else int(result.n_likelihood_evaluations)
        ),
        "n_selection_unsupported": (
            None if result.n_selection_unsupported is None else int(result.n_selection_unsupported)
        ),
        "n_zero_likelihood_points": diagnostics.get("n_zero_likelihood_points"),
        "final_delta_logz": _optional_float(diagnostics.get("final_delta_logz")),
        "termination": run_termination(result),
        "elapsed_seconds": float(result.elapsed_seconds),
        "config": config.to_dict(),
        "versions": dict(result.versions),
    }
