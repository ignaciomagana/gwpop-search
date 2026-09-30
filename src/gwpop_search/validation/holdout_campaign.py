"""Resumable K-fold held-out predictive validation with dynesty refits.

Per model and fold, the model is refit on the training events with ``repeats``
independent static dynesty runs (``run_dynesty_population``; ``bound='multi'``,
``sample='rslice'``, ``nlive=1000`` and ``slices = 2 (3 + ndim)`` by default,
decision D1) and the pooled posterior (equal mixture of separately normalized
runs) is gated with the F4 posterior and importance thresholds of decision D3
(cross-run rank-normalized R-hat <= 1.01, Kish ESS >= 2000 per run, dlogz
termination, and min event ESS 20 / selection ESS 200 / max event weight 0.25
/ max selection weight 0.10 / Var(log L) <= 1.0 at the posterior median, the
median over posterior draws and the tail); evidence precision is not gated.
Only folds that pass contribute held-out scores ``log E_post[ell_i / A]``
(weighted pooled points, vectorized backend terms) and a model total is
reported only when every fold passed. Totals are predictive diagnostics, not
Bayes factors. All dynesty evaluations use ``HBIConfig(selection_chunk_size=None)``
(decision D5) with the campaign's rate treatment and observing-time convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from gwpop_search.analysis._common import json_ready, pool_dynesty_results, write_json
from gwpop_search.analysis.posterior_gates import PosteriorGateCriteria, evaluate_posterior_gates
from gwpop_search.data import subset_posterior_events
from gwpop_search.grammar import ModelSpec
from gwpop_search.hbi import HBIConfig
from gwpop_search.inference.dynesty_backend import DynestyConfig, run_dynesty_population
from gwpop_search.inference.model_spec import prior_specs_from_model_spec
from gwpop_search.models import compile_model_spec

from .holdout import (
    compare_holdout_models,
    deterministic_event_folds,
    heldout_detected_log_predictive,
)

CAMPAIGN_FORMAT = "gwpop-search-holdout-campaign-2.0"
MANIFEST_FORMAT = "gwpop-search-holdout-manifest-2.0"
SUMMARY_FORMAT = "gwpop-search-holdout-summary-2.0"
SLICES_RULE = "2*(3+ndim)"


def _default_posterior_config() -> DynestyConfig:
    return DynestyConfig(nlive=1000, bound="multi", sample="rslice", dlogz=0.1)


@dataclass(frozen=True)
class HoldoutCampaignConfig:
    """K-fold holdout settings.

    ``posterior_config.slices=None`` together with ``slices_rule="2*(3+ndim)"``
    resolves the number of slices per model (decision D1); ``slices_rule=None``
    leaves ``slices`` as given (``None`` = dynesty's default).
    ``predictive_draws=None`` scores with every weighted posterior point;
    an integer uses that many pooled systematic-resampling draws.
    """

    n_folds: int = 5
    seed: int | None = None
    repeats: int = 2
    posterior_config: DynestyConfig = field(default_factory=_default_posterior_config)
    slices_rule: str | None = SLICES_RULE
    criteria: PosteriorGateCriteria = field(default_factory=PosteriorGateCriteria.f4)
    predictive_draws: int | None = None
    format_version: str = CAMPAIGN_FORMAT

    def __post_init__(self) -> None:
        if self.format_version != CAMPAIGN_FORMAT:
            raise ValueError(
                f"unsupported holdout campaign format {self.format_version!r}; the NUTS-era "
                "1.0 campaigns cannot be resumed with dynesty refits"
            )
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")
        if self.repeats < 2:
            raise ValueError("repeats must be at least two (cross-run R-hat gate)")
        if not isinstance(self.posterior_config, DynestyConfig):
            raise TypeError("posterior_config must be a DynestyConfig")
        if not isinstance(self.criteria, PosteriorGateCriteria):
            raise TypeError("criteria must be a PosteriorGateCriteria")
        if self.slices_rule not in (None, SLICES_RULE):
            raise ValueError(f"slices_rule must be None or {SLICES_RULE!r}")
        if self.predictive_draws is not None and int(self.predictive_draws) <= 0:
            raise ValueError("predictive_draws must be positive when supplied")

    def resolved_dynesty_config(self, ndim: int) -> DynestyConfig:
        cfg = self.posterior_config
        if (
            self.slices_rule == SLICES_RULE
            and cfg.slices is None
            and cfg.sample in ("slice", "rslice")
        ):
            cfg = replace(cfg, slices=2 * (3 + int(ndim)))
        return cfg

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "n_folds": int(self.n_folds),
            "seed": self.seed,
            "repeats": int(self.repeats),
            "posterior_config": self.posterior_config.to_dict(),
            "slices_rule": self.slices_rule,
            "criteria": self.criteria.to_dict(),
            "predictive_draws": self.predictive_draws,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HoldoutCampaignConfig":
        payload = dict(payload)
        return cls(
            n_folds=int(payload["n_folds"]),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
            repeats=int(payload["repeats"]),
            posterior_config=DynestyConfig.from_dict(dict(payload["posterior_config"])),
            slices_rule=payload.get("slices_rule"),
            criteria=PosteriorGateCriteria.from_dict(dict(payload["criteria"])),
            predictive_draws=(
                None if payload.get("predictive_draws") is None else int(payload["predictive_draws"])
            ),
            format_version=str(payload.get("format_version", "")),
        )


def _fit_seed(root_seed: int, model_hash: str, fold: int, repeat: int = 0) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{model_hash}:holdout:{int(fold)}:{int(repeat)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _manifest_identity(payload: Mapping[str, object]) -> dict[str, object]:
    """The resume identity (the checkpoint cadence and posterior size do not change results)."""
    payload = json.loads(json.dumps(json_ready(payload), sort_keys=True))
    posterior_config = payload.get("holdout_config", {}).get("posterior_config", {})
    for name in ("checkpoint_every", "num_posterior_samples"):
        posterior_config.pop(name, None)
    return payload


def _write_manifest_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if _manifest_identity(json.loads(path.read_text())) != _manifest_identity(payload):
            raise ValueError("existing holdout campaign manifest does not match request")
    else:
        write_json(path, payload)


def _hbi_config(campaign) -> HBIConfig:
    base = campaign.fidelity.hbi
    return HBIConfig(
        rate_treatment=base.rate_treatment,
        raw_selection_use_observing_time=bool(base.raw_selection_use_observing_time),
        selection_chunk_size=None,
        variance_taper=base.variance_taper,
    )


def run_holdout_campaign(
    root: str | Path,
    posterior,
    selection,
    campaign,
    *,
    dataset_identity: str,
    models: Iterable[ModelSpec],
    config: HoldoutCampaignConfig | None = None,
) -> dict[str, object]:
    """Refit each model on K-1 folds with dynesty and score the held-out events."""
    config = HoldoutCampaignConfig() if config is None else config
    models = tuple(models)
    if not models:
        raise ValueError("holdout campaign requires at least one model")
    hashes = [model.model_hash for model in models]
    if len(set(hashes)) != len(hashes):
        raise ValueError("holdout campaign model hashes must be unique")
    if config.n_folds > posterior.n_events:
        raise ValueError("n_folds cannot exceed the number of events")

    seed = campaign.seed_policy.root_seed if config.seed is None else int(config.seed)
    folds = deterministic_event_folds(
        posterior.event_names,
        n_folds=config.n_folds,
        seed=seed,
    )
    fold_members = {
        fold: tuple(
            name for name in posterior.event_names if folds[name] == fold
        )
        for fold in range(config.n_folds)
    }
    empty = [fold for fold, names in fold_members.items() if not names]
    if empty:
        raise ValueError(
            f"deterministic holdout assignment produced empty fold(s) {empty}; "
            "change the predeclared fold seed or n_folds before running"
        )

    hbi = _hbi_config(campaign)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": MANIFEST_FORMAT,
        "production_campaign_hash": campaign.campaign_hash,
        "dataset_identity": str(dataset_identity),
        "model_hashes": hashes,
        "n_folds": int(config.n_folds),
        "fold_seed": int(seed),
        "fold_assignment": folds,
        "holdout_config": config.to_dict(),
        "sampler_backend": "dynesty",
        "hbi_config": hbi.to_dict(),
    }
    _write_manifest_once(root / "manifest.json", manifest)

    model_summaries = []
    complete_scores: dict[str, dict[str, float]] = {}
    for model in models:
        population_model = compile_model_spec(model)
        priors = prior_specs_from_model_spec(model)
        dynesty_config = config.resolved_dynesty_config(len(priors))
        event_scores: dict[str, float] = {}
        fold_rows = []
        model_valid = True

        for fold in range(config.n_folds):
            heldout = fold_members[fold]
            training = tuple(
                name for name in posterior.event_names if name not in set(heldout)
            )
            train_posterior = subset_posterior_events(
                posterior,
                training,
                reason=f"holdout-fold-{fold}",
            )
            fold_dir = root / model.model_hash / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            seeds = [_fit_seed(seed, model.model_hash, fold, r) for r in range(config.repeats)]
            results = [
                run_dynesty_population(
                    train_posterior,
                    selection,
                    population_model,
                    priors,
                    seed=fit_seed,
                    config=dynesty_config,
                    hbi_config=hbi,
                    run_dir=fold_dir / f"repeat_{r:03d}",
                )
                for r, fit_seed in enumerate(seeds)
            ]
            diagnostics = evaluate_posterior_gates(
                results,
                train_posterior,
                selection,
                population_model,
                criteria=config.criteria,
                hbi_config=hbi,
                seed=seeds[0],
            )

            predictive = {}
            if diagnostics["passed"]:
                pooled = pool_dynesty_results(
                    results, n_draws=config.predictive_draws, seed=seeds[0]
                )
                predictive = heldout_detected_log_predictive(
                    posterior,
                    selection,
                    population_model,
                    {name: pooled.points[:, k] for k, name in enumerate(pooled.names)},
                    heldout_events=heldout,
                    config=hbi,
                    log_weights=np.log(pooled.weights),
                )
                event_scores.update(predictive)
            else:
                model_valid = False

            log_z = [float(r.log_evidence) for r in results]
            row = {
                "fold": int(fold),
                "fit_seeds": [int(s) for s in seeds],
                "training_events": list(training),
                "heldout_events": list(heldout),
                "dynesty_config": dynesty_config.to_dict(),
                "diagnostics": diagnostics,
                "training_log_evidence": {
                    "estimates": log_z,
                    "mean": float(np.mean(log_z)),
                    "interpretation": "advisory; not a held-out score",
                },
                "heldout_log_predictive": predictive,
            }
            write_json(fold_dir / "holdout_fold_summary.json", row)
            fold_rows.append(row)

        if set(event_scores) != set(posterior.event_names):
            model_valid = False
        if model_valid:
            complete_scores[model.model_hash] = event_scores

        model_summary = {
            "model_hash": model.model_hash,
            "all_folds_numerically_valid": bool(model_valid),
            "n_scored_events": len(event_scores),
            "event_scores": event_scores,
            "folds": fold_rows,
        }
        write_json(root / model.model_hash / "holdout_model_summary.json", model_summary)
        model_summaries.append(model_summary)

    totals = (
        compare_holdout_models(complete_scores)
        if complete_scores
        else {}
    )
    summary = {
        "format_version": SUMMARY_FORMAT,
        "production_campaign_hash": campaign.campaign_hash,
        "dataset_identity": str(dataset_identity),
        "n_folds": int(config.n_folds),
        "fold_seed": int(seed),
        "n_models": len(models),
        "n_models_complete": len(complete_scores),
        "model_total_log_predictive": totals,
        "models": model_summaries,
        "interpretation": (
            "posterior_predictive_validation_only; totals are not Bayes factors"
        ),
    }
    write_json(root / "holdout_summary.json", summary)
    return summary
