"""Resumable K-fold held-out predictive validation for frozen population models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from gwpop_search.data import subset_posterior_events
from gwpop_search.grammar import ModelSpec
from gwpop_search.inference.fidelity import summarize_nuts_fit
from gwpop_search.inference.model_spec import prior_specs_from_model_spec
from gwpop_search.inference.numpyro import run_resumable_chains, save_result
from gwpop_search.models import compile_model_spec
from gwpop_search.production import ProductionCampaignConfig

from .holdout import (
    compare_holdout_models,
    deterministic_event_folds,
    heldout_detected_log_predictive,
)


@dataclass(frozen=True)
class HoldoutCampaignConfig:
    n_folds: int = 5
    seed: int | None = None
    format_version: str = "gwpop-search-holdout-campaign-1.0"

    def __post_init__(self) -> None:
        if self.format_version != "gwpop-search-holdout-campaign-1.0":
            raise ValueError("unsupported holdout campaign format")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two")


def _fit_seed(root_seed: int, model_hash: str, fold: int) -> int:
    digest = hashlib.sha256(
        f"{int(root_seed)}:{model_hash}:holdout:{int(fold)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _write_manifest_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("existing holdout campaign manifest does not match request")
    else:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def run_holdout_campaign(
    root: str | Path,
    posterior,
    selection,
    campaign: ProductionCampaignConfig,
    *,
    dataset_identity: str,
    models: Iterable[ModelSpec],
    config: HoldoutCampaignConfig | None = None,
) -> dict[str, object]:
    """Refit each model on K-1 folds and score the held-out detected events."""
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

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": "gwpop-search-holdout-manifest-1.0",
        "production_campaign_hash": campaign.campaign_hash,
        "dataset_identity": str(dataset_identity),
        "model_hashes": hashes,
        "n_folds": int(config.n_folds),
        "fold_seed": int(seed),
        "fold_assignment": folds,
        "nuts_config": asdict(campaign.fidelity.f4_nuts),
        "hbi_config": {
            "rate_treatment": campaign.fidelity.hbi.rate_treatment.value,
            "raw_selection_use_observing_time": bool(
                campaign.fidelity.hbi.raw_selection_use_observing_time
            ),
            "selection_chunk_size": campaign.fidelity.hbi.selection_chunk_size,
        },
        "numerical_criteria": asdict(campaign.fidelity.f4_criteria),
    }
    _write_manifest_once(root / "manifest.json", manifest)

    model_summaries = []
    complete_scores: dict[str, dict[str, float]] = {}
    for model in models:
        population_model = compile_model_spec(model)
        priors = prior_specs_from_model_spec(model)
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
            fit_seed = _fit_seed(seed, model.model_hash, fold)

            result = run_resumable_chains(
                fold_dir / "nuts",
                train_posterior,
                selection,
                population_model,
                priors,
                seed=fit_seed,
                config=campaign.fidelity.f4_nuts,
                hbi_config=campaign.fidelity.hbi,
            )
            save_result(fold_dir / "posterior.npz", result)
            diagnostics = summarize_nuts_fit(
                result,
                train_posterior,
                selection,
                population_model,
                hbi_config=campaign.fidelity.hbi,
                criteria=campaign.fidelity.f4_criteria,
            )

            predictive = {}
            if diagnostics["passed"]:
                predictive = heldout_detected_log_predictive(
                    posterior,
                    selection,
                    population_model,
                    result.samples,
                    heldout_events=heldout,
                    config=campaign.fidelity.hbi,
                )
                event_scores.update(predictive)
            else:
                model_valid = False

            row = {
                "fold": int(fold),
                "fit_seed": int(fit_seed),
                "training_events": list(training),
                "heldout_events": list(heldout),
                "diagnostics": diagnostics,
                "heldout_log_predictive": predictive,
            }
            (fold_dir / "holdout_fold_summary.json").write_text(
                json.dumps(row, sort_keys=True, indent=2)
            )
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
        (root / model.model_hash / "holdout_model_summary.json").write_text(
            json.dumps(model_summary, sort_keys=True, indent=2)
        )
        model_summaries.append(model_summary)

    totals = (
        compare_holdout_models(complete_scores)
        if complete_scores
        else {}
    )
    summary = {
        "format_version": "gwpop-search-holdout-summary-1.0",
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
    (root / "holdout_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return summary
