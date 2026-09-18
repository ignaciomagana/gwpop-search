"""Phase-3 synthetic baseline recovery campaign entry point."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from gwpop_search.hbi import HBIConfig, shape_log_likelihood
from gwpop_search.models import GwcatChiEffBBHModel

from .numpyro import NUTSConfig, run_resumable_chains, save_result
from .priors import BASELINE_SYNTHETIC_PRIORS
from .synthetic import SyntheticSurveyConfig, generate_baseline_synthetic_dataset


def posterior_summary(samples, truth):
    """Return compact per-parameter recovery summaries."""
    summary: dict[str, dict[str, float | bool]] = {}
    for name, values in samples.items():
        flat = np.asarray(values, dtype=float).reshape(-1)
        q05, q50, q95 = np.quantile(flat, [0.05, 0.50, 0.95])
        entry: dict[str, float | bool] = {
            "q05": float(q05),
            "median": float(q50),
            "q95": float(q95),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
        }
        if name in truth:
            t = float(truth[name])
            entry["truth"] = t
            entry["truth_in_90pct_interval"] = bool(q05 <= t <= q95)
        summary[name] = entry
    return summary


def chain_diagnostics(samples):
    """Return NumPyro split-Rhat and effective-sample-size diagnostics."""
    from numpyro.diagnostics import summary as numpyro_summary

    raw = numpyro_summary(samples, prob=0.90, group_by_chain=True)
    per_parameter: dict[str, dict[str, float]] = {}
    for name, stats in raw.items():
        r_hat = np.asarray(stats["r_hat"], dtype=float)
        n_eff = np.asarray(stats["n_eff"], dtype=float)
        per_parameter[name] = {
            "r_hat_max": float(np.nanmax(r_hat)),
            "n_eff_min": float(np.nanmin(n_eff)),
        }

    finite_rhat = [
        item["r_hat_max"]
        for item in per_parameter.values()
        if np.isfinite(item["r_hat_max"])
    ]
    finite_neff = [
        item["n_eff_min"]
        for item in per_parameter.values()
        if np.isfinite(item["n_eff_min"])
    ]
    return {
        "per_parameter": per_parameter,
        "max_r_hat": max(finite_rhat) if finite_rhat else None,
        "min_n_eff": min(finite_neff) if finite_neff else None,
    }


def importance_diagnostics(dataset, model, hyperparameters, hbi_config):
    """Summarize PE/selection importance quality at one hyperparameter point."""
    evaluated = shape_log_likelihood(
        dataset.posterior,
        dataset.selection,
        model,
        hyperparameters,
        config=hbi_config,
    )
    event_diagnostics = evaluated.terms.events.diagnostics
    selection_diagnostics = evaluated.terms.selection.diagnostics

    return {
        "log_likelihood": float(evaluated.log_likelihood),
        "min_event_ess": float(min(item.ess for item in event_diagnostics)),
        "min_event_ess_fraction": float(
            min(item.ess_fraction_of_draws for item in event_diagnostics)
        ),
        "max_event_weight_fraction": float(
            max(item.max_weight_fraction for item in event_diagnostics)
        ),
        "selection_ess": float(selection_diagnostics.ess),
        "selection_ess_fraction": float(
            selection_diagnostics.ess_fraction_of_draws
        ),
        "selection_max_weight_fraction": float(
            selection_diagnostics.max_weight_fraction
        ),
        "shape_log_likelihood_variance": float(
            evaluated.terms.variance.shape_log_likelihood_variance
        ),
    }


def run_synthetic_baseline_recovery(
    run_dir: str | Path,
    *,
    data_seed: int = 20260917,
    sampler_seed: int = 20260918,
    survey_config: SyntheticSurveyConfig | None = None,
    nuts_config: NUTSConfig | None = None,
    selection_chunk_size: int | None = 4096,
):
    """Generate the closed Phase-3 mock and run/resume the baseline NUTS fit."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    survey_cfg = SyntheticSurveyConfig() if survey_config is None else survey_config
    nuts_cfg = NUTSConfig() if nuts_config is None else nuts_config
    model = GwcatChiEffBBHModel()

    dataset = generate_baseline_synthetic_dataset(
        seed=int(data_seed),
        model=model,
        config=survey_cfg,
    )
    hbi_config = HBIConfig(selection_chunk_size=selection_chunk_size)

    chains_dir = run_dir / "chains"
    result = run_resumable_chains(
        chains_dir,
        dataset.posterior,
        dataset.selection,
        model,
        BASELINE_SYNTHETIC_PRIORS,
        seed=int(sampler_seed),
        config=nuts_cfg,
        hbi_config=hbi_config,
    )
    save_result(run_dir / "posterior.npz", result)

    diverging = np.asarray(result.extra_fields.get("diverging", []), dtype=bool)
    mcmc_diagnostics = chain_diagnostics(result.samples)
    median_hyperparameters = {
        name: float(np.median(np.asarray(values, dtype=float)))
        for name, values in result.samples.items()
    }
    truth_importance = importance_diagnostics(
        dataset,
        model,
        dataset.truth_hyperparameters,
        hbi_config,
    )
    median_importance = importance_diagnostics(
        dataset,
        model,
        median_hyperparameters,
        hbi_config,
    )
    summary = {
        "format_version": "gwpop-search-phase3-recovery-1.0",
        "data_seed": int(data_seed),
        "sampler_seed": int(sampler_seed),
        "survey_config": asdict(survey_cfg),
        "nuts_config": asdict(nuts_cfg),
        "hbi_config": {
            "selection_chunk_size": selection_chunk_size,
            "rate_treatment": "shape",
            "raw_selection_use_observing_time": True,
        },
        "truth_hyperparameters": dict(dataset.truth_hyperparameters),
        "posterior": posterior_summary(
            result.samples,
            dataset.truth_hyperparameters,
        ),
        "importance_diagnostics": {
            "truth": truth_importance,
            "posterior_median": median_importance,
        },
        "diagnostics": {
            "n_divergent": int(diverging.sum()) if diverging.size else None,
            "max_r_hat": mcmc_diagnostics["max_r_hat"],
            "min_n_eff": mcmc_diagnostics["min_n_eff"],
            "per_parameter": mcmc_diagnostics["per_parameter"],
            "n_events": int(dataset.posterior.n_events),
            "n_pe_samples": int(dataset.posterior.n_samples_total),
            "n_selected_injections": int(dataset.selection.n_selected),
            "n_draw_injections": int(dataset.selection.campaigns[0].n_draw),
        },
    }
    (run_dir / "recovery_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return result, summary
