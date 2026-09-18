"""Phase-3 synthetic baseline recovery campaign entry point."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from gwpop_search.hbi import HBIConfig
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
        "diagnostics": {
            "n_divergent": int(diverging.sum()) if diverging.size else None,
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
