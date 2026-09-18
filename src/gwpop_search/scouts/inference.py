"""Resumable standardized-HBI inference for conditional HSGP scouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from gwpop_search.grammar import ModelSpec
from gwpop_search.hbi import HBIConfig, shape_log_likelihood
from gwpop_search.inference.numpyro import (
    NUTSConfig,
    run_resumable_chains,
    save_result,
)
from gwpop_search.inference.recovery import chain_diagnostics

from .conditional import (
    ConditionalHSGPConfig,
    ConditionalHSGPResidualModel,
    coefficient_priors,
)
from .summary import (
    ConditionalMomentSummaryConfig,
    summarize_conditional_hsgp,
)


@dataclass(frozen=True)
class ScoutNumericalCriteria:
    max_r_hat: float = 1.05
    min_mcmc_n_eff: float = 200.0
    max_divergences: int = 0
    min_event_ess: float = 10.0
    min_selection_ess: float = 100.0
    max_event_weight_fraction: float = 0.35
    max_selection_weight_fraction: float = 0.15
    max_shape_log_likelihood_variance: float = 2.0

    def __post_init__(self) -> None:
        if self.max_r_hat <= 1.0:
            raise ValueError("max_r_hat must exceed one")
        if self.min_mcmc_n_eff <= 0.0:
            raise ValueError("min_mcmc_n_eff must be positive")
        if self.max_divergences < 0:
            raise ValueError("max_divergences cannot be negative")
        if self.min_event_ess <= 0.0 or self.min_selection_ess <= 0.0:
            raise ValueError("importance ESS thresholds must be positive")
        for name in (
            "max_event_weight_fraction",
            "max_selection_weight_fraction",
        ):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.max_shape_log_likelihood_variance <= 0.0:
            raise ValueError("likelihood variance threshold must be positive")


@dataclass(frozen=True)
class ConditionalScoutRunConfig:
    nuts: NUTSConfig = field(
        default_factory=lambda: NUTSConfig(
            num_warmup=1_000,
            num_samples=1_000,
            num_chains=4,
            target_accept_prob=0.90,
            max_tree_depth=10,
            progress_bar=False,
        )
    )
    hbi: HBIConfig = field(
        default_factory=lambda: HBIConfig(selection_chunk_size=4096)
    )
    numerical: ScoutNumericalCriteria = field(
        default_factory=ScoutNumericalCriteria
    )
    structure: ConditionalMomentSummaryConfig = field(
        default_factory=ConditionalMomentSummaryConfig
    )


def _importance_diagnostics(
    posterior,
    selection,
    model,
    hyperparameters,
    *,
    hbi_config: HBIConfig,
) -> dict[str, float]:
    evaluated = shape_log_likelihood(
        posterior,
        selection,
        model,
        hyperparameters,
        config=hbi_config,
    )
    events = evaluated.terms.events.diagnostics
    sel = evaluated.terms.selection.diagnostics
    return {
        "log_likelihood": float(evaluated.log_likelihood),
        "min_event_ess": float(min(item.ess for item in events)),
        "max_event_weight_fraction": float(
            max(item.max_weight_fraction for item in events)
        ),
        "selection_ess": float(sel.ess),
        "selection_max_weight_fraction": float(sel.max_weight_fraction),
        "shape_log_likelihood_variance": float(
            evaluated.terms.variance.shape_log_likelihood_variance
        ),
    }


def _check(name, value, limit, *, comparison):
    if value is None or not np.isfinite(float(value)):
        passed = False
    elif comparison == "le":
        passed = float(value) <= float(limit)
    elif comparison == "ge":
        passed = float(value) >= float(limit)
    else:  # pragma: no cover
        raise ValueError(comparison)
    return {
        "name": name,
        "value": None if value is None else float(value),
        "limit": float(limit),
        "comparison": comparison,
        "passed": bool(passed),
    }


def assess_scout_numerics(
    result,
    importance: Mapping[str, float],
    criteria: ScoutNumericalCriteria,
) -> dict[str, object]:
    chain = chain_diagnostics(result.samples)
    divergent = np.asarray(
        result.extra_fields.get("diverging", []),
        dtype=bool,
    )
    checks = [
        _check(
            "max_r_hat",
            chain["max_r_hat"],
            criteria.max_r_hat,
            comparison="le",
        ),
        _check(
            "min_mcmc_n_eff",
            chain["min_n_eff"],
            criteria.min_mcmc_n_eff,
            comparison="ge",
        ),
        _check(
            "n_divergent",
            int(divergent.sum()),
            criteria.max_divergences,
            comparison="le",
        ),
        _check(
            "min_event_ess",
            importance["min_event_ess"],
            criteria.min_event_ess,
            comparison="ge",
        ),
        _check(
            "selection_ess",
            importance["selection_ess"],
            criteria.min_selection_ess,
            comparison="ge",
        ),
        _check(
            "max_event_weight_fraction",
            importance["max_event_weight_fraction"],
            criteria.max_event_weight_fraction,
            comparison="le",
        ),
        _check(
            "selection_max_weight_fraction",
            importance["selection_max_weight_fraction"],
            criteria.max_selection_weight_fraction,
            comparison="le",
        ),
        _check(
            "shape_log_likelihood_variance",
            importance["shape_log_likelihood_variance"],
            criteria.max_shape_log_likelihood_variance,
            comparison="le",
        ),
    ]
    return {
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
        "chain": chain,
        "n_divergent": int(divergent.sum()),
        "importance": dict(importance),
    }


def run_conditional_hsgp_scout(
    run_dir: str | Path,
    posterior,
    selection,
    *,
    base_spec: ModelSpec,
    base_hyperparameters: Mapping[str, float],
    hsgp_config: ConditionalHSGPConfig,
    seed: int,
    config: ConditionalScoutRunConfig | None = None,
) -> tuple[object, dict[str, object]]:
    """Run/resume one flexible conditional scout and summarize legal descendants."""
    config = (
        ConditionalScoutRunConfig()
        if config is None
        else config
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ConditionalHSGPResidualModel(
        base_spec,
        base_hyperparameters,
        hsgp_config,
    )
    result = run_resumable_chains(
        run_dir / "chains",
        posterior,
        selection,
        model,
        coefficient_priors(hsgp_config),
        seed=int(seed),
        config=config.nuts,
        hbi_config=config.hbi,
    )
    save_result(run_dir / "posterior.npz", result)

    median = {
        name: float(np.median(np.asarray(values, dtype=float)))
        for name, values in result.samples.items()
    }
    importance = _importance_diagnostics(
        posterior,
        selection,
        model,
        median,
        hbi_config=config.hbi,
    )
    numerical = assess_scout_numerics(
        result,
        importance,
        config.numerical,
    )
    structure = summarize_conditional_hsgp(
        model,
        result.samples,
        config=config.structure,
    )

    raw_proposals = list(structure["proposals"])
    validated_proposals = raw_proposals if numerical["passed"] else []
    summary = {
        "format_version": "gwpop-search-conditional-hsgp-run-1.0",
        "base_model_hash": base_spec.model_hash,
        "base_hyperparameters": {
            str(name): float(value)
            for name, value in base_hyperparameters.items()
        },
        "hsgp_config": hsgp_config.to_dict(),
        "run_config": {
            "nuts": asdict(config.nuts),
            "hbi": {
                "rate_treatment": config.hbi.rate_treatment.value,
                "raw_selection_use_observing_time": bool(
                    config.hbi.raw_selection_use_observing_time
                ),
                "selection_chunk_size": config.hbi.selection_chunk_size,
            },
            "numerical": asdict(config.numerical),
            "structure": asdict(config.structure),
        },
        "numerical": numerical,
        "structure": structure,
        "raw_proposals": raw_proposals,
        "validated_proposals": validated_proposals,
        "proposal_firewall": (
            "numerical_pass_required"
            if numerical["passed"]
            else "suppressed_due_to_numerical_failure"
        ),
    }
    (run_dir / "scout_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2)
    )
    return result, summary
