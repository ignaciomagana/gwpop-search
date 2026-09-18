"""Concrete F0--F4 evaluator backed by the standardized HBI engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from gwpop_search.data import thin_catalog_pair
from gwpop_search.grammar import ModelSpec
from gwpop_search.hbi import HBIConfig, shape_log_likelihood
from gwpop_search.models import compile_model_spec
from gwpop_search.search.scheduler import EvaluationRecord, Fidelity

from .campaign import RecoveryAcceptanceCriteria
from .evidence import EvidenceResult, NestedSamplingConfig
from .evidence_campaign import (
    EvidenceCampaignConfig,
    run_model_evidence_repeats,
)
from .model_spec import prior_specs_from_model_spec
from .numpyro import (
    NUTSConfig,
    run_resumable_chains,
    save_result,
)
from .recovery import chain_diagnostics


@dataclass(frozen=True)
class NumericalCriteria:
    max_r_hat: float | None = None
    min_mcmc_n_eff: float | None = None
    max_divergences: int | None = None
    min_event_ess: float = 1.0
    min_selection_ess: float = 1.0
    max_event_weight_fraction: float = 1.0
    max_selection_weight_fraction: float = 1.0
    max_shape_log_likelihood_variance: float = math.inf
    max_evidence_error: float | None = None
    max_evidence_repeat_std: float | None = None

    def __post_init__(self) -> None:
        if self.max_r_hat is not None and self.max_r_hat <= 1.0:
            raise ValueError("max_r_hat must exceed one")
        if self.min_mcmc_n_eff is not None and self.min_mcmc_n_eff <= 0.0:
            raise ValueError("min_mcmc_n_eff must be positive")
        if self.max_divergences is not None and self.max_divergences < 0:
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
            raise ValueError(
                "max_shape_log_likelihood_variance must be positive"
            )
        for name in ("max_evidence_error", "max_evidence_repeat_std"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when supplied")


@dataclass(frozen=True)
class FidelityRunConfig:
    f0_pe_samples_per_event: int = 32
    f0_selected_per_campaign: int = 512
    f1_pe_samples_per_event: int = 128
    f1_selected_per_campaign: int = 5_000
    hbi: HBIConfig = field(
        default_factory=lambda: HBIConfig(selection_chunk_size=4096)
    )
    f1_nuts: NUTSConfig = field(
        default_factory=lambda: NUTSConfig(
            num_warmup=300,
            num_samples=300,
            num_chains=2,
            target_accept_prob=0.85,
            max_tree_depth=9,
            progress_bar=False,
        )
    )
    f2_nuts: NUTSConfig = field(
        default_factory=lambda: NUTSConfig(
            num_warmup=1_000,
            num_samples=1_000,
            num_chains=4,
            target_accept_prob=0.90,
            max_tree_depth=10,
            progress_bar=False,
        )
    )
    f4_nuts: NUTSConfig = field(
        default_factory=lambda: NUTSConfig(
            num_warmup=2_000,
            num_samples=2_000,
            num_chains=4,
            target_accept_prob=0.95,
            max_tree_depth=12,
            progress_bar=False,
        )
    )
    f3_evidence: EvidenceCampaignConfig = field(
        default_factory=lambda: EvidenceCampaignConfig(
            repeats=2,
            nested_sampling=NestedSamplingConfig(
                num_live_points=250,
                max_samples=100_000,
                dlogz=0.10,
                num_posterior_samples=1_000,
            ),
        )
    )
    f4_evidence: EvidenceCampaignConfig = field(
        default_factory=lambda: EvidenceCampaignConfig(
            repeats=3,
            nested_sampling=NestedSamplingConfig(
                num_live_points=500,
                max_samples=300_000,
                dlogz=0.03,
                num_posterior_samples=2_000,
            ),
        )
    )
    f1_criteria: NumericalCriteria = field(
        default_factory=lambda: NumericalCriteria(
            max_r_hat=1.10,
            min_mcmc_n_eff=50.0,
            max_divergences=5,
            min_event_ess=5.0,
            min_selection_ess=30.0,
            max_event_weight_fraction=0.50,
            max_selection_weight_fraction=0.25,
            max_shape_log_likelihood_variance=4.0,
        )
    )
    f2_criteria: NumericalCriteria = field(
        default_factory=lambda: NumericalCriteria(
            max_r_hat=1.05,
            min_mcmc_n_eff=200.0,
            max_divergences=0,
            min_event_ess=10.0,
            min_selection_ess=100.0,
            max_event_weight_fraction=0.35,
            max_selection_weight_fraction=0.15,
            max_shape_log_likelihood_variance=2.0,
        )
    )
    f3_criteria: NumericalCriteria = field(
        default_factory=lambda: NumericalCriteria(
            min_event_ess=10.0,
            min_selection_ess=100.0,
            max_event_weight_fraction=0.35,
            max_selection_weight_fraction=0.15,
            max_shape_log_likelihood_variance=2.0,
            max_evidence_error=0.50,
            max_evidence_repeat_std=0.50,
        )
    )
    f4_criteria: NumericalCriteria = field(
        default_factory=lambda: NumericalCriteria(
            max_r_hat=1.01,
            min_mcmc_n_eff=400.0,
            max_divergences=0,
            min_event_ess=20.0,
            min_selection_ess=200.0,
            max_event_weight_fraction=0.25,
            max_selection_weight_fraction=0.10,
            max_shape_log_likelihood_variance=1.0,
            max_evidence_error=0.20,
            max_evidence_repeat_std=0.20,
        )
    )

    def __post_init__(self) -> None:
        for name in (
            "f0_pe_samples_per_event",
            "f0_selected_per_campaign",
            "f1_pe_samples_per_event",
            "f1_selected_per_campaign",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def _prior_center(spec) -> float:
    parameters = spec.parameters
    if spec.family == "uniform":
        return 0.5 * (parameters["low"] + parameters["high"])
    if spec.family == "log_uniform":
        return float(np.sqrt(parameters["low"] * parameters["high"]))
    if spec.family == "normal":
        return float(parameters["loc"])
    raise ValueError(f"unsupported prior family {spec.family!r}")


def prior_center(model: ModelSpec) -> dict[str, float]:
    return {
        name: _prior_center(prior)
        for name, prior in model.priors.items()
    }


def posterior_median(samples: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.median(np.asarray(values, dtype=float)))
        for name, values in samples.items()
    }


def _importance_metrics(
    posterior,
    selection,
    population_model,
    hyperparameters,
    *,
    hbi_config: HBIConfig,
) -> dict[str, float]:
    evaluated = shape_log_likelihood(
        posterior,
        selection,
        population_model,
        hyperparameters,
        config=hbi_config,
    )
    event = evaluated.terms.events.diagnostics
    sel = evaluated.terms.selection.diagnostics
    return {
        "log_likelihood": float(evaluated.log_likelihood),
        "min_event_ess": float(min(item.ess for item in event)),
        "max_event_weight_fraction": float(
            max(item.max_weight_fraction for item in event)
        ),
        "selection_ess": float(sel.ess),
        "selection_max_weight_fraction": float(sel.max_weight_fraction),
        "shape_log_likelihood_variance": float(
            evaluated.terms.variance.shape_log_likelihood_variance
        ),
    }


def _check(
    name: str,
    value,
    limit,
    *,
    comparison: str,
) -> dict[str, object]:
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
        "comparison": comparison,
        "limit": float(limit),
        "passed": bool(passed),
    }


def _importance_checks(
    metrics: Mapping[str, float],
    criteria: NumericalCriteria,
) -> list[dict[str, object]]:
    return [
        _check(
            "min_event_ess",
            metrics["min_event_ess"],
            criteria.min_event_ess,
            comparison="ge",
        ),
        _check(
            "selection_ess",
            metrics["selection_ess"],
            criteria.min_selection_ess,
            comparison="ge",
        ),
        _check(
            "max_event_weight_fraction",
            metrics["max_event_weight_fraction"],
            criteria.max_event_weight_fraction,
            comparison="le",
        ),
        _check(
            "selection_max_weight_fraction",
            metrics["selection_max_weight_fraction"],
            criteria.max_selection_weight_fraction,
            comparison="le",
        ),
        _check(
            "shape_log_likelihood_variance",
            metrics["shape_log_likelihood_variance"],
            criteria.max_shape_log_likelihood_variance,
            comparison="le",
        ),
    ]


def _nuts_summary(
    result,
    posterior,
    selection,
    population_model,
    *,
    hbi_config: HBIConfig,
    criteria: NumericalCriteria,
) -> dict[str, object]:
    chain = chain_diagnostics(result.samples)
    divergent = np.asarray(
        result.extra_fields.get("diverging", []),
        dtype=bool,
    )
    median = posterior_median(result.samples)
    importance = _importance_metrics(
        posterior,
        selection,
        population_model,
        median,
        hbi_config=hbi_config,
    )
    checks = _importance_checks(importance, criteria)
    if criteria.max_r_hat is not None:
        checks.append(
            _check(
                "max_r_hat",
                chain["max_r_hat"],
                criteria.max_r_hat,
                comparison="le",
            )
        )
    if criteria.min_mcmc_n_eff is not None:
        checks.append(
            _check(
                "min_mcmc_n_eff",
                chain["min_n_eff"],
                criteria.min_mcmc_n_eff,
                comparison="ge",
            )
        )
    if criteria.max_divergences is not None:
        checks.append(
            _check(
                "n_divergent",
                int(divergent.sum()),
                criteria.max_divergences,
                comparison="le",
            )
        )
    return {
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
        "chain": chain,
        "n_divergent": int(divergent.sum()),
        "posterior_median": median,
        "importance": importance,
    }


def _combine_evidence_samples(
    results: list[EvidenceResult],
) -> dict[str, np.ndarray]:
    if not results:
        raise ValueError("evidence results cannot be empty")
    keys = set(results[0].posterior_samples)
    for result in results[1:]:
        keys &= set(result.posterior_samples)
    if not keys:
        raise ValueError("evidence repeats have no common posterior parameters")
    return {
        name: np.concatenate(
            [
                np.asarray(result.posterior_samples[name]).reshape(-1)
                for result in results
            ]
        )
        for name in sorted(keys)
    }


def _evidence_summary(
    results: list[EvidenceResult],
    summary: Mapping[str, object],
    posterior,
    selection,
    population_model,
    *,
    hbi_config: HBIConfig,
    criteria: NumericalCriteria,
) -> dict[str, object]:
    median = posterior_median(_combine_evidence_samples(results))
    importance = _importance_metrics(
        posterior,
        selection,
        population_model,
        median,
        hbi_config=hbi_config,
    )
    checks = _importance_checks(importance, criteria)
    if criteria.max_evidence_error is not None:
        checks.append(
            _check(
                "conservative_evidence_error",
                summary["conservative_error"],
                criteria.max_evidence_error,
                comparison="le",
            )
        )
    if criteria.max_evidence_repeat_std is not None:
        checks.append(
            _check(
                "evidence_repeat_std",
                summary["repeat_std"],
                criteria.max_evidence_repeat_std,
                comparison="le",
            )
        )
    return {
        "passed": bool(all(item["passed"] for item in checks)),
        "checks": checks,
        "posterior_median": median,
        "importance": importance,
        "evidence": dict(summary),
    }


def screening_score(
    log_likelihood: float,
    *,
    n_hyperparameters: int,
    n_events: int,
) -> float:
    """BIC-like allocation score used only at F1/F2."""
    return float(
        log_likelihood
        - 0.5 * int(n_hyperparameters) * np.log(max(int(n_events), 2))
    )


@dataclass
class DeterministicHBIEvaluator:
    posterior: object
    selection: object
    config: FidelityRunConfig = field(default_factory=FidelityRunConfig)
    dataset_identity: str = "unspecified"

    def _write_summary(
        self,
        run_dir: Path,
        *,
        model: ModelSpec,
        fidelity: Fidelity,
        diagnostics: Mapping[str, object],
        screen_value: float,
        elapsed_seconds: float,
    ) -> None:
        payload = {
            "format_version": "gwpop-search-fidelity-evaluation-1.0",
            "model_hash": model.model_hash,
            "fidelity": fidelity.value,
            "dataset_identity": self.dataset_identity,
            "screen_value": float(screen_value),
            "screen_value_semantics": (
                "bic_like_allocation_only"
                if fidelity in {Fidelity.F1_SCREEN, Fidelity.F2_INFERENCE}
                else (
                    "log_evidence_mean"
                    if fidelity in {Fidelity.F3_EVIDENCE, Fidelity.F4_PRODUCTION}
                    else "sanity_constant"
                )
            ),
            "elapsed_seconds": float(elapsed_seconds),
            "diagnostics": dict(diagnostics),
        }
        (run_dir / "evaluation.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2)
        )

    def evaluate(
        self,
        model: ModelSpec,
        fidelity: Fidelity,
        *,
        seed: int,
        run_dir: Path,
    ) -> EvaluationRecord:
        fidelity = Fidelity(fidelity)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()

        population_model = compile_model_spec(model)
        priors = prior_specs_from_model_spec(model)

        if fidelity is Fidelity.F0_SANITY:
            pe, sel = thin_catalog_pair(
                self.posterior,
                self.selection,
                max_samples_per_event=self.config.f0_pe_samples_per_event,
                max_selected_per_campaign=self.config.f0_selected_per_campaign,
                seed=seed,
            )
            metrics = _importance_metrics(
                pe,
                sel,
                population_model,
                prior_center(model),
                hbi_config=self.config.hbi,
            )
            diagnostics = {
                "passed": bool(np.isfinite(metrics["log_likelihood"])),
                "checks": [
                    {
                        "name": "finite_log_likelihood",
                        "value": metrics["log_likelihood"],
                        "passed": bool(np.isfinite(metrics["log_likelihood"])),
                    }
                ],
                "importance": metrics,
            }
            screen_value = 0.0

        elif fidelity in {Fidelity.F1_SCREEN, Fidelity.F2_INFERENCE}:
            if fidelity is Fidelity.F1_SCREEN:
                pe, sel = thin_catalog_pair(
                    self.posterior,
                    self.selection,
                    max_samples_per_event=self.config.f1_pe_samples_per_event,
                    max_selected_per_campaign=self.config.f1_selected_per_campaign,
                    seed=seed,
                )
                nuts_config = self.config.f1_nuts
                criteria = self.config.f1_criteria
            else:
                pe, sel = self.posterior, self.selection
                nuts_config = self.config.f2_nuts
                criteria = self.config.f2_criteria

            result = run_resumable_chains(
                run_dir / "nuts",
                pe,
                sel,
                population_model,
                priors,
                seed=seed,
                config=nuts_config,
                hbi_config=self.config.hbi,
            )
            save_result(run_dir / "posterior.npz", result)
            diagnostics = _nuts_summary(
                result,
                pe,
                sel,
                population_model,
                hbi_config=self.config.hbi,
                criteria=criteria,
            )
            screen_value = screening_score(
                diagnostics["importance"]["log_likelihood"],
                n_hyperparameters=len(model.priors),
                n_events=pe.n_events,
            )

        elif fidelity is Fidelity.F3_EVIDENCE:
            results, evidence = run_model_evidence_repeats(
                run_dir / "evidence",
                model,
                self.posterior,
                self.selection,
                root_seed=seed,
                config=self.config.f3_evidence,
                hbi_config=self.config.hbi,
                dataset_identity=self.dataset_identity,
            )
            diagnostics = _evidence_summary(
                results,
                evidence,
                self.posterior,
                self.selection,
                population_model,
                hbi_config=self.config.hbi,
                criteria=self.config.f3_criteria,
            )
            screen_value = float(evidence["log_evidence_mean"])

        elif fidelity is Fidelity.F4_PRODUCTION:
            result = run_resumable_chains(
                run_dir / "nuts",
                self.posterior,
                self.selection,
                population_model,
                priors,
                seed=seed,
                config=self.config.f4_nuts,
                hbi_config=self.config.hbi,
            )
            save_result(run_dir / "posterior.npz", result)
            nuts_diagnostics = _nuts_summary(
                result,
                self.posterior,
                self.selection,
                population_model,
                hbi_config=self.config.hbi,
                criteria=self.config.f4_criteria,
            )
            results, evidence = run_model_evidence_repeats(
                run_dir / "evidence",
                model,
                self.posterior,
                self.selection,
                root_seed=seed,
                config=self.config.f4_evidence,
                hbi_config=self.config.hbi,
                dataset_identity=self.dataset_identity,
            )
            evidence_diagnostics = _evidence_summary(
                results,
                evidence,
                self.posterior,
                self.selection,
                population_model,
                hbi_config=self.config.hbi,
                criteria=self.config.f4_criteria,
            )
            diagnostics = {
                "passed": bool(
                    nuts_diagnostics["passed"]
                    and evidence_diagnostics["passed"]
                ),
                "nuts": nuts_diagnostics,
                "evidence": evidence_diagnostics,
            }
            screen_value = float(evidence["log_evidence_mean"])

        else:  # pragma: no cover
            raise ValueError(f"unsupported fidelity {fidelity}")

        elapsed = time.perf_counter() - start
        self._write_summary(
            run_dir,
            model=model,
            fidelity=fidelity,
            diagnostics=diagnostics,
            screen_value=screen_value,
            elapsed_seconds=elapsed,
        )
        return EvaluationRecord(
            model_hash=model.model_hash,
            fidelity=fidelity,
            diagnostics_pass=bool(diagnostics["passed"]),
            screen_value=float(screen_value),
            compute_cost=float(elapsed / 3600.0),
            status="complete",
        )
