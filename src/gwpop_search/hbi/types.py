"""Typed results and configuration for the standardized HBI engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

class PopulationLogDensity(Protocol):
    def __call__(self, samples: Mapping[str, Any], hyperparameters: Any) -> Any: ...

class RateTreatment(str, Enum):
    SHAPE = "shape"
    POISSON = "poisson"

@dataclass(frozen=True)
class HBIConfig:
    rate_treatment: RateTreatment = RateTreatment.SHAPE
    raw_selection_use_observing_time: bool = True
    selection_chunk_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rate_treatment", RateTreatment(self.rate_treatment))
        if self.selection_chunk_size is not None and int(self.selection_chunk_size) <= 0:
            raise ValueError("selection_chunk_size must be positive when supplied")

@dataclass(frozen=True)
class ImportanceDiagnostics:
    n_retained: int
    n_draw: int
    n_zero_weight: int
    ess: float
    ess_fraction_of_draws: float
    max_weight_fraction: float
    variance_log_estimate: float

@dataclass(frozen=True)
class EventLikelihoodResult:
    event_names: tuple[str, ...]
    log_likelihoods: Any
    diagnostics: tuple[ImportanceDiagnostics, ...]

    @property
    def log_likelihood(self) -> float:
        return float(self.log_likelihoods.sum())

@dataclass(frozen=True)
class CampaignSelectionResult:
    campaign_id: str
    log_exposure: float
    log_efficiency: float | None
    diagnostics: ImportanceDiagnostics

@dataclass(frozen=True)
class SelectionResult:
    log_exposure: float
    diagnostics: ImportanceDiagnostics
    campaigns: tuple[CampaignSelectionResult, ...]
    variance_log_exposure: float

    @property
    def exposure(self) -> float:
        import numpy as np
        return float(np.exp(self.log_exposure))

@dataclass(frozen=True)
class LikelihoodVarianceDiagnostics:
    event_variance: float
    selection_variance: float
    shape_log_likelihood_variance: float

    def poisson_log_likelihood_variance(self, expected_count: float) -> float:
        return float(self.event_variance + expected_count**2 * self.selection_variance)

@dataclass(frozen=True)
class CatalogTerms:
    events: EventLikelihoodResult
    selection: SelectionResult
    variance: LikelihoodVarianceDiagnostics

@dataclass(frozen=True)
class CatalogLikelihoodResult:
    log_likelihood: float
    terms: CatalogTerms
    rate_treatment: RateTreatment
    rate: float | None = None
