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
    """Likelihood configuration.

    ``variance_taper`` (a :class:`gwpop_search.hbi.taper.VarianceTaper`, a
    mapping of its fields, or ``None``) multiplies the rate-marginalised shape
    likelihood by a taper on its Monte-Carlo variance ``sigma^2_lnL``
    (GWTC-5 / Callister & Farr 2024). ``None`` (the default) is the untapered
    likelihood of every pre-v2 run; serializers omit the key then, so older
    likelihood identities and configuration hashes are unchanged.
    """

    rate_treatment: RateTreatment = RateTreatment.SHAPE
    raw_selection_use_observing_time: bool = True
    selection_chunk_size: int | None = None
    variance_taper: Any = None

    def __post_init__(self) -> None:
        from .taper import VarianceTaper

        object.__setattr__(self, "rate_treatment", RateTreatment(self.rate_treatment))
        if self.selection_chunk_size is not None and int(self.selection_chunk_size) <= 0:
            raise ValueError("selection_chunk_size must be positive when supplied")
        taper = VarianceTaper.coerce(self.variance_taper)
        if taper is not None and self.rate_treatment is not RateTreatment.SHAPE:
            raise ValueError(
                "the variance taper is defined for the rate-marginalised shape likelihood only"
            )
        object.__setattr__(self, "variance_taper", taper)

    def to_dict(self) -> dict[str, object]:
        """JSON form; ``variance_taper`` appears only when a taper is configured."""
        payload: dict[str, object] = {
            "rate_treatment": self.rate_treatment.value,
            "raw_selection_use_observing_time": bool(self.raw_selection_use_observing_time),
            "selection_chunk_size": (
                None if self.selection_chunk_size is None else int(self.selection_chunk_size)
            ),
        }
        if self.variance_taper is not None:
            payload["variance_taper"] = self.variance_taper.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HBIConfig":
        payload = dict(payload)
        known = {
            "rate_treatment",
            "raw_selection_use_observing_time",
            "selection_chunk_size",
            "variance_taper",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown HBI configuration field(s): {unknown}")
        return cls(**payload)

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
    """``log_likelihood`` is the (tapered, when a variance taper is configured)
    likelihood; the taper fields are ``None`` without a taper."""

    log_likelihood: float
    terms: CatalogTerms
    rate_treatment: RateTreatment
    rate: float | None = None
    log_likelihood_untapered: float | None = None
    taper_variance: float | None = None
    log_taper: float | None = None
