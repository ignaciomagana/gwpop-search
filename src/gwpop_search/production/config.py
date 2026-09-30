"""Frozen production-campaign configuration (format 2.0, dynesty ladder).

A campaign freezes the dataset manifest hash, the model graph (hash and
root), the code revision, the explicit model prior, the fidelity ladder v2
configuration (F0 -> F3 -> F4 on dynesty), the scheduler (with its ladder),
the seed policy, the budgets and the sampler backend pin
(``sampler_backend = {"name": "dynesty", "version": "3.1.0"}``; evidences and
checkpoints depend on that exact release). NUTS/JAXNS-era campaigns (1.x)
are refused: re-freeze them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from gwpop_search.inference.fidelity import (
    FidelityRunConfig,
    fidelity_run_config_from_dict,
    fidelity_run_config_to_dict,
)
from gwpop_search.search import SchedulerConfig

PRODUCTION_CAMPAIGN_FORMAT_VERSION = "gwpop-search-production-campaign-2.0"
LEGACY_PRODUCTION_CAMPAIGN_FORMATS = (
    "gwpop-search-production-campaign-1.0",
    "gwpop-search-production-campaign-1.1",
)
SUPPORTED_SAMPLER_BACKENDS = {"dynesty": ("3.1.0",)}


class LegacyCampaignError(ValueError):
    """A NUTS/JAXNS-era production campaign was presented to the dynesty pipeline."""


def validate_sampler_backend(payload: Mapping[str, object]) -> dict[str, str]:
    """Return a validated ``{"name", "version"}`` sampler-backend pin."""
    payload = dict(payload)
    if set(payload) != {"name", "version"}:
        raise ValueError("sampler_backend must have exactly the keys 'name' and 'version'")
    name, version = str(payload["name"]), str(payload["version"])
    if name not in SUPPORTED_SAMPLER_BACKENDS:
        raise ValueError(f"unsupported sampler backend {name!r}")
    if version not in SUPPORTED_SAMPLER_BACKENDS[name]:
        raise ValueError(
            f"{name} {version} is not a supported pin {SUPPORTED_SAMPLER_BACKENDS[name]}; "
            "evidences and checkpoints depend on the exact release"
        )
    return {"name": name, "version": version}


def installed_sampler_backend() -> dict[str, str]:
    """The installed dynesty as a sampler-backend pin (validated)."""
    from gwpop_search.inference.evidence import sampler_backend_identity

    return validate_sampler_backend(sampler_backend_identity())


def _default_sampler_backend() -> dict[str, str]:
    return {"name": "dynesty", "version": SUPPORTED_SAMPLER_BACKENDS["dynesty"][0]}


@dataclass(frozen=True)
class SearchBudget:
    max_gpu_hours: float
    max_f3_models: int
    max_f4_models: int
    max_null_replays: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_gpu_hours) or self.max_gpu_hours <= 0:
            raise ValueError("max_gpu_hours must be finite and positive")
        for name in ("max_f3_models", "max_f4_models", "max_null_replays"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SeedPolicy:
    root_seed: int
    policy: str = "sha256-derived-v1"

    def __post_init__(self) -> None:
        if self.policy != "sha256-derived-v1":
            raise ValueError("unsupported seed policy")


@dataclass(frozen=True)
class ProductionCampaignConfig:
    campaign_id: str
    dataset_manifest_hash: str
    model_graph_hash: str
    model_graph_root_hash: str
    git_commit: str
    model_prior: Mapping[str, object]
    fidelity: FidelityRunConfig
    scheduler: SchedulerConfig
    seed_policy: SeedPolicy
    budget: SearchBudget
    artifact_root: str
    state_database: str
    agents_enabled: bool = False
    sampler_backend: Mapping[str, str] = field(default_factory=_default_sampler_backend)
    format_version: str = PRODUCTION_CAMPAIGN_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version in LEGACY_PRODUCTION_CAMPAIGN_FORMATS:
            raise LegacyCampaignError(
                f"unsupported production campaign format {self.format_version!r}: "
                "NUTS/JAXNS-era campaign; re-freeze with the dynesty fidelity ladder"
            )
        if self.format_version != PRODUCTION_CAMPAIGN_FORMAT_VERSION:
            raise ValueError(
                f"unsupported production campaign format {self.format_version!r}"
            )
        if not isinstance(self.fidelity, FidelityRunConfig):
            raise TypeError("fidelity must be a FidelityRunConfig (format 2.0)")
        if not isinstance(self.scheduler, SchedulerConfig):
            raise TypeError("scheduler must be a SchedulerConfig")
        object.__setattr__(
            self, "sampler_backend", validate_sampler_backend(self.sampler_backend)
        )
        if not self.campaign_id:
            raise ValueError("campaign_id cannot be empty")
        for name in (
            "dataset_manifest_hash",
            "model_graph_hash",
            "model_graph_root_hash",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        if not self.git_commit or self.git_commit == "unknown":
            raise ValueError("production campaign requires a concrete git commit")
        if not self.artifact_root or not self.state_database:
            raise ValueError("artifact_root and state_database are required")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "campaign_id": self.campaign_id,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "model_graph_hash": self.model_graph_hash,
            "model_graph_root_hash": self.model_graph_root_hash,
            "git_commit": self.git_commit,
            "model_prior": dict(self.model_prior),
            "fidelity": fidelity_run_config_to_dict(self.fidelity),
            "scheduler": asdict(self.scheduler),
            "seed_policy": asdict(self.seed_policy),
            "budget": asdict(self.budget),
            "artifact_root": self.artifact_root,
            "state_database": self.state_database,
            "agents_enabled": bool(self.agents_enabled),
            "sampler_backend": dict(self.sampler_backend),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def campaign_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProductionCampaignConfig":
        version = payload.get("format_version")
        if version != PRODUCTION_CAMPAIGN_FORMAT_VERSION:
            found = "no format_version" if version is None else repr(version)
            raise LegacyCampaignError(
                f"unsupported production campaign format ({found}): NUTS/JAXNS-era "
                f"campaign; re-freeze as {PRODUCTION_CAMPAIGN_FORMAT_VERSION} with the "
                "dynesty fidelity ladder"
            )
        return cls(
            campaign_id=str(payload["campaign_id"]),
            dataset_manifest_hash=str(payload["dataset_manifest_hash"]),
            model_graph_hash=str(payload["model_graph_hash"]),
            model_graph_root_hash=str(payload["model_graph_root_hash"]),
            git_commit=str(payload["git_commit"]),
            model_prior=dict(payload["model_prior"]),
            fidelity=fidelity_run_config_from_dict(dict(payload["fidelity"])),
            scheduler=SchedulerConfig(**dict(payload["scheduler"])),
            seed_policy=SeedPolicy(**dict(payload["seed_policy"])),
            budget=SearchBudget(**dict(payload["budget"])),
            artifact_root=str(payload["artifact_root"]),
            state_database=str(payload["state_database"]),
            agents_enabled=bool(payload.get("agents_enabled", False)),
            sampler_backend=dict(payload["sampler_backend"]),
            format_version=str(version),
        )


def load_production_campaign(path: str | Path) -> ProductionCampaignConfig:
    return ProductionCampaignConfig.from_dict(json.loads(Path(path).read_text()))


def save_production_campaign(
    path: str | Path,
    config: ProductionCampaignConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2))
