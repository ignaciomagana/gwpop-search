"""Frozen production-campaign configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    format_version: str = "gwpop-search-production-campaign-1.0"

    def __post_init__(self) -> None:
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
            format_version=str(
                payload.get(
                    "format_version",
                    "gwpop-search-production-campaign-1.0",
                )
            ),
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
