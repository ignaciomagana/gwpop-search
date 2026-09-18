"""Serializable configuration for conditional-HSGP scout campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

from gwpop_search.hbi import HBIConfig
from gwpop_search.inference.numpyro import NUTSConfig

from .conditional import ConditionalHSGPConfig
from .hsgp import HSGPAxis
from .inference import (
    ConditionalScoutRunConfig,
    ScoutNumericalCriteria,
)
from .summary import ConditionalMomentSummaryConfig


@dataclass(frozen=True)
class ScoutCampaignConfig:
    hsgp: ConditionalHSGPConfig
    run: ConditionalScoutRunConfig
    format_version: str = "gwpop-search-hsgp-scout-campaign-1.0"

    def __post_init__(self) -> None:
        if self.format_version != "gwpop-search-hsgp-scout-campaign-1.0":
            raise ValueError("unsupported HSGP scout campaign format")

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "hsgp": self.hsgp.to_dict(),
            "run": {
                "nuts": asdict(self.run.nuts),
                "hbi": {
                    "rate_treatment": self.run.hbi.rate_treatment.value,
                    "raw_selection_use_observing_time": bool(
                        self.run.hbi.raw_selection_use_observing_time
                    ),
                    "selection_chunk_size": self.run.hbi.selection_chunk_size,
                },
                "numerical": asdict(self.run.numerical),
                "structure": asdict(self.run.structure),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ScoutCampaignConfig":
        run = dict(payload["run"])
        return cls(
            hsgp=ConditionalHSGPConfig.from_dict(dict(payload["hsgp"])),
            run=ConditionalScoutRunConfig(
                nuts=NUTSConfig(**dict(run["nuts"])),
                hbi=HBIConfig(**dict(run["hbi"])),
                numerical=ScoutNumericalCriteria(**dict(run["numerical"])),
                structure=ConditionalMomentSummaryConfig(
                    **dict(run["structure"])
                ),
            ),
            format_version=str(
                payload.get(
                    "format_version",
                    "gwpop-search-hsgp-scout-campaign-1.0",
                )
            ),
        )


def default_scout_campaign_config(
    target: str,
    covariate: str,
) -> ScoutCampaignConfig:
    target = str(target)
    covariate = str(covariate)

    if target == "q":
        target_axis = HSGPAxis("q", 0.05, 1.0, modes=6)
        target_length = 0.18
    elif target == "chi_eff":
        target_axis = HSGPAxis("chi_eff", -1.0, 1.0, modes=8)
        target_length = 0.30
    else:
        raise ValueError(f"unsupported HSGP target {target!r}")

    if covariate == "m1_source":
        covariate_axis = HSGPAxis(
            "m1_source",
            5.0,
            90.0,
            modes=6,
        )
        covariate_length = 18.0
    elif covariate == "q":
        covariate_axis = HSGPAxis("q", 0.05, 1.0, modes=6)
        covariate_length = 0.20
    elif covariate == "z":
        covariate_axis = HSGPAxis("z", 0.0, 2.5, modes=6)
        covariate_length = 0.50
    else:
        raise ValueError(f"unsupported HSGP covariate {covariate!r}")

    return ScoutCampaignConfig(
        hsgp=ConditionalHSGPConfig(
            target=target,
            covariate=covariate,
            target_axis=target_axis,
            covariate_axis=covariate_axis,
            amplitude=1.0,
            target_length_scale=target_length,
            covariate_length_scale=covariate_length,
            quadrature_order=64,
        ),
        run=ConditionalScoutRunConfig(),
    )


def save_scout_campaign_config(
    path: str | Path,
    config: ScoutCampaignConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2))


def load_scout_campaign_config(
    path: str | Path,
) -> ScoutCampaignConfig:
    return ScoutCampaignConfig.from_dict(json.loads(Path(path).read_text()))
