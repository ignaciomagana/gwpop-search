"""Serializable Phase-3 hyperprior specifications."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class PriorSpec:
    family: str
    low: float | None = None
    high: float | None = None
    loc: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        family = self.family.lower()
        object.__setattr__(self, "family", family)

        if family in {"uniform", "log_uniform"}:
            if (
                self.low is None
                or self.high is None
                or not math.isfinite(self.low)
                or not math.isfinite(self.high)
                or self.high <= self.low
            ):
                raise ValueError(f"{family} requires finite low < high")
            if family == "log_uniform" and self.low <= 0:
                raise ValueError("log_uniform low must be positive")
        elif family == "normal":
            if (
                self.loc is None
                or self.scale is None
                or not math.isfinite(self.loc)
                or not math.isfinite(self.scale)
                or self.scale <= 0
            ):
                raise ValueError("normal requires finite loc and positive scale")
        else:
            raise ValueError(f"unsupported prior family {family!r}")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"family": self.family}
        for name in ("low", "high", "loc", "scale"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = float(value)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PriorSpec":
        return cls(**dict(payload))


def serialize_prior_map(priors: Mapping[str, PriorSpec]) -> dict[str, dict[str, object]]:
    return {str(name): spec.to_dict() for name, spec in sorted(priors.items())}


# These priors are for Phase-3 synthetic recovery only. They are not the
# production GWTC-5 prior contract and can be replaced when that comparison is frozen.
BASELINE_SYNTHETIC_PRIORS: dict[str, PriorSpec] = {
    "alpha": PriorSpec("uniform", low=0.0, high=8.0),
    "mmin": PriorSpec("uniform", low=2.0, high=10.0),
    "mmax": PriorSpec("uniform", low=60.0, high=120.0),
    "peak_fraction": PriorSpec("uniform", low=0.0, high=0.5),
    "peak_mu": PriorSpec("uniform", low=20.0, high=50.0),
    "peak_sigma": PriorSpec("log_uniform", low=1.0, high=15.0),
    "beta_q": PriorSpec("uniform", low=-4.0, high=12.0),
    "kappa": PriorSpec("uniform", low=-6.0, high=12.0),
    "chi_mu": PriorSpec("uniform", low=-0.3, high=0.3),
    "chi_sigma": PriorSpec("log_uniform", low=0.03, high=0.5),
}
