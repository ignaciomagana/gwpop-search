"""Bridge declarative model hyperpriors to sampler-independent :class:`PriorSpec` maps.

The resulting map is the hyperprior every sampler integrates against (the
dynesty prior transform of the production ladder, and the optional NumPyro
HSGP scouts); it is part of the model hash and never changed here.
"""

from __future__ import annotations

from gwpop_search.grammar import ModelSpec

from .priors import PriorSpec


def prior_specs_from_model_spec(model: ModelSpec) -> dict[str, PriorSpec]:
    result: dict[str, PriorSpec] = {}
    for name, prior in model.priors.items():
        parameters = dict(prior.parameters)
        if prior.family in {"uniform", "log_uniform"}:
            result[name] = PriorSpec(
                prior.family,
                low=parameters["low"],
                high=parameters["high"],
            )
        elif prior.family == "normal":
            result[name] = PriorSpec(
                "normal",
                loc=parameters["loc"],
                scale=parameters["scale"],
            )
        else:  # guarded by grammar PriorConfig
            raise ValueError(prior.family)
    return result
