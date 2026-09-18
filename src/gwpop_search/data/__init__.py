"""Canonical data contracts for gwpop-search."""

from .canonicalize import canonicalize_gwcat_v2_pair
from .pair import validate_pair
from .posterior import PosteriorCatalog
from .schema import (
    BasisMismatchError,
    CoordinateBasis,
    DataContractError,
    MissingCoordinateError,
    ReferenceDensityError,
)
from .selection import Campaign, SelectionCatalog, SelectionMode
from .subset import drop_posterior_events, subset_posterior_events
from .thinning import (
    thin_catalog_pair,
    thin_posterior_catalog,
    thin_selection_catalog,
)

__all__ = [
    "BasisMismatchError",
    "canonicalize_gwcat_v2_pair",
    "Campaign",
    "CoordinateBasis",
    "DataContractError",
    "MissingCoordinateError",
    "PosteriorCatalog",
    "ReferenceDensityError",
    "SelectionCatalog",
    "SelectionMode",
    "drop_posterior_events",
    "subset_posterior_events",
    "thin_catalog_pair",
    "thin_posterior_catalog",
    "thin_selection_catalog",
    "validate_pair",
]
