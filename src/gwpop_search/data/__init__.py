"""Canonical data contracts for gwpop-search."""

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
from .thinning import (
    thin_catalog_pair,
    thin_posterior_catalog,
    thin_selection_catalog,
)

__all__ = [
    "BasisMismatchError",
    "Campaign",
    "CoordinateBasis",
    "DataContractError",
    "MissingCoordinateError",
    "PosteriorCatalog",
    "ReferenceDensityError",
    "SelectionCatalog",
    "SelectionMode",
    "thin_catalog_pair",
    "thin_posterior_catalog",
    "thin_selection_catalog",
    "validate_pair",
]
