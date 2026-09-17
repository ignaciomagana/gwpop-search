"""Cross-validation for posterior and selection products."""

from __future__ import annotations

from typing import Iterable

from .posterior import PosteriorCatalog
from .schema import require_compatible_bases
from .selection import SelectionCatalog


def validate_pair(
    posterior: PosteriorCatalog,
    selection: SelectionCatalog,
    required_coordinates: Iterable[str] | None = None,
) -> None:
    """Fail before inference if PE/selection contracts cannot be paired."""

    require_compatible_bases(posterior.basis, selection.basis)
    if required_coordinates is not None:
        required = tuple(required_coordinates)
        posterior.require(required)
        selection.require(required)
