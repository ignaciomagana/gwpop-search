"""Adapters from external validated data products into canonical containers."""

from .gwcat_v2 import basis_for_spin as gwcat_v2_basis_for_spin
from .gwcat_v2 import load_pair as load_gwcat_v2_pair
from .gwcat_v2 import load_pe as load_gwcat_v2_pe
from .gwcat_v2 import load_selection as load_gwcat_v2_selection
from .gwcat_v2 import (
    validate_reference_pairing as validate_gwcat_v2_reference_pairing,
)

__all__ = [
    "gwcat_v2_basis_for_spin",
    "load_gwcat_v2_pair",
    "load_gwcat_v2_pe",
    "load_gwcat_v2_selection",
    "validate_gwcat_v2_reference_pairing",
]
