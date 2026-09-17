"""Adapters from external validated data products into canonical containers."""

from .gwcat_v2 import load_pair as load_gwcat_v2_pair
from .gwcat_v2 import load_pe as load_gwcat_v2_pe
from .gwcat_v2 import load_selection as load_gwcat_v2_selection

__all__ = [
    "load_gwcat_v2_pair",
    "load_gwcat_v2_pe",
    "load_gwcat_v2_selection",
]
