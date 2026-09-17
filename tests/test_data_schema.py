import pytest

from gwpop_search.data import BasisMismatchError, CoordinateBasis, DataContractError
from gwpop_search.data.schema import require_compatible_bases


def test_basis_identity_is_stable_and_semantic():
    a = CoordinateBasis("x", ("m1", "q"), "source", "none", "dm1 dq")
    b = CoordinateBasis("x", ("m1", "q"), "source", "none", "dm1 dq")
    c = CoordinateBasis("x", ("m1", "q"), "detector", "none", "dm1 dq")
    assert a.identity == b.identity
    assert a.identity != c.identity


def test_basis_rejects_duplicate_coordinates():
    with pytest.raises(DataContractError, match="not unique"):
        CoordinateBasis("bad", ("m1", "m1"), "source", "none", "dm1")


def test_pair_basis_mismatch_is_explicit():
    a = CoordinateBasis("a", ("m1",), "source", "none", "dm1")
    b = CoordinateBasis("b", ("m1",), "detector", "none", "dm1")
    with pytest.raises(BasisMismatchError, match="density-basis mismatch"):
        require_compatible_bases(a, b)
