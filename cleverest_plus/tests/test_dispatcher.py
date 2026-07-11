# pyright: reportImplicitRelativeImport=false
"""Dispatcher unit tests: exact-budget allocation is committed before observation."""
from __future__ import annotations

import pytest

from cleverest_plus.dispatcher import allocate_budget


@pytest.mark.parametrize("total,members,expected", [
    (10, 3, (4, 3, 3)),
    (10, 1, (10,)),
    (10, 5, (2, 2, 2, 2, 2)),
    (7, 3, (3, 2, 2)),
    (0, 3, (0, 0, 0)),
    (2, 3, (1, 1, 0)),
])
def test_allocation_is_deterministic_and_sums_to_total(total: int, members: int,
                                                      expected: tuple[int, ...]) -> None:
    got = allocate_budget(total, members)
    assert got == expected
    assert sum(got) == total


def test_negative_or_zero_members_rejected() -> None:
    with pytest.raises(ValueError):
        _ = allocate_budget(10, 0)
    with pytest.raises(ValueError):
        _ = allocate_budget(-1, 3)
