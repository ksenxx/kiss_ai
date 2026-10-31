# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""One cost parser for every reader of the ``cost`` / ``cost_usd`` event fields.

The daemon client's ledger, the cost report, killed-task recovery and the
printer's ``cost_fields`` each parsed costs their own way and disagreed on
inputs no producer sends (``True``, ``"$1,234.5"``, ``" $1"``).  They now
share :func:`parse_cost` / :func:`event_cost`; the figures they read from
real events (``"$x.xxxx"``, ``"N/A"``, a float) are unchanged.
"""

from __future__ import annotations

import pytest

from kiss.agents.sorcar.persistence import _structured_usage_totals
from kiss.core.printer import cost_fields, event_cost, parse_cost


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("$0.1234", 0.1234),
        ("$12.0000", 12.0),
        (0.25, 0.25),
        (3, 3.0),
        ("$1,234.5", 1234.5),
        ("$1,25", None),
        ("$12,34,567", None),
        (" $1 ", 1.0),
        ("N/A", None),
        ("$abc", None),
        ("nan", None),
        ("", None),
        (None, None),
        (True, None),
    ],
)
def test_parse_cost(value: object, expected: float | None) -> None:
    """A number or a dollar figure parses; anything else is no cost."""
    assert parse_cost(value) == expected


def test_event_cost_prefers_the_exact_figure() -> None:
    """``cost_usd`` wins unless it is not a number; then the display string counts."""
    assert event_cost({"cost_usd": 0.000012, "cost": "$0.0000"}) == 0.000012
    assert event_cost({"cost_usd": True, "cost": "$0.5000"}) == 0.5
    assert event_cost({"cost": "N/A"}) is None


def test_recovery_keeps_the_text_cost_when_the_field_is_no_cost() -> None:
    """Killed-task recovery reports no cost from an event whose fields carry none."""
    event = {"type": "usage_info", "cost": "N/A", "total_steps": 3, "total_tokens": 9}
    assert _structured_usage_totals(event) == {"steps": 3, "tokens": 9}


def test_cost_fields_adds_the_offset_to_a_parsed_figure() -> None:
    """The producer side reads the same figures and passes junk through."""
    assert cost_fields("$1,000.5", None, 0.5) == ("$1001.0000", 1001.0)
    assert cost_fields("N/A", None, 1.0) == ("N/A", None)
