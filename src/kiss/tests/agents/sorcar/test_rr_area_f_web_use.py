# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the O(n) element numbering (F-R3).

``_number_interactive_elements`` used to recompute each element's
``occurrence``/``role_occurrence`` by rescanning the whole accumulated
element list per element (O(n^2) on every accessibility-tree render);
it now keeps running counters.  These tests assert the numbering is
IDENTICAL to the definitional per-element rescan on snapshots full of
duplicate (role, name) pairs.
"""

from __future__ import annotations

from kiss.agents.sorcar.web_use_tool import _number_interactive_elements


def _brute_force_occurrences(
    elements: list[dict[str, str]],
) -> list[tuple[str, str]]:
    """Recompute (occurrence, role_occurrence) by definition: rescans."""
    expected: list[tuple[str, str]] = []
    for i, e in enumerate(elements):
        occurrence = sum(
            1 for prev in elements[:i]
            if prev["role"] == e["role"] and prev["name"] == e["name"]
        )
        role_occurrence = sum(
            1 for prev in elements[:i] if prev["role"] == e["role"]
        )
        expected.append((str(occurrence), str(role_occurrence)))
    return expected


def test_numbering_identical_on_page_with_duplicates() -> None:
    # Duplicate (role, name) pairs, same role with different names,
    # non-interactive lines, and nameless elements all mixed together.
    snapshot = "\n".join([
        '- heading "Products"',
        '- button "Add to cart"',
        '- link "Details"',
        '- button "Add to cart"',
        '- textbox "Search"',
        '- button "Add to cart"',
        '- link "Details"',
        '- button "Checkout"',
        "- button",
        "- button",
        '- link "Details"',
    ])
    numbered, elements = _number_interactive_elements(snapshot)
    assert len(elements) == 10  # heading is not interactive
    got = [(e["occurrence"], e["role_occurrence"]) for e in elements]
    assert got == _brute_force_occurrences(elements)
    # Spot-check document-order disambiguation of exact duplicates.
    carts = [e for e in elements if e["name"] == "Add to cart"]
    assert [e["occurrence"] for e in carts] == ["0", "1", "2"]
    details = [e for e in elements if e["name"] == "Details"]
    assert [e["occurrence"] for e in details] == ["0", "1", "2"]
    checkout = next(e for e in elements if e["name"] == "Checkout")
    assert checkout["occurrence"] == "0"
    # role_occurrence counts ALL buttons before it, named or not.
    assert checkout["role_occurrence"] == "3"
    nameless = [e for e in elements if e["role"] == "button" and not e["name"]]
    assert [e["occurrence"] for e in nameless] == ["0", "1"]
    # The visible numbering is sequential over interactive lines only.
    assert "- [1] button" in numbered and "- [10] link" in numbered
    assert 'heading "Products"' in numbered and "[0]" not in numbered


def test_numbering_identical_on_large_uniform_page() -> None:
    # 300 identical rows — the worst case for the old rescan.
    snapshot = "\n".join('- link "row"' for _ in range(300))
    _, elements = _number_interactive_elements(snapshot)
    assert len(elements) == 300
    got = [(e["occurrence"], e["role_occurrence"]) for e in elements]
    assert got == _brute_force_occurrences(elements)
    assert got[0] == ("0", "0") and got[299] == ("299", "299")
