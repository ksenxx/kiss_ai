# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — YAML single-quote escaping in aria-snapshot names.

When an accessible name contains ``": "`` Playwright wraps the whole
YAML key in single quotes, and YAML single-quoted scalars escape an
embedded apostrophe by *doubling* it::

    - 'link "Bob''s: list"':

``_number_interactive_elements`` unescapes only the double-quote-style
escapes (``\\"`` and ``\\\\``), so the recorded accessible name keeps
the doubled apostrophe (``Bob''s: list``).  ``_resolve_locator`` then
calls ``get_by_role(..., name="Bob''s: list", exact=True)`` which can
never match, so every click/type/hover on such an element fails with
"Element not found" even though it is plainly on the page.
"""

from pathlib import Path

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool, _number_interactive_elements

APOS_COLON_PAGE = """<!DOCTYPE html>
<html><head><title>Apos Page</title></head>
<body>
  <button onclick="document.title='Clicked!'">Bob's: list</button>
  <a href="#">plain link</a>
</body></html>"""


def test_single_quoted_yaml_name_unescapes_doubled_apostrophe() -> None:
    """Parse-level: the doubled ``''`` inside a single-quoted YAML key
    must be collapsed back to a single apostrophe."""
    # Exact serialization produced by Playwright's aria_snapshot() for
    # an accessible name containing both an apostrophe and ": ".
    snap = "- 'link \"Bob''s: list\"':\n  - /url: \"#\""

    _, elements = _number_interactive_elements(snap)

    assert elements == [{"role": "link", "name": "Bob's: list"}]


def test_plain_double_quoted_name_keeps_literal_apostrophes() -> None:
    """Names NOT wrapped in single quotes must keep ``''`` verbatim."""
    snap = '- button "a\'\'b"'

    _, elements = _number_interactive_elements(snap)

    assert elements == [{"role": "button", "name": "a''b"}]


def test_click_element_with_apostrophe_and_colon_in_name(tmp_path: Path) -> None:
    """E2E: clicking an element whose accessible name contains an
    apostrophe plus ``": "`` must succeed, not report Element not found."""
    page = tmp_path / "apos.html"
    page.write_text(APOS_COLON_PAGE)
    tool = WebUseTool(user_data_dir=None, headless=True)
    try:
        tree = tool.go_to_url(page.as_uri())
        assert "Bob" in tree, tree

        # Find the [N] id assigned to the button.
        element_id = None
        for entry in tool._elements:
            if entry["role"] == "button":
                element_id = tool._elements.index(entry) + 1
                break
        assert element_id is not None, tool._elements

        result = tool.click(element_id)

        assert not result.startswith("Error"), result
        assert "Clicked!" in result, result
    finally:
        tool.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
