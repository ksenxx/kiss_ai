# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — YAML single-quote escaping in aria-snapshot names.

When an accessible name contains ``": "`` Playwright wraps the whole
YAML key in single quotes, and YAML single-quoted scalars escape an
embedded apostrophe by *doubling* it::

    - 'link "Bob''s: list"':

The name was recorded with the doubled apostrophe (``Bob''s: list``) and
looked up as such, which could never match: every click/type/hover on
such an element failed with "Element not found" even though it is
plainly on the page.
"""

from pathlib import Path

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

APOS_COLON_PAGE = """<!DOCTYPE html>
<html><head><title>Apos Page</title></head>
<body>
  <button onclick="document.title='Clicked!'">Bob's: list</button>
  <a href="#">plain link</a>
</body></html>"""


def test_click_element_with_apostrophe_and_colon_in_name(tmp_path: Path) -> None:
    """E2E: clicking an element whose accessible name contains an
    apostrophe plus ``": "`` must succeed, not report Element not found."""
    page = tmp_path / "apos.html"
    page.write_text(APOS_COLON_PAGE)
    tool = WebUseTool(user_data_dir=None, headless=True)
    try:
        tree = tool.go_to_url(page.as_uri())
        assert "- [1] 'button \"Bob''s: list\"'\n" in tree, tree

        result = tool.click(1)

        assert not result.startswith("Error"), result
        assert "Clicked!" in result, result
    finally:
        tool.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
