# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Catalog-drift check that stays behind from
``kiss.tests.agents.vscode.test_server_api`` (now
``kiss.tests.server.test_server_api``): it reads the real
``media/api.js`` asset (bundled under ``src/kiss/agents/vscode/media``),
so it lives in tests/agents/vscode while the server-only majority of
the original file moved to tests/server.
"""

from __future__ import annotations

import unittest

from kiss.server.sorcar import API


class TestCatalogSync(unittest.TestCase):
    """The handwritten client catalogs must not drift from the API."""

    def test_browser_catalog_is_a_subset_of_the_server_api(self) -> None:
        import re
        from pathlib import Path

        api_js = (
            Path(__file__).resolve().parents[3]
            / "agents"
            / "vscode"
            / "media"
            / "api.js"
        ).read_text()
        match = re.search(
            r"const SORCAR_API_COMMANDS = \[(.*?)\];", api_js, re.DOTALL
        )
        assert match is not None
        names = re.findall(r"'([A-Za-z]+)'", match.group(1))
        self.assertGreater(len(names), 30)
        missing = set(names) - set(API)
        self.assertEqual(
            missing,
            set(),
            f"media/api.js lists commands missing from the API: {missing}",
        )
