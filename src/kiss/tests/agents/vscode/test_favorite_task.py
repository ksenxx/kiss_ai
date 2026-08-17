# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the favourite (star) feature on the history panel.

Covers:

1. Backend: ``_get_history`` includes ``is_favorite`` (default False).
2. Backend: ``_handle_set_favorite`` persists the flag.
3. Command dispatch: a ``setFavorite`` command updates the DB row.
4. Frontend (static checks on ``media/main.js``): a star button is
   rendered for each history row and posts ``setFavorite`` on click.
"""

from __future__ import annotations

import re
from pathlib import Path

MAIN_JS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)
MAIN_CSS = (
    Path(__file__).parent.parent.parent.parent
    / "agents"
    / "vscode"
    / "media"
    / "main.css"
)


class TestMainJsFavoriteButton:
    """Static checks that ``media/main.js`` and ``media/main.css``
    contain the favourite-button wiring."""

    def _js(self) -> str:
        assert MAIN_JS.is_file(), f"main.js not found at {MAIN_JS}"
        return MAIN_JS.read_text()

    def _css(self) -> str:
        assert MAIN_CSS.is_file(), f"main.css not found at {MAIN_CSS}"
        return MAIN_CSS.read_text()

    def test_favorite_button_class_defined_in_js(self) -> None:
        src = self._js()
        assert "sidebar-item-favorite" in src, (
            "main.js must create a button with class sidebar-item-favorite"
        )

    def test_favorite_click_posts_set_favorite_message(self) -> None:
        src = self._js()
        assert re.search(
            r"api\.setFavorite\(\s*\{",
            src,
        ), "main.js must call api.setFavorite({...}) on click"
        api_src = MAIN_JS.with_name("api.js").read_text()
        assert "'setFavorite'" in api_src, (
            "api.js must catalog the setFavorite wire command"
        )

    def test_favorite_button_reads_s_is_favorite(self) -> None:
        src = self._js()
        assert "s.is_favorite" in src, (
            "main.js must read s.is_favorite to decide the icon state"
        )

    def test_favorite_css_defines_favorited_class(self) -> None:
        css = self._css()
        assert ".sidebar-item-favorite" in css
        assert ".sidebar-item-favorite.favorited" in css
