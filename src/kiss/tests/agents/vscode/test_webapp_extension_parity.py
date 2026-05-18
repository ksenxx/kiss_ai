"""Parity between the VS Code chat webview and the remote webapp HTML.

The VS Code extension renders its chat panel from
``SorcarTab.buildChatHtml``.  The standalone remote webapp renders its
UI from :func:`kiss.agents.vscode.web_server._build_html`.  Whenever
the extension webview's button layout changes, the same change must be
reflected in the webapp so remote users see the same controls.  These
integration tests assert that the two surfaces stay in sync.

The tests use the real extension source file and the real
``_build_html`` output — no mocks, fakes, or fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.agents.vscode.web_server import _build_html

_SORCAR_TAB_TS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "src"
    / "SorcarTab.ts"
)


def _ext_html() -> str:
    """Return the chat HTML embedded in ``SorcarTab.ts``."""
    return _SORCAR_TAB_TS.read_text(encoding="utf-8")


def _section(html: str, container_id: str) -> str:
    """Return the markup of the ``<div>`` with the given id (balanced).

    Slices from the opening ``<div id="container_id">`` through the
    matching ``</div>``.  Correctly handles nested ``<div>`` elements
    by tracking depth.
    """
    pat = re.compile(r'<div\s+id="' + re.escape(container_id) + r'"')
    m = pat.search(html)
    if not m:
        return ""
    depth = 1
    i = m.end()
    while i < len(html) and depth:
        nxt = html.find("<", i)
        if nxt < 0:
            break
        if html.startswith("<div", nxt) and html[nxt + 4] in " \t\n>":
            depth += 1
            i = nxt + 4
        elif html.startswith("</div>", nxt):
            depth -= 1
            if depth == 0:
                return html[m.start() : nxt + len("</div>")]
            i = nxt + len("</div>")
        else:
            i = nxt + 1
    return html[m.start() :]


_TAB_BAR_ACTIONS = ("frequent-btn", "history-btn", "config-btn")


def test_webapp_tab_bar_contains_action_buttons() -> None:
    """Tab-bar action buttons in the webapp match those in the extension."""
    ext_bar = _section(_ext_html(), "tab-bar")
    web_bar = _section(_build_html(), "tab-bar")
    assert ext_bar, "could not locate #tab-bar in SorcarTab.ts"
    assert web_bar, "could not locate #tab-bar in webapp HTML"
    for btn in _TAB_BAR_ACTIONS:
        assert f'id="{btn}"' in ext_bar, f"sanity: extension tab-bar missing {btn}"
        assert f'id="{btn}"' in web_bar, f"webapp tab-bar missing {btn}"


def test_webapp_menu_dropdown_omits_tab_bar_actions() -> None:
    """Tab-bar actions are not duplicated inside the menu dropdown."""
    ext_menu = _section(_ext_html(), "menu-dropdown")
    web_menu = _section(_build_html(), "menu-dropdown")
    assert ext_menu and web_menu
    for btn in _TAB_BAR_ACTIONS:
        assert (
            f'id="{btn}"' not in ext_menu
        ), f"sanity: extension menu-dropdown still has {btn}"
        assert (
            f'id="{btn}"' not in web_menu
        ), f"webapp menu-dropdown still has {btn}"


def test_webapp_menu_dropdown_toggle_order_matches_extension() -> None:
    """Menu toggle buttons appear in the same order in both webviews."""
    ids = (
        "worktree-toggle-btn",
        "parallel-toggle-btn",
        "autocommit-toggle-btn",
        "demo-toggle-btn",
    )
    ext_menu = _section(_ext_html(), "menu-dropdown")
    web_menu = _section(_build_html(), "menu-dropdown")
    ext_positions = [(ext_menu.find(f'id="{i}"'), i) for i in ids]
    web_positions = [(web_menu.find(f'id="{i}"'), i) for i in ids]
    assert all(p >= 0 for p, _ in ext_positions), ext_positions
    assert all(p >= 0 for p, _ in web_positions), web_positions
    assert [i for _, i in sorted(ext_positions)] == list(ids)
    assert [i for _, i in sorted(web_positions)] == list(ids)


def test_webapp_omits_work_dir_config_field() -> None:
    """The removed ``work_dir`` config field is absent from both webviews."""
    assert (
        "cfg-work-dir" not in _ext_html()
    ), "sanity: extension still ships the removed cfg-work-dir field"
    assert (
        "cfg-work-dir" not in _build_html()
    ), "webapp still ships the removed cfg-work-dir field"


def test_webapp_autocommit_button_outside_menu_dropdown() -> None:
    """``#autocommit-btn`` is a top-level input-footer button, not a menu item."""
    web = _build_html()
    web_menu = _section(web, "menu-dropdown")
    assert 'id="autocommit-btn"' in web
    assert (
        'id="autocommit-btn"' not in web_menu
    ), "autocommit-btn should live outside the dropdown"


def test_webapp_delete_task_button_has_no_tooltip_attribute() -> None:
    """``data-tooltip`` was stripped from any delete-task button markup."""
    ext = _ext_html()
    web = _build_html()
    for src in (ext, web):
        # Match a button whose markup names both ``delete-task`` and
        # ``data-tooltip`` (in either order) within the same tag.
        pat = re.compile(
            r"<button[^>]*delete-task[^>]*data-tooltip[^>]*>"
            r"|<button[^>]*data-tooltip[^>]*delete-task[^>]*>",
            re.IGNORECASE,
        )
        assert pat.search(src) is None
