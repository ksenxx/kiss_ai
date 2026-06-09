# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Parity between the VS Code chat webview and the remote webapp HTML.

Both the VS Code extension (``SorcarTab.buildChatHtml``) and the
standalone remote web server (:func:`kiss.agents.vscode.web_server._build_html`)
now load the SAME ``media/chat.html`` template and only substitute
mode-specific values (CSP, nonces, webview URIs, the WS shim, the
auth-modal block).  These integration tests guard that single source
of truth: the shared template must contain the canonical layout, and
the remote-rendered output must still expose every id the extension
expects.
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.agents.vscode.web_server import _build_html

_CHAT_TPL = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "chat.html"
)


def _ext_html() -> str:
    """Return the shared chat HTML template used by the extension."""
    return _CHAT_TPL.read_text(encoding="utf-8")


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


_TAB_BAR_ACTIONS: tuple[str, ...] = ()


def test_webapp_tab_bar_contains_action_buttons() -> None:
    """Tab-bar action buttons in the webapp match those in the extension."""
    ext_bar = _section(_ext_html(), "tab-bar")
    web_bar = _section(_build_html(), "tab-bar")
    assert ext_bar, "could not locate #tab-bar in SorcarTab.ts"
    assert web_bar, "could not locate #tab-bar in webapp HTML"
    for btn in _TAB_BAR_ACTIONS:
        assert f'id="{btn}"' in ext_bar, f"sanity: extension tab-bar missing {btn}"
        assert f'id="{btn}"' in web_bar, f"webapp tab-bar missing {btn}"


def test_webapp_input_footer_omits_tab_bar_actions() -> None:
    """Tab-bar actions are not duplicated inside the input footer."""
    ext_footer = _section(_ext_html(), "input-footer")
    web_footer = _section(_build_html(), "input-footer")
    assert ext_footer and web_footer
    for btn in _TAB_BAR_ACTIONS:
        assert (
            f'id="{btn}"' not in ext_footer
        ), f"sanity: extension input-footer still has {btn}"
        assert (
            f'id="{btn}"' not in web_footer
        ), f"webapp input-footer still has {btn}"


def test_webapp_omits_work_dir_config_field() -> None:
    """The removed ``work_dir`` config field is absent from both webviews."""
    assert (
        "cfg-work-dir" not in _ext_html()
    ), "sanity: extension still ships the removed cfg-work-dir field"
    assert (
        "cfg-work-dir" not in _build_html()
    ), "webapp still ships the removed cfg-work-dir field"


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
