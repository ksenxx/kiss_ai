# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: the remote webapp is restyled like the Codex mobile app.

The remote webapp (served by ``RemoteAccessServer`` via
``_build_html()``) must adopt the OpenAI Codex / ChatGPT mobile visual
language — near-black ``#0d0d0d`` page, an elevated rounded ``#212121``
composer card with circular controls, pill-shaped chips, and rounded
drawer panels — WITHOUT changing any control: every DOM id present in
``media/chat.html`` must survive, and the VS Code extension webview
(which shares ``chat.html`` + ``main.css``) must be completely
unaffected.  The restyle therefore lives in a NEW stylesheet
``media/remote-codex.css`` whose every rule is scoped under
``body.remote-chat`` — the class only the remote page's
``BODY_CLASS_ATTR`` substitution adds.

Coverage:

* ``_build_html()`` links ``remote-codex.css`` with a cache-busted
  ``/media/`` URL (after ``main.css`` so it can override it).
* ``RemoteAccessServer._process_request`` serves the new stylesheet
  over HTTP with a CSS content type (real server, real socket).
* Control parity — the full inventory of element ids from
  ``chat.html`` is still present in the built page.
* Every top-level selector in ``remote-codex.css`` is scoped under
  ``body.remote-chat`` (including selectors inside ``@media`` blocks),
  so the VS Code webview cannot pick up any rule.
* The Codex design tokens are actually in the stylesheet: page
  ``#0d0d0d``, composer surface ``#212121`` at 28px radius with
  circular 36px controls, white circular send button, pill tab chips.
* ``chat.html`` itself does not hardcode the new stylesheet (the VS
  Code webview must never load it) and ``buildChatHtml``'s HEAD_STYLE
  placeholder path is untouched.
"""

from __future__ import annotations

import asyncio
import re
import ssl
import threading
import urllib.request
from pathlib import Path

import pytest

MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
CHAT_HTML = MEDIA_DIR / "chat.html"

# Every interactive-control id in media/chat.html.  If ANY id ever
# disappears from the built remote page, a control was lost and this
# parity guard must fail.
CONTROL_IDS = [
    "kiss-server-loading",
    "app",
    "tab-bar",
    "tab-list",
    "tab-status-bar",
    "status-text",
    "status-tokens",
    "status-budget",
    "status-steps",
    "task-panel",
    "task-panel-text",
    "task-panel-collapse-btn",
    "task-panel-copy",
    "output",
    "welcome",
    "welcome-config",
    "welcome-remote-url",
    "welcome-cfg-remote-password",
    "welcome-cfg-remote-password-toggle",
    "suggestions",
    "input-area",
    "autocomplete",
    "input-container",
    "file-chips",
    "input-wrap",
    "input-text-wrap",
    "ghost-overlay",
    "listening-overlay",
    "task-input",
    "input-clear-btn",
    "input-footer",
    "model-picker",
    "menu-btn",
    "model-btn",
    "model-name",
    "upload-btn",
    "tricks-btn",
    "voice-btn",
    "model-dropdown",
    "model-search",
    "model-search-clear",
    "model-list",
    "input-actions",
    "wait-spinner",
    "send-btn",
    "demo-pause-btn",
    "stop-btn",
    "sidebar",
    "sidebar-close",
    "sidebar-resizer",
    "sidebar-tab-history-panel",
    "history-search",
    "history-search-clear",
    "hf-running",
    "hf-errors",
    "hf-completed",
    "hf-workspace",
    "hf-favorite",
    "hf-from",
    "hf-from-btn",
    "hf-to",
    "hf-to-btn",
    "history-list",
    "sidebar-overlay",
    "frequent-panel",
    "frequent-panel-close",
    "frequent-list",
    "frequent-overlay",
    "tricks-panel",
    "tricks-panel-close",
    "tricks-list",
    "tricks-overlay",
    "settings-panel",
    "settings-panel-close",
    "remote-url",
    "config-form",
    "cfg-remote-password",
    "cfg-remote-password-toggle",
    "cfg-work-dir",
    "cfg-max-budget",
    "cfg-auto-commit",
    "cfg-use-worktree",
    "cfg-demo-mode",
    "cfg-voice-sensitivity",
    "cfg-voice-sensitivity-value",
    "tips-btn",
    "autocommit-btn",
    "cfg-update-btn",
    "cfg-server-reset-btn",
    "cfg-key-ANTHROPIC_API_KEY",
    "cfg-key-OPENAI_API_KEY",
    "cfg-key-ZAI_API_KEY",
    "cfg-key-MOONSHOT_API_KEY",
    "cfg-key-OPENROUTER_API_KEY",
    "cfg-key-TOGETHER_API_KEY",
    "cfg-key-GEMINI_API_KEY",
    "cfg-custom-endpoint",
    "cfg-custom-api-key",
    "cfg-custom-headers",
    "server-reset-confirm-modal",
    "server-reset-confirm-cancel",
    "server-reset-confirm-ok",
    "settings-overlay",
    "ask-user-modal",
    "ask-user-slot",
    "auth-modal",
    "auth-modal-input",
    "auth-modal-cancel",
    "auth-modal-ok",
]


def _build_html() -> str:
    from kiss.server.web_server import _build_html

    return _build_html()


def _read_codex_css() -> str:
    assert CODEX_CSS.is_file(), "media/remote-codex.css must exist"
    return CODEX_CSS.read_text(encoding="utf-8")


# ── Stylesheet wiring ────────────────────────────────────────────────


def test_built_html_links_codex_stylesheet_cache_busted() -> None:
    """The built remote page links remote-codex.css with ?v=<sha16>."""
    html = _build_html()
    m = re.search(
        r'<link href="(/media/remote-codex\.css\?v=[0-9a-f]{16})"'
        r'\s+rel="stylesheet">',
        html,
    )
    assert m, "remote-codex.css <link> missing from built HTML"


def test_codex_stylesheet_linked_after_main_css() -> None:
    """remote-codex.css must come AFTER main.css so overrides win."""
    html = _build_html()
    main_pos = html.find("/media/main.css")
    codex_pos = html.find("/media/remote-codex.css")
    assert main_pos != -1 and codex_pos != -1
    assert codex_pos > main_pos, (
        "remote-codex.css must be linked after main.css to override it"
    )


def test_no_unsubstituted_placeholders_remain() -> None:
    """Adding the link must not leave {{...}} placeholders behind."""
    html = _build_html()
    assert not re.search(r"\{\{[A-Z_]+\}\}", html)


# ── Control parity ───────────────────────────────────────────────────


def test_all_control_ids_still_present() -> None:
    """Every existing control/template id survives unchanged.

    The explicit inventory protects all known interactive controls,
    while the template-derived set catches non-interactive wiring ids
    added in the future (labels, live regions, modal descriptions, etc.).
    """
    html = _build_html()
    built_ids = set(re.findall(r'\bid="([^"]+)"', html))
    template = CHAT_HTML.read_text(encoding="utf-8")
    template_ids = set(re.findall(r'\bid="([^"]+)"', template))
    expected = set(CONTROL_IDS) | template_ids
    missing = sorted(expected - built_ids)
    assert not missing, f"controls lost from remote page: {missing}"


def test_body_keeps_remote_chat_class() -> None:
    """The remote page body keeps the remote-chat scoping class."""
    html = _build_html()
    assert '<body class="remote-chat">' in html


# ── VS Code webview isolation ────────────────────────────────────────


def test_chat_html_template_does_not_hardcode_codex_css() -> None:
    """chat.html is shared with the VS Code webview: it must not
    reference remote-codex.css directly (only the remote-only
    HEAD_STYLE substitution may inject it)."""
    tpl = CHAT_HTML.read_text(encoding="utf-8")
    assert "remote-codex" not in tpl


def test_main_css_not_polluted_with_codex_tokens() -> None:
    """main.css (shared with the extension) must not gain the Codex
    page-background token; the restyle lives only in remote-codex.css."""
    main_css = (MEDIA_DIR / "main.css").read_text(encoding="utf-8")
    assert "#0d0d0d" not in main_css


def _iter_top_level_selectors(css: str) -> list[str]:
    """Return every top-level selector in *css*, descending into
    at-rule blocks (@media/@supports/@container) one level."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    selectors: list[str] = []

    def scan(text: str) -> None:
        i = 0
        n = len(text)
        while i < n:
            brace = text.find("{", i)
            if brace == -1:
                break
            header = text[i:brace].strip()
            # find matching close brace
            depth = 1
            j = brace + 1
            while j < n and depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            body = text[brace + 1 : j - 1]
            if header.startswith("@"):
                if header.split("(")[0].split()[0] in (
                    "@media",
                    "@supports",
                    "@container",
                ):
                    scan(body)
                # @keyframes / @font-face bodies contain no selectors
                # that could leak into the extension webview.
            else:
                selectors.extend(
                    s.strip() for s in header.split(",") if s.strip()
                )
            i = j
        return

    scan(css)
    return selectors


def test_every_codex_rule_scoped_under_remote_chat() -> None:
    """EVERY selector in remote-codex.css must be scoped under
    body.remote-chat so the VS Code webview can never match it."""
    selectors = _iter_top_level_selectors(_read_codex_css())
    assert selectors, "remote-codex.css must contain rules"
    bad = [
        s
        for s in selectors
        if not (
            s.startswith("body.remote-chat")
            or s.startswith("html:has(body.remote-chat)")
        )
    ]
    assert not bad, f"unscoped selectors leak into VS Code webview: {bad}"


# ── Codex design language ────────────────────────────────────────────


def test_codex_page_palette() -> None:
    """Near-black Codex page background + #ececec primary text."""
    css = _read_codex_css()
    assert "#0d0d0d" in css, "Codex page background #0d0d0d missing"
    assert "#ececec" in css, "Codex primary text #ececec missing"


def test_codex_composer_card() -> None:
    """Composer = #212121 card, 28px radius, inset white edge."""
    css = _read_codex_css()
    assert "#212121" in css, "composer surface #212121 missing"
    assert "28px" in css, "28px composer radius missing"
    assert re.search(
        r"inset 0 0 1px rgba?\(255[,\s]+255[,\s]+255", css
    ), "inset white edge shadow missing"


def test_codex_circular_composer_controls() -> None:
    """Composer controls are 36px circles; send is a white circle."""
    css = _read_codex_css()
    assert re.search(
        r"body\.remote-chat #send-btn[^{]*\{[^}]*background:\s*#fff",
        css,
    ), "send button must be a white circle"
    assert "36px" in css, "36px circular control size missing"
    m = re.search(r"body\.remote-chat #send-btn[^{]*\{([^}]*)\}", css)
    assert m and "border-radius: 50%" in m.group(1)


def test_codex_pill_tabs_and_status() -> None:
    """Tab chips and the status row become rounded pills."""
    css = _read_codex_css()
    assert re.search(
        r"body\.remote-chat #tab-bar .chat-tab[^{]*\{[^}]*"
        r"border-radius:\s*(999px|9999px)",
        css,
    ), "tab chips must be pills"
    assert "body.remote-chat #tab-status-bar" in css


def test_codex_user_prompt_bubble() -> None:
    """The pinned user prompt is a right-aligned dark bubble on the
    composer's #212121 surface so it blends with the dark thread."""
    css = _read_codex_css()
    m = re.search(r"body\.remote-chat #task-panel\s*\{([^}]*)\}", css)
    assert m, "remote task/user-prompt rule missing"
    rule = m.group(1)
    assert "background: #212121" in rule
    assert "color: #ececec" in rule
    assert "border-radius: 22px" in rule
    assert "max-width: 90%" in rule
    assert "margin-left: auto" in rule


def test_codex_rounded_panels() -> None:
    """Sidebar / settings / modals get the dark rounded treatment."""
    css = _read_codex_css()
    assert "body.remote-chat #sidebar" in css
    assert "body.remote-chat #settings-panel" in css
    assert "#171717" in css, "drawer surface #171717 missing"


# ── Desktop docked sidebar + colorless chat panels ──────────────────


def _find_rule(css: str, selector: str) -> str:
    """Return the declaration body of the first ``body.remote-chat``
    scoped rule for *selector* in remote-codex.css, or fail."""
    pattern = (
        r"body\.remote-chat[^{,]*"
        + re.escape(selector)
        + r"\s*(?:,[^{]*)?\{([^}]*)\}"
    )
    m = re.search(pattern, css)
    assert m, f"body.remote-chat scoped rule for {selector!r} missing"
    return m.group(1)


def test_desktop_media_query_docks_sidebar() -> None:
    """A min-width:900px block docks the history sidebar on the left
    (transform:none, fixed 300px column, overlay hidden, #app cleared)."""
    css = _read_codex_css()
    m = re.search(
        r"@media \((?:min-width: 900px|width >= 900px)\)\s*\{(.*)\}\s*$",
        css,
        flags=re.S,
    )
    assert m, "@media (min-width: 900px) desktop block missing"
    block = m.group(1)
    sidebar = re.search(
        r"body\.remote-chat\.remote-desktop #sidebar\s*\{([^}]*)\}", block
    )
    assert sidebar, "docked #sidebar rule missing from desktop block"
    assert "transform: none" in sidebar.group(1)
    assert "width: var(--sidebar-w," in sidebar.group(1), (
        "sidebar width must be driven by the --sidebar-w variable"
    )
    overlay = re.search(
        r"body\.remote-chat\.remote-desktop #sidebar-overlay\s*"
        r"\{([^}]*)\}",
        block,
    )
    assert overlay and "display: none" in overlay.group(1), (
        "the dark overlay must be hidden while docked"
    )
    app = re.search(
        r"body\.remote-chat\.remote-desktop #app\s*\{([^}]*)\}", block
    )
    assert app and "margin-left: var(--sidebar-w," in app.group(1), (
        "#app must clear the docked sidebar via the SAME width variable"
    )


DECOLORIZED_PANEL_SELECTORS = [
    ".tc-h",
    ".tc-h.tc-h-bash",
    ".rc",
    ".rc-h",
    ".rc-h h3",
    ".system-prompt-h",
    ".prompt-h",
    ".think",
    ".think .lbl",
    ".llm-panel",
    ".tr",
    ".tr.note strong",
    ".tr.warn",
    ".merge-info",
    ".merge-info-hdr",
    ".wt-result-ok",
]


@pytest.mark.parametrize("selector", DECOLORIZED_PANEL_SELECTORS)
def test_chat_panel_decolorized(selector: str) -> None:
    """Each colored chat panel gets a neutral remote override rule."""
    _find_rule(_read_codex_css(), selector)


def test_no_decorative_color_tokens_in_codex_css() -> None:
    """remote-codex.css must stay neutral: none of main.css's
    decorative color variables may appear (red stays in main.css for
    errors and is intentionally NOT overridden here)."""
    css = _read_codex_css()
    for token in (
        "var(--orange)",
        "var(--purple)",
        "var(--green)",
        "var(--yellow)",
        "var(--cyan)",
        "var(--accent)",
    ):
        assert token not in css, f"decorative token {token} leaked in"


def test_error_panels_keep_red() -> None:
    """Error affordances stay red for usability: remote-codex.css must
    NOT override .tr.err / .tr .rl.fail / .wt-result-err (comments are
    stripped before scanning for selectors)."""
    css = re.sub(r"/\*.*?\*/", "", _read_codex_css(), flags=re.S)
    assert ".tr.err" not in css
    assert ".rl.fail" not in css
    assert ".wt-result-err" not in css


def test_chat_panels_left_aligned() -> None:
    """Non-task chat panels align LEFT (Codex assistant column) while
    the pinned #task-panel keeps its right-aligned bubble."""
    css = _read_codex_css()
    left = re.search(
        r"body\.remote-chat #output > \*:not\(#welcome\)\s*\{([^}]*)\}",
        css,
    )
    assert left, "left-align rule for #output children missing"
    rule = left.group(1)
    assert "margin-left: 0" in rule
    assert "margin-right: auto" in rule
    assert "max-width" in rule
    # The pinned user bubble stays right-aligned (outside #output).
    task = re.search(r"body\.remote-chat #task-panel\s*\{([^}]*)\}", css)
    assert task and "margin-left: auto" in task.group(1)


# ── Codex flat-thread panels (indistinguishable from Codex) ─────────
#
# The real Codex desktop/app thread is FLATTER than a card layout:
# assistant/tool content sits directly on the page background, tool
# and thinking labels are quiet sentence-case muted-gray rows with a
# chevron, and only monospace output blocks (bash/tool results) get a
# subtle inset #171717 surface.  These tests pin that language so the
# remote webapp thread is visually indistinguishable from Codex.

FLAT_PAGE_BG_SELECTORS = [
    ".tc",
    ".rc",
    ".system-prompt",
    ".prompt",
    ".think",
    ".llm-panel",
    ".merge-info",
]


@pytest.mark.parametrize("selector", FLAT_PAGE_BG_SELECTORS)
def test_chat_panel_sits_on_page_background(selector: str) -> None:
    """Codex thread items are flat: each container panel must use the
    transparent page background, not an elevated #171717 card."""
    rule = _find_rule(_read_codex_css(), selector)
    assert "background: transparent" in rule, (
        f"{selector} must sit flat on the page background like Codex"
    )
    assert "#171717" not in rule


QUIET_HEADER_SELECTORS = [
    ".tc-h",
    ".rc-h",
    ".system-prompt-h",
    ".prompt-h",
    ".llm-panel-hdr",
]


@pytest.mark.parametrize("selector", QUIET_HEADER_SELECTORS)
def test_chat_panel_headers_are_quiet(selector: str) -> None:
    """Codex activity labels are quiet sentence-case muted rows: no
    uppercase transform, no letter-spacing, no filled header strip."""
    rule = _find_rule(_read_codex_css(), selector)
    assert "text-transform: none" in rule, (
        f"{selector} must drop main.css's UPPERCASE transform"
    )
    assert "letter-spacing: normal" in rule, (
        f"{selector} must drop main.css's letter-spacing"
    )
    assert "background: transparent" in rule, (
        f"{selector} must not paint a filled header strip"
    )


def test_think_label_is_quiet() -> None:
    """The Thinking label matches Codex: sentence case, muted gray."""
    rule = _find_rule(_read_codex_css(), ".think .lbl")
    assert "text-transform: none" in rule
    assert "letter-spacing: normal" in rule


def test_mono_output_blocks_keep_inset_surface() -> None:
    """Monospace output (bash stream, tool results) keeps a subtle
    inset #171717 surface at a 12px radius — the one elevated element
    in the Codex thread."""
    css = _read_codex_css()
    for selector in (".bash-panel", ".tr"):
        rule = _find_rule(css, selector)
        assert "#171717" in rule, (
            f"{selector} must keep the inset #171717 mono surface"
        )
        assert "border-radius: 12px" in rule, (
            f"{selector} must use the Codex 12px block radius"
        )


def test_result_card_is_flat_and_quiet() -> None:
    """The final result renders like a plain Codex assistant reply:
    flat container, no green border, quiet plain heading."""
    css = _read_codex_css()
    rc = _find_rule(css, ".rc")
    assert "border: none" in rc or "border-color: transparent" in rc, (
        ".rc must not keep a visible card border"
    )
    h3 = _find_rule(css, ".rc-h h3")
    assert "#ececec" in h3


def test_muted_label_color_tokens_present() -> None:
    """Activity labels use the Codex muted grays."""
    css = _read_codex_css()
    assert "#8e8e8e" in css or "#afafaf" in css


def test_markdown_code_blocks_match_inset_surface() -> None:
    """Fenced code blocks in streamed text (.txt pre) AND rendered
    markdown bodies (.md-body pre) both use the #171717 inset surface
    at the Codex 12px radius."""
    css = _read_codex_css()
    m = re.search(
        r"body\.remote-chat \.txt pre,\s*"
        r"body\.remote-chat \.md-body pre\s*\{([^}]*)\}",
        css,
    )
    assert m, ".txt pre / .md-body pre inset-surface rule missing"
    rule = m.group(1)
    assert "#171717" in rule
    assert "border-radius: 12px" in rule


def test_hljs_background_neutralized_inside_code_blocks() -> None:
    """main.css paints .hljs with `var(--surface2) !important`; the
    remote stylesheet must force it transparent inside pre blocks so
    the #171717 inset surface shows through."""
    css = _read_codex_css()
    m = re.search(
        r"body\.remote-chat \.txt pre code\.hljs,\s*"
        r"body\.remote-chat \.md-body pre code\.hljs\s*\{([^}]*)\}",
        css,
    )
    assert m, "code.hljs transparency override missing"
    assert "background: transparent !important" in m.group(1)


def test_main_js_remote_desktop_wiring() -> None:
    """main.js wires the desktop dock: remote-chat guard FIRST, then a
    typeof-guarded matchMedia('(min-width: 900px)') listener that
    toggles the remote-desktop body class."""
    js = (MEDIA_DIR / "main.js").read_text(encoding="utf-8")
    assert "remote-desktop" in js
    assert "matchMedia" in js
    assert "typeof window.matchMedia === 'function'" in js
    assert "(min-width: 900px)" in js
    guard = js.find(
        "document.body.classList.contains('remote-chat') &&\n"
    )
    if guard == -1:
        guard = js.find(
            "document.body.classList.contains('remote-chat') &&"
        )
    mm = js.find("window.matchMedia('(min-width: 900px)')")
    assert guard != -1 and mm != -1 and guard < mm, (
        "the remote-chat guard must run before matchMedia is queried"
    )


# ── Resizable docked sidebar ─────────────────────────────────────────


def test_sidebar_resizer_in_chat_html() -> None:
    """chat.html ships an accessible resize handle inside #sidebar."""
    html = CHAT_HTML.read_text(encoding="utf-8")
    m = re.search(r"<div id=\"sidebar-resizer\"[^>]*>", html)
    assert m, "#sidebar-resizer handle missing from chat.html"
    tag = m.group(0)
    assert 'role="separator"' in tag
    assert 'aria-orientation="vertical"' in tag
    assert 'tabindex="0"' in tag


def test_sidebar_resizer_css() -> None:
    """The handle is hidden by default and becomes a col-resize strip
    on the docked desktop sidebar; drags disable text selection."""
    css = _read_codex_css()
    hidden = re.search(
        r"body\.remote-chat #sidebar-resizer\s*\{([^}]*)\}", css
    )
    assert hidden and "display: none" in hidden.group(1), (
        "resizer must be hidden outside the desktop dock"
    )
    resizing = re.search(
        r"body\.remote-chat\.sidebar-resizing\s*\{([^}]*)\}", css
    )
    assert resizing and "user-select: none" in resizing.group(1), (
        "text selection must be disabled while dragging"
    )
    m = re.search(
        r"@media \((?:min-width: 900px|width >= 900px)\)\s*\{(.*)\}\s*$",
        css,
        flags=re.S,
    )
    assert m
    desktop = re.search(
        r"body\.remote-chat\.remote-desktop #sidebar-resizer\s*"
        r"\{([^}]*)\}",
        m.group(1),
    )
    assert desktop, "desktop resizer rule missing from the media block"
    rule = desktop.group(1)
    assert "display: block" in rule
    assert "cursor: col-resize" in rule
    assert "touch-action: none" in rule


def test_main_js_sidebar_resize_wiring() -> None:
    """main.js wires the resizer: pointer capture drag, persistence,
    pointercancel handling, all driven through --sidebar-w."""
    js = (MEDIA_DIR / "main.js").read_text(encoding="utf-8")
    assert "sidebar-resizer" in js
    assert "kiss-sidebar-w" in js
    assert "--sidebar-w" in js
    assert "setPointerCapture" in js
    assert "pointercancel" in js


# ── Desktop width defaults (1/4-screen sidebar, 90% chat column) ────


def test_sidebar_defaults_to_quarter_screen() -> None:
    """The docked sidebar (and #app's margin) defaults to 1/4 of the
    browser screen: the --sidebar-w fallback is a 25vw-based clamp()
    bounded by the resize range, identical in BOTH rules."""
    css = _read_codex_css()
    m = re.search(
        r"@media \((?:min-width: 900px|width >= 900px)\)\s*\{(.*)\}\s*$",
        css,
        flags=re.S,
    )
    assert m, "@media (min-width: 900px) desktop block missing"
    block = m.group(1)
    fallback = "var(--sidebar-w, clamp(220px, 25vw, 600px))"
    sidebar = re.search(
        r"body\.remote-chat\.remote-desktop #sidebar\s*\{([^}]*)\}", block
    )
    assert sidebar and f"width: {fallback}" in sidebar.group(1), (
        "docked sidebar must default to 25vw (1/4 screen), clamped"
    )
    app = re.search(
        r"body\.remote-chat\.remote-desktop #app\s*\{([^}]*)\}", block
    )
    assert app and f"margin-left: {fallback}" in app.group(1), (
        "#app must clear the sidebar via the SAME 25vw-based fallback"
    )


def test_chat_column_spans_ninety_percent() -> None:
    """Chat panels, the pinned task panel, and the composer all span
    90% of the chat webview column (no more 768px / 85% / 75% caps)."""
    css = _read_codex_css()
    panels = re.search(
        r"body\.remote-chat #output > \*:not\(#welcome\)\s*\{([^}]*)\}",
        css,
    )
    assert panels and re.search(r"max-width:\s*90%", panels.group(1)), (
        "chat panels must span 90% of the chat webview"
    )
    assert "768px" not in panels.group(1)
    task = re.search(r"body\.remote-chat #task-panel\s*\{([^}]*)\}", css)
    assert task and re.search(r"max-width:\s*90%", task.group(1)), (
        "the fixed task panel must span 90% of the chat webview"
    )
    composer = re.search(
        r"body\.remote-chat #input-container\s*\{([^}]*)\}", css
    )
    assert composer and re.search(
        r"max-width:\s*90%", composer.group(1)
    ), "the composer must span 90% of the chat webview"
    assert "768px" not in composer.group(1)


def test_main_js_quarter_screen_default() -> None:
    """main.js seeds the sidebar resize default from 25% of the window
    width instead of a fixed 300px."""
    js = (MEDIA_DIR / "main.js").read_text(encoding="utf-8")
    assert "window.innerWidth * 0.25" in js, (
        "the resize default must be computed as 1/4 of the window width"
    )


# ── Live HTTP serving ────────────────────────────────────────────────


@pytest.mark.timeout(120)
def test_server_serves_codex_stylesheet_over_http(
    tmp_path: Path,
) -> None:
    """The real ``RemoteAccessServer`` must serve the linked stylesheet.

    This starts production ``RemoteAccessServer.start_async`` with a real
    TLS listener on an ephemeral port, requests both ``/`` and the exact
    cache-busted URL emitted by the page, and shuts the server down through
    its production lifecycle.  Thus the test covers the actual
    ``_process_request`` route rather than a look-alike test server.
    """
    from kiss.server.web_server import MEDIA_DIR as SRV_MEDIA
    from kiss.server.web_server import (
        RemoteAccessServer,
        _generate_self_signed_cert,
    )

    certfile = tmp_path / "cert.pem"
    keyfile = tmp_path / "key.pem"
    _generate_self_signed_cert(certfile, keyfile)
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}

    def run_server() -> None:
        async def scenario() -> None:
            server = RemoteAccessServer(
                host="127.0.0.1",
                port=0,
                work_dir=str(tmp_path),
                certfile=str(certfile),
                keyfile=str(keyfile),
                url_file=tmp_path / "remote-url.json",
                uds_path=tmp_path / "sorcar.sock",
            )
            started = False
            try:
                await server.start_async()
                started = True
                assert server._ws_server is not None
                state["port"] = next(
                    iter(server._ws_server.sockets)
                ).getsockname()[1]
                ready.set()
                while not done.is_set():
                    await asyncio.sleep(0.02)
            except BaseException as exc:
                state["error"] = exc
                ready.set()
            finally:
                if started:
                    await server.stop_async()

        asyncio.run(scenario())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    assert ready.wait(30), "RemoteAccessServer failed to start"
    startup_error = state.get("error")
    if isinstance(startup_error, BaseException):
        raise AssertionError(
            "RemoteAccessServer startup failed"
        ) from startup_error
    port = state["port"]
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    def fetch(url: str) -> tuple[int, str, bytes]:
        with urllib.request.urlopen(
            url, timeout=10, context=context
        ) as response:
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                response.read(),
            )

    try:
        status, content_type, page_bytes = fetch(
            f"https://127.0.0.1:{port}/"
        )
        assert status == 200
        assert "text/html" in content_type
        page = page_bytes.decode("utf-8")
        m = re.search(
            r'href="(/media/remote-codex\.css\?v=[0-9a-f]{16})"', page
        )
        assert m, "served page must link remote-codex.css"

        status, content_type, body = fetch(
            f"https://127.0.0.1:{port}{m.group(1)}"
        )
        assert status == 200
        assert "text/css" in content_type
        assert body == (SRV_MEDIA / "remote-codex.css").read_bytes()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"
    thread_error = state.get("error")
    if isinstance(thread_error, BaseException):
        raise AssertionError(
            "RemoteAccessServer thread failed"
        ) from thread_error


def test_media_url_cache_busts_codex_css() -> None:
    """_media_url must hash remote-codex.css for cache busting."""
    import hashlib

    from kiss.server.web_server import (
        _MEDIA_VERSION_CACHE,
        _media_url,
    )

    _MEDIA_VERSION_CACHE.pop("remote-codex.css", None)
    url = _media_url("remote-codex.css")
    expected = hashlib.sha256(CODEX_CSS.read_bytes()).hexdigest()[:16]
    assert url == f"/media/remote-codex.css?v={expected}"
    # And cached on second call.
    assert _media_url("remote-codex.css") == url
