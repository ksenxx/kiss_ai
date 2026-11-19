"""The remote webapp paints with Claude Code Desktop's dark / light
palette and the Claude apps' fallback font stacks only.

``web_server.py`` inlines the two palettes as ``--theme-*`` variables
(dark on ``:root``, light on ``html.light-theme,
body.remote-chat.light-theme``) together with the font stacks;
``media/remote-codex.css`` maps every semantic colour name onto
those variables, with the dark palette's value as the fallback; the
code blocks take ``highlight-dark.css`` / ``highlight-light.css``
(the app's five syntax hues) instead of GitHub's themes.

The static tests pin those invariants from the sources; the live test
serves the page through the production ``RemoteAccessServer`` into a
real Chromium and reads the computed colours and fonts in both themes.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from kiss.server.web_server import (
    _DARK_THEME_CSS,
    _LIGHT_THEME_CSS,
    _app_shell_urls,
    _brand_css_asset_names,
    _build_html,
    _build_share_page,
)
from kiss.tests.conftest import goto_retrying_network_change
from kiss.tests.webapp.test_codex_task_panel_style import (
    _start_live_server,
)

MEDIA_DIR = Path(__file__).resolve().parents[2] / "webapp" / "media"
CODEX_CSS = MEDIA_DIR / "remote-codex.css"

COLOR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)")
VAR_DECL = re.compile(r"(--theme-[\w-]+):\s*([^;]+);")

# The colours of each highlight sheet: the theme's body colour, its five
# syntax hues (keyword, function / type, number, string, comment) and
# the diff green and red.
DARK_TOKEN_COLOURS = {
    "#f0efec", "#ff8d77", "#c8a6ff", "#e3c07b", "#e5b07e", "#8b8b81",
    "#4cc38a", "#ff5a5a",
}
LIGHT_TOKEN_COLOURS = {
    "#0b0b0b", "#c2452d", "#6f42c1", "#9a6700", "#9c4a21", "#6e7781",
    "#2d8a4e", "#e5484d",
}
# highlight.js scope class -> the colour of the matching syntax role.
DARK_SCOPE_COLOURS = {
    ".hljs-keyword": "#ff8d77",          # keyword
    ".hljs-title.function_": "#c8a6ff",  # function name
    ".hljs-title.class_": "#c8a6ff",     # type name
    ".hljs-variable": "#f0efec",         # variable: body colour
    ".hljs-attr": "#f0efec",             # attribute name: body colour
    ".hljs-number": "#e3c07b",           # number
    ".hljs-string": "#e5b07e",           # string
    ".hljs-regexp": "#e5b07e",           # regular expression
    ".hljs-comment": "#8b8b81",          # comment
    ".hljs-addition": "#4cc38a",         # inserted line
    ".hljs-deletion": "#ff5a5a",         # deleted line
}
LIGHT_SCOPE_COLOURS = {
    ".hljs-keyword": "#c2452d",
    ".hljs-title.function_": "#6f42c1",
    ".hljs-title.class_": "#6f42c1",
    ".hljs-variable": "#0b0b0b",
    ".hljs-attr": "#0b0b0b",
    ".hljs-number": "#9a6700",
    ".hljs-string": "#9c4a21",
    ".hljs-regexp": "#9c4a21",
    ".hljs-comment": "#6e7781",
    ".hljs-addition": "#2d8a4e",
    ".hljs-deletion": "#e5484d",
}


def _norm(colour: str) -> str:
    """Lower-case *colour* and expand a 3/4-digit hex (stylelint's
    short form, e.g. ``#ccc``) to its 6/8-digit spelling."""
    colour = colour.strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3,4}", colour):
        colour = "#" + "".join(ch * 2 for ch in colour[1:])
    return colour


def _stripped_codex_css() -> str:
    css = CODEX_CSS.read_text(encoding="utf-8")
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _palette(block: str) -> dict[str, str]:
    return dict(VAR_DECL.findall(block))


def test_dark_and_light_palettes_declare_the_same_variables() -> None:
    """One line per token in each theme block, in the same order, so a
    token can never be themed in dark but fall through in light."""
    dark = [name for name, _ in VAR_DECL.findall(_DARK_THEME_CSS)]
    light = [name for name, _ in VAR_DECL.findall(_LIGHT_THEME_CSS)]
    assert dark == light
    assert len(dark) == len(set(dark)), "duplicate variable in the dark block"
    assert len(dark) >= 60


def test_remote_css_colour_literals_are_dark_palette_fallbacks() -> None:
    """Every colour literal in remote-codex.css is the fallback of a
    ``var(--theme-…, <literal>)`` reference, and equals the dark
    palette's value web_server.py injects for that variable."""
    css = _stripped_codex_css()
    dark = _palette(_DARK_THEME_CSS)
    refs = re.findall(r"var\((--theme-[\w-]+),\s*([^)]+)\)", css)
    assert refs, "remote-codex.css must map its palette onto --theme-* variables"
    for name, fallback in refs:
        assert name in dark, f"{name} is not injected by web_server.py"
        assert _norm(fallback) == _norm(dark[name]), (
            f"{name} fallback {fallback!r} differs from the dark palette's {dark[name]!r}"
        )
    without_refs = re.sub(r"var\(--theme-[\w-]+,\s*[^)]+\)", "", css)
    leaked = COLOR_LITERAL.findall(without_refs)
    assert not leaked, f"colour literals outside the theme mapping: {leaked}"


def test_theme_dependent_root_tokens_are_rederived_on_the_remote_body() -> None:
    """A main.css ``:root`` token that reads a ``--theme-*`` variable
    resolves on the root, before ``body.remote-chat.light-theme`` swaps
    the variable, so it would keep its dark value in the light theme
    unless ``body.remote-chat`` declares the token again (e.g.
    ``--hover`` on the content-tab menu hover)."""
    main_css = re.sub(
        r"/\*.*?\*/", "", (MEDIA_DIR / "main.css").read_text(encoding="utf-8"), flags=re.S
    )
    root_block = re.search(r":root\s*\{(.*?)\n\}", main_css, re.S)
    body_block = re.search(r"body\.remote-chat\s*\{(.*?)\n\}", _stripped_codex_css(), re.S)
    assert root_block and body_block, "token blocks not found"
    root, body = root_block.group(1), body_block.group(1)
    # Only variables the light theme redefines matter: the font
    # variables (_FONT_VARS_CSS) are injected once for both
    # themes, so --font-ui / --font-mono resolve the same either way.
    themed = set(_palette(_LIGHT_THEME_CSS))
    theme_dependent = [
        name
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root)
        if themed & set(re.findall(r"--theme-[\w-]+", value))
    ]
    assert theme_dependent, "main.css :root must derive tokens from --theme-* variables"
    rederived = set(re.findall(r"(--[\w-]+)\s*:", body))
    missing = [name for name in theme_dependent if name not in rederived]
    assert not missing, f"not re-derived on body.remote-chat: {missing}"


def test_every_theme_variable_the_page_uses_is_injected() -> None:
    """main.css, remote-codex.css and main.js may only reference
    ``--theme-*`` names that both theme blocks define (plus the font
    variables, which are theme-independent)."""
    dark = set(_palette(_DARK_THEME_CSS))
    fonts = {
        "--theme-font-size", "--theme-font-family", "--theme-font-weight",
        "--theme-editor-font-family", "--theme-editor-font-weight",
    }
    used: set[str] = set()
    for name in ("main.css", "remote-codex.css", "main.js"):
        text = (MEDIA_DIR / name).read_text(encoding="utf-8")
        used.update(re.findall(r"--theme-[A-Za-z-]+", text))
    missing = sorted(used - dark - fonts)
    assert not missing, f"referenced but not injected: {missing}"


def test_remote_page_inlines_fonts_and_both_palettes() -> None:
    """The built remote page carries the Claude apps' fallback font
    stacks at 13px, the dark palette's ``:root`` block and the light
    block keyed on ``body.remote-chat.light-theme``."""
    page = _build_html()
    assert (
        "--theme-font-family: system-ui, -apple-system, "
        '"Segoe UI", Roboto, Helvetica, Arial, sans-serif;'
    ) in page
    assert (
        "--theme-editor-font-family: ui-monospace, SFMono-Regular, "
        'Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", '
        "monospace;"
    ) in page
    assert "--theme-font-size: 13px;" in page
    # One size variable: nothing reads a separate editor size.
    assert "--theme-editor-font-size" not in page
    assert "html.light-theme,\n    body.remote-chat.light-theme {" in page
    root_pos = page.index(":root {")
    light_pos = page.index("body.remote-chat.light-theme {")
    assert root_pos < light_pos, "the light block must follow :root to win"
    assert page.count("--theme-editor-background: #0b0b0b;") == 1
    # The light palette paints an off-white canvas and its
    # full-contrast variant a white one; each block is inlined once.
    assert page.count("--theme-editor-background: #fafafa;") == 1
    assert page.count("--theme-editor-background: #ffffff;") == 1
    assert page.count("body.remote-chat.light-theme {") == 1
    assert page.count("body.remote-chat.light-theme.high-contrast {") == 1
    assert "highlight-github" not in page
    # Light is the default theme, so the page ships light-themed markup
    # and main.js only has to swap in the dark sheet for dark-savers.
    assert '<body class="remote-chat light-theme">' in page
    assert re.search(r'id="hljs-theme" href="/media/highlight-light\.css\?v=', page)
    assert '"dark": "/media/highlight-dark.css?v=' in page


def test_app_shell_precaches_both_highlight_sheets() -> None:
    """The service worker's app shell includes the light sheet main.js
    swaps in, so the theme toggle works offline too."""
    urls = _app_shell_urls(_brand_css_asset_names())
    assert any(u.startswith("/media/highlight-dark.css?v=") for u in urls)
    assert any(u.startswith("/media/highlight-light.css?v=") for u in urls)
    assert not any("highlight-github" in u for u in urls)


def _scope_colour(css: str, cls: str) -> str:
    """The ``color`` the LAST rule listing *cls* as a whole selector
    declares (later rules win at equal specificity), 6-digit lower-case."""
    colour = None
    for header, body in re.findall(r"([^{}]+)\{([^}]*)\}", css):
        selectors = [sel.strip() for sel in header.split(",")]
        m = re.search(r"(?<![\w-])color:\s*(#[0-9a-fA-F]{3,8})", body)
        if cls in selectors and m:
            colour = _norm(m.group(1))
    assert colour, f"{cls} has no colour rule"
    return colour


@pytest.mark.parametrize(
    ("name", "allowed", "scopes"),
    [
        ("highlight-dark.css", DARK_TOKEN_COLOURS, DARK_SCOPE_COLOURS),
        ("highlight-light.css", LIGHT_TOKEN_COLOURS, LIGHT_SCOPE_COLOURS),
    ],
)
def test_highlight_sheets_map_scopes_to_the_syntax_hues(
    name: str, allowed: set[str], scopes: dict[str, str]
) -> None:
    """Every colour in the sheet is one of the theme's syntax hues,
    and each highlight.js scope class carries the hue of its role."""
    css = re.sub(r"/\*.*?\*/", "", (MEDIA_DIR / name).read_text(encoding="utf-8"), flags=re.S)
    colours = {_norm(c) for c in COLOR_LITERAL.findall(css)}
    assert colours, f"{name} defines no colours"
    assert colours <= allowed, f"{name}: colours off the palette {sorted(colours - allowed)}"
    for cls, expected in scopes.items():
        assert _scope_colour(css, cls) == expected, f"{name}: {cls}"


def test_share_page_inlines_both_highlight_sheets() -> None:
    page = _build_share_page("t", "<div></div>")
    assert ".hljs { color: #f0efec; }" in page
    assert ".hljs { color: #0b0b0b; }" in page
    assert "highlight-github" not in page


_PROBE_JS = """() => {
  const cs = (sel, prop) => getComputedStyle(document.querySelector(sel))[prop];
  return {
    light: document.body.classList.contains('light-theme'),
    bodyBg: cs('body', 'backgroundColor'),
    bodyFg: cs('body', 'color'),
    fontFamily: cs('body', 'fontFamily'),
    fontSize: cs('body', 'fontSize'),
    inputFont: cs('#task-input', 'fontFamily'),
    codeFont: cs('#probe-code', 'fontFamily'),
    preBg: cs('#probe-pre', 'backgroundColor'),
    codeBg: cs('#probe-code', 'backgroundColor'),
    keyword: cs('#probe-code .hljs-keyword', 'color'),
    // The content pane's tab row (desktop split layout) carries the
    // editor-group header colour, which both palettes set to the canvas.
    tabBar: cs('#content-tab-bar', 'backgroundColor'),
    composer: cs('#input-container', 'backgroundColor'),
    send: cs('#send-btn', 'backgroundColor'),
    sendFg: cs('#send-btn', 'color'),
    sidebar: cs('#sidebar', 'backgroundColor'),
    link: cs('#probe-link', 'color'),
    hljsHref: document.getElementById('hljs-theme').getAttribute('href'),
  };
}"""

_INJECT_JS = """() => {
  document.getElementById('auth-modal').style.display = 'none';
  const out = document.getElementById('output');
  const div = document.createElement('div');
  div.className = 'msg assistant';
  // The production class combination: main.js renders assistant text
  // as .txt and adds .md-body after the Markdown pass.
  div.innerHTML = '<div class="txt md-body"><p><a id="probe-link" href="#">link</a></p>'
    + '<pre id="probe-pre"><code id="probe-code" class="hljs language-python">'
    + '<span class="hljs-keyword">def</span> f(): <span class="hljs-string">"s"</span>'
    + '</code></pre></div>';
  out.appendChild(div);
  return true;
}"""

# The font stacks web_server.py injects (_FONT_VARS_CSS): the Claude
# apps' fallbacks, the system sans for the UI and the system monospace
# for code.
UI_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
MONO_FONT = (
    "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
    '"Liberation Mono", "Courier New", monospace'
)

# True once the theme's colour transitions (the Send button's fill among
# them) have run to their end.
_SETTLED_JS = "!document.getAnimations().some(a => a instanceof CSSTransition)"


def _font_families(stack: str) -> list[str]:
    """Split a ``font-family`` value into comparable family names.

    Computed styles come back re-serialised by the engine, with the
    quotes normalised.  Comparing the lists of names keeps the test
    about which families are requested, not about how Chromium spells
    them.
    """
    return [name.strip().strip('"') for name in stack.split(",")]


@pytest.mark.timeout(180)
def test_live_remote_page_uses_theme_colours_and_fonts(tmp_path: Path) -> None:
    """Served page + real Chromium: the light palette by default, the
    dark palette after the theme toggle, the injected font stacks
    throughout, and the code block recoloured by the matching highlight
    sheet."""
    ready = threading.Event()
    done = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(
        target=_start_live_server, args=(tmp_path, ready, done, state), daemon=True
    )
    thread.start()
    try:
        assert ready.wait(30), "RemoteAccessServer failed to start"
        startup_error = state.get("error")
        if isinstance(startup_error, BaseException):
            raise AssertionError("RemoteAccessServer startup failed") from startup_error
        port = state["port"]
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--ignore-certificate-errors"])
            try:
                page = browser.new_page(
                    ignore_https_errors=True, viewport={"width": 1400, "height": 900}
                )
                goto_retrying_network_change(page, f"https://127.0.0.1:{port}/", wait_until="load")
                page.wait_for_selector("#output", state="attached")
                page.evaluate(_INJECT_JS)
                # The theme sheet (code block background), the colour
                # transitions of the page's controls (#send-btn's fill among
                # them) and the highlight.js sheet, a separate <link> whose
                # href is swapped per theme (keyword colour), all settle
                # independently: wait for every one of them before probing.
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('probe-pre'))"
                    ".backgroundColor === 'rgb(244, 244, 242)' && "
                    "getComputedStyle(document.querySelector('#probe-code .hljs-keyword'))"
                    ".color === 'rgb(194, 69, 45)' && " + _SETTLED_JS,
                    timeout=10000,
                )
                light = page.evaluate(_PROBE_JS)
                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('probe-pre'))"
                    ".backgroundColor === 'rgb(21, 21, 21)' && "
                    "getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(239, 239, 239)' && "
                    "getComputedStyle(document.querySelector('#probe-code .hljs-keyword'))"
                    ".color === 'rgb(255, 141, 119)' && " + _SETTLED_JS,
                    timeout=10000,
                )
                dark = page.evaluate(_PROBE_JS)
                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => document.body.classList.contains('light-theme') && "
                    "getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(25, 25, 25)' && "
                    "getComputedStyle(document.querySelector('#probe-code .hljs-keyword'))"
                    ".color === 'rgb(194, 69, 45)' && " + _SETTLED_JS,
                    timeout=10000,
                )
                back = page.evaluate(_PROBE_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    # The dark palette: editor.background #0b0b0b, editor.foreground
    # #f0efec, editorGroupHeader.tabsBackground #0b0b0b, input.background
    # #1f1f1f, button.background #efefef with button.foreground #0b0b0b,
    # sideBar.background #151515, textLink.foreground #4d9fff, keyword
    # token #ff8d77, textCodeBlock.background #151515.
    assert dark["light"] is False, dark
    assert dark["bodyBg"] == "rgb(11, 11, 11)", dark
    assert dark["bodyFg"] == "rgb(240, 239, 236)", dark
    assert dark["tabBar"] == "rgb(11, 11, 11)", dark
    assert dark["composer"] == "rgb(31, 31, 31)", dark
    assert dark["send"] == "rgb(239, 239, 239)", dark
    assert dark["sendFg"] == "rgb(11, 11, 11)", dark
    assert dark["sidebar"] == "rgb(21, 21, 21)", dark
    assert dark["link"] == "rgb(77, 159, 255)", dark
    assert dark["keyword"] == "rgb(255, 141, 119)", dark
    assert dark["preBg"] == "rgb(21, 21, 21)", dark
    assert "highlight-dark.css?v=" in dark["hljsHref"], dark

    # The light palette: #fafafa / #0b0b0b, tabs #fafafa, input #ffffff,
    # button #191919 with #ffffff, sideBar #f5f5f3, link #1f6fe5,
    # keyword #c2452d, code block #f4f4f2.
    assert light["light"] is True, light
    assert light["bodyBg"] == "rgb(250, 250, 250)", light
    assert light["bodyFg"] == "rgb(11, 11, 11)", light
    assert light["tabBar"] == "rgb(250, 250, 250)", light
    assert light["composer"] == "rgb(255, 255, 255)", light
    assert light["send"] == "rgb(25, 25, 25)", light
    assert light["sendFg"] == "rgb(255, 255, 255)", light
    assert light["sidebar"] == "rgb(245, 245, 243)", light
    assert light["link"] == "rgb(31, 111, 229)", light
    assert light["keyword"] == "rgb(194, 69, 45)", light
    assert light["preBg"] == "rgb(244, 244, 242)", light
    assert "highlight-light.css?v=" in light["hljsHref"], light

    # The Markdown <pre> paints the code block's surface; the highlighted
    # <code> inside it adds no background of its own in either theme.
    assert dark["codeBg"] == light["codeBg"] == "rgba(0, 0, 0, 0)", (dark, light)

    # Fonts do not depend on the theme: the system sans stack at 13px
    # for the UI and the composer, the system monospace stack for code.
    for probe in (dark, light):
        assert _font_families(probe["fontFamily"]) == _font_families(UI_FONT), probe
        assert _font_families(probe["inputFont"]) == _font_families(UI_FONT), probe
        assert probe["fontSize"] == "13px", probe
        assert _font_families(probe["codeFont"]) == _font_families(MONO_FONT), probe

    # Toggling back restores the light palette exactly.
    assert back["light"] is True, back
    assert back["bodyBg"] == light["bodyBg"], back
    assert back["send"] == light["send"], back
    assert back["hljsHref"] == light["hljsHref"], back
