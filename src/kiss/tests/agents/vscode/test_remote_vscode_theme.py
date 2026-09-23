"""The remote webapp paints with VS Code's Dark Modern / Light Modern
theme colours and VS Code's fonts only.

``web_server.py`` inlines the two palettes as ``--vscode-*`` variables
(Dark Modern on ``:root``, Light Modern on ``html.light-theme,
body.remote-chat.light-theme``) together with VS Code's default font
stacks; ``media/remote-codex.css`` maps every semantic colour name onto
those variables, with the Dark Modern value as the fallback; the code
blocks take ``highlight-vscode-dark.css`` / ``highlight-vscode-light.css``
(VS Code's Dark+ / Light+ token colours) instead of GitHub's themes.

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

import kiss.agents.vscode as vscode_pkg
from kiss.server.web_server import (
    _VSCODE_DARK_MODERN_CSS,
    _VSCODE_LIGHT_MODERN_CSS,
    _app_shell_urls,
    _build_html,
    _build_share_page,
)
from kiss.tests.agents.vscode.test_codex_task_panel_style import (
    _start_live_server,
)

MEDIA_DIR = Path(vscode_pkg.__file__).parent / "media"
CODEX_CSS = MEDIA_DIR / "remote-codex.css"

COLOR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)")
VAR_DECL = re.compile(r"(--vscode-[\w-]+):\s*([^;]+);")

# Token colours of Dark Modern (dark_modern.json -> dark_plus.json ->
# dark_vs.json) and Light Modern (light_modern.json -> light_plus.json
# -> light_vs.json), plus each theme's textCodeBlock.background.
DARK_TOKEN_COLOURS = {
    "#cccccc", "#2b2b2b", "#569cd6", "#c586c0", "#dcdcaa", "#4ec9b0",
    "#9cdcfe", "#b5cea8", "#ce9178", "#d16969", "#4fc1ff", "#6a9955",
    "#d7ba7d", "#d4d4d4", "#6796e6",
}
LIGHT_TOKEN_COLOURS = {
    "#3b3b3b", "#f8f8f8", "#0000ff", "#af00db", "#795e26", "#267f99",
    "#001080", "#098658", "#a31515", "#ee0000", "#811f3f", "#0070c1",
    "#008000", "#800000", "#000000", "#0451a5", "#800080", "#000080",
    "#e50000",
}
# highlight.js scope class -> the VS Code TextMate scope's colour.
DARK_SCOPE_COLOURS = {
    ".hljs-keyword": "#569cd6",          # keyword
    ".hljs-keyword.control_": "#c586c0", # keyword.control
    ".hljs-title.function_": "#dcdcaa",  # entity.name.function
    ".hljs-title.class_": "#4ec9b0",     # entity.name.type
    ".hljs-variable": "#9cdcfe",         # variable
    ".hljs-attr": "#9cdcfe",             # entity.other.attribute-name
    ".hljs-number": "#b5cea8",           # constant.numeric
    ".hljs-string": "#ce9178",           # string
    ".hljs-char.escape_": "#d7ba7d",     # constant.character.escape
    ".hljs-regexp": "#d16969",           # string.regexp
    ".hljs-comment": "#6a9955",          # comment
    ".hljs-addition": "#b5cea8",         # markup.inserted
    ".hljs-deletion": "#ce9178",         # markup.deleted
}
LIGHT_SCOPE_COLOURS = {
    ".hljs-keyword": "#0000ff",
    ".hljs-keyword.control_": "#af00db",
    ".hljs-title.function_": "#795e26",
    ".hljs-title.class_": "#267f99",
    ".hljs-variable": "#001080",
    ".hljs-attr": "#e50000",
    ".hljs-number": "#098658",
    ".hljs-string": "#a31515",
    ".hljs-char.escape_": "#ee0000",
    ".hljs-regexp": "#811f3f",
    ".hljs-comment": "#008000",
    ".hljs-addition": "#098658",
    ".hljs-deletion": "#a31515",
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
    dark = [name for name, _ in VAR_DECL.findall(_VSCODE_DARK_MODERN_CSS)]
    light = [name for name, _ in VAR_DECL.findall(_VSCODE_LIGHT_MODERN_CSS)]
    assert dark == light
    assert len(dark) == len(set(dark)), "duplicate variable in the dark block"
    assert len(dark) >= 60


def test_remote_css_colour_literals_are_dark_modern_fallbacks() -> None:
    """Every colour literal in remote-codex.css is the fallback of a
    ``var(--vscode-…, <literal>)`` reference, and equals the Dark
    Modern value web_server.py injects for that variable."""
    css = _stripped_codex_css()
    dark = _palette(_VSCODE_DARK_MODERN_CSS)
    refs = re.findall(r"var\((--vscode-[\w-]+),\s*([^)]+)\)", css)
    assert refs, "remote-codex.css must map its palette onto --vscode-* variables"
    for name, fallback in refs:
        assert name in dark, f"{name} is not injected by web_server.py"
        assert _norm(fallback) == _norm(dark[name]), (
            f"{name} fallback {fallback!r} differs from Dark Modern {dark[name]!r}"
        )
    without_refs = re.sub(r"var\(--vscode-[\w-]+,\s*[^)]+\)", "", css)
    leaked = COLOR_LITERAL.findall(without_refs)
    assert not leaked, f"colour literals outside the theme mapping: {leaked}"


def test_every_vscode_variable_the_page_uses_is_injected() -> None:
    """main.css, remote-codex.css and main.js may only reference
    ``--vscode-*`` names that both theme blocks define (plus the font
    variables, which are theme-independent)."""
    dark = set(_palette(_VSCODE_DARK_MODERN_CSS))
    fonts = {
        "--vscode-font-size", "--vscode-font-family", "--vscode-font-weight",
        "--vscode-editor-font-size", "--vscode-editor-font-family",
        "--vscode-editor-font-weight",
    }
    used: set[str] = set()
    for name in ("main.css", "remote-codex.css", "main.js"):
        text = (MEDIA_DIR / name).read_text(encoding="utf-8")
        used.update(re.findall(r"--vscode-[A-Za-z-]+", text))
    missing = sorted(used - dark - fonts)
    assert not missing, f"referenced but not injected: {missing}"


def test_remote_page_inlines_vscode_fonts_and_both_palettes() -> None:
    """The built remote page carries VS Code's font stacks at the
    editor font size, the Dark Modern ``:root`` block and the Light
    Modern block keyed on ``body.remote-chat.light-theme``."""
    page = _build_html()
    assert (
        '--vscode-font-family: -apple-system, BlinkMacSystemFont, '
        '"Segoe WPC", "Segoe UI", system-ui, "Ubuntu", "Droid Sans", '
        "sans-serif;"
    ) in page
    assert (
        '--vscode-editor-font-family: Menlo, Monaco, Consolas, '
        '"Droid Sans Mono", "Courier New", monospace;'
    ) in page
    assert "--vscode-font-size: 14px;" in page
    assert "--vscode-editor-font-size: 14px;" in page
    assert "html.light-theme,\n    body.remote-chat.light-theme {" in page
    root_pos = page.index(":root {")
    light_pos = page.index("body.remote-chat.light-theme {")
    assert root_pos < light_pos, "the light block must follow :root to win"
    assert page.count("--vscode-editor-background: #1f1f1f;") == 1
    assert page.count("--vscode-editor-background: #ffffff;") == 1
    assert "highlight-github" not in page
    assert re.search(r'id="hljs-theme" href="/media/highlight-vscode-dark\.css\?v=', page)
    assert '"light": "/media/highlight-vscode-light.css?v=' in page


def test_app_shell_precaches_both_highlight_sheets() -> None:
    """The service worker's app shell includes the light sheet main.js
    swaps in, so the theme toggle works offline too."""
    urls = _app_shell_urls()
    assert any(u.startswith("/media/highlight-vscode-dark.css?v=") for u in urls)
    assert any(u.startswith("/media/highlight-vscode-light.css?v=") for u in urls)
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
        ("highlight-vscode-dark.css", DARK_TOKEN_COLOURS, DARK_SCOPE_COLOURS),
        ("highlight-vscode-light.css", LIGHT_TOKEN_COLOURS, LIGHT_SCOPE_COLOURS),
    ],
)
def test_highlight_sheets_map_scopes_to_vscode_token_colours(
    name: str, allowed: set[str], scopes: dict[str, str]
) -> None:
    """Every colour in the sheet is one of the theme's token colours,
    and each highlight.js scope class carries the colour VS Code gives
    the matching TextMate scope."""
    css = re.sub(r"/\*.*?\*/", "", (MEDIA_DIR / name).read_text(encoding="utf-8"), flags=re.S)
    colours = {_norm(c) for c in COLOR_LITERAL.findall(css)}
    assert colours, f"{name} defines no colours"
    assert colours <= allowed, f"non-VS-Code colours in {name}: {sorted(colours - allowed)}"
    for cls, expected in scopes.items():
        assert _scope_colour(css, cls) == expected, f"{name}: {cls}"


def test_share_page_inlines_the_vscode_highlight_sheets() -> None:
    page = _build_share_page("t", "<div></div>")
    assert ".hljs { color: #ccc; background: #2b2b2b; }" in page
    assert ".hljs { color: #3b3b3b; background: #f8f8f8; }" in page
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
    tabBar: cs('#tab-bar', 'backgroundColor'),
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

VSCODE_UI_FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", '
    'system-ui, Ubuntu, "Droid Sans", sans-serif'
)
VSCODE_EDITOR_FONT = 'Menlo, Monaco, Consolas, "Droid Sans Mono", "Courier New", monospace'


@pytest.mark.timeout(180)
def test_live_remote_page_uses_vscode_theme_colours_and_fonts(tmp_path: Path) -> None:
    """Served page + real Chromium: Dark Modern by default, Light
    Modern after the theme toggle, VS Code's font stacks throughout,
    and the code block restyled by the matching highlight sheet."""
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
                page.goto(f"https://127.0.0.1:{port}/", wait_until="load")
                page.wait_for_selector("#output", state="attached")
                page.evaluate(_INJECT_JS)
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('probe-code'))"
                    ".backgroundColor === 'rgb(43, 43, 43)'",
                    timeout=10000,
                )
                dark = page.evaluate(_PROBE_JS)
                page.evaluate("() => document.getElementById('theme-btn').click()")
                # The light sheet has to load and #send-btn's 0.2s
                # colour transition (main.css) has to settle.
                page.wait_for_function(
                    "() => getComputedStyle(document.getElementById('probe-code'))"
                    ".backgroundColor === 'rgb(248, 248, 248)' && "
                    "getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(0, 95, 184)'",
                    timeout=10000,
                )
                light = page.evaluate(_PROBE_JS)
                page.evaluate("() => document.getElementById('theme-btn').click()")
                page.wait_for_function(
                    "() => !document.body.classList.contains('light-theme') && "
                    "getComputedStyle(document.getElementById('send-btn'))"
                    ".backgroundColor === 'rgb(0, 120, 212)'",
                    timeout=10000,
                )
                back = page.evaluate(_PROBE_JS)
            finally:
                browser.close()
    finally:
        done.set()
        thread.join(timeout=30)
    assert not thread.is_alive(), "RemoteAccessServer failed to stop"

    # Dark Modern: editor.background #1f1f1f, editor.foreground #cccccc,
    # editorGroupHeader.tabsBackground #2b2b2b, input.background #313131,
    # button.background #0078d4, sideBar.background #181818,
    # textLink.foreground #4daafc, keyword token #569cd6.
    assert dark["light"] is False, dark
    assert dark["bodyBg"] == "rgb(31, 31, 31)", dark
    assert dark["bodyFg"] == "rgb(204, 204, 204)", dark
    assert dark["tabBar"] == "rgb(43, 43, 43)", dark
    assert dark["composer"] == "rgb(49, 49, 49)", dark
    assert dark["send"] == "rgb(0, 120, 212)", dark
    assert dark["sendFg"] == "rgb(255, 255, 255)", dark
    assert dark["sidebar"] == "rgb(24, 24, 24)", dark
    assert dark["link"] == "rgb(77, 170, 252)", dark
    assert dark["keyword"] == "rgb(86, 156, 214)", dark
    assert dark["preBg"] == dark["codeBg"] == "rgb(43, 43, 43)", dark
    assert "highlight-vscode-dark.css?v=" in dark["hljsHref"], dark

    # Light Modern: #ffffff / #3b3b3b, tabs #e5e5e5, input #ffffff,
    # button #005fb8, sideBar #f8f8f8, link #005fb8, keyword #0000ff.
    assert light["light"] is True, light
    assert light["bodyBg"] == "rgb(255, 255, 255)", light
    assert light["bodyFg"] == "rgb(59, 59, 59)", light
    assert light["tabBar"] == "rgb(229, 229, 229)", light
    assert light["composer"] == "rgb(255, 255, 255)", light
    assert light["send"] == "rgb(0, 95, 184)", light
    assert light["sendFg"] == "rgb(255, 255, 255)", light
    assert light["sidebar"] == "rgb(248, 248, 248)", light
    assert light["link"] == "rgb(0, 95, 184)", light
    assert light["keyword"] == "rgb(0, 0, 255)", light
    assert light["preBg"] == light["codeBg"] == "rgb(248, 248, 248)", light
    assert "highlight-vscode-light.css?v=" in light["hljsHref"], light

    # Fonts do not depend on the theme: VS Code's workbench stack at
    # 14px for the UI and the composer, the editor stack for code.
    for probe in (dark, light):
        assert probe["fontFamily"] == VSCODE_UI_FONT, probe
        assert probe["inputFont"] == VSCODE_UI_FONT, probe
        assert probe["fontSize"] == "14px", probe
        assert probe["codeFont"] == VSCODE_EDITOR_FONT, probe

    # Toggling back restores Dark Modern exactly.
    assert back["light"] is False, back
    assert back["bodyBg"] == dark["bodyBg"], back
    assert back["send"] == dark["send"], back
    assert back["hljsHref"] == dark["hljsHref"], back
