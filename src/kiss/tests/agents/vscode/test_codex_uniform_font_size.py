# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests: result-card status lines use classes, not inline styles.

Historic context: the remote webapp once repinned every chat panel's
body text to one uniform ``--fs-base`` size in ``remote-codex.css``.
That restyle is gone — the remote page now inherits the extension's
main.css type scale verbatim (pinned end to end by
``test_remote_panels_match_extension.py``).  What must NOT regress is
the class-based status-line contract that made any stylesheet control
possible in the first place:

* ``main.js``'s ``createResultPanel`` must not emit inline
  ``font-size`` styles (inline styles beat every stylesheet); it uses
  ``.rc-status`` / ``.rc-status-fail`` classes instead.
* ``main.css`` styles ``.rc-status`` exactly like the old inline
  declarations so the VS Code webview keeps its former look.
* ``remote-codex.css`` must not re-introduce panel body-text size
  overrides (no ``--fs-base`` repins, no hard-coded px outliers).
"""

from __future__ import annotations

import re
from pathlib import Path

MEDIA_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "media"
)
CODEX_CSS = MEDIA_DIR / "remote-codex.css"
MAIN_CSS = MEDIA_DIR / "main.css"
MAIN_JS = MEDIA_DIR / "main.js"


def _read_codex_css() -> str:
    return CODEX_CSS.read_text(encoding="utf-8")


def test_no_panel_body_text_size_overrides_remain() -> None:
    """remote-codex.css must not repin any panel body-text size: the
    remote page inherits the extension's main.css type scale."""
    css = re.sub(r"/\*.*?\*/", "", _read_codex_css(), flags=re.S)
    for token in (".txt", ".tc-b", ".tp", ".tr", ".rc", ".rs",
                  ".think", ".sys", ".md-body", ".merge-", ".llm-panel",
                  ".bash-panel", ".system-prompt", ".prompt"):
        assert token not in css, (
            f"remote-codex.css must not restyle {token} panels"
        )


def test_main_js_result_status_uses_class_not_inline_style() -> None:
    """createResultPanel must not emit an inline font-size (inline
    styles are unbeatable by stylesheets); it uses .rc-status classes."""
    js = MAIN_JS.read_text(encoding="utf-8")
    assert "font-size:var(--fs-xl)" not in js, (
        "main.js must not emit the inline font-size:var(--fs-xl) "
        "status line; use the .rc-status class instead"
    )
    assert 'class="rc-status"' in js, (
        "the Continue status line must carry class rc-status"
    )
    assert 'class="rc-status rc-status-fail"' in js, (
        "the FAILED status line must carry class rc-status rc-status-fail"
    )


def test_main_css_styles_rc_status_like_old_inline_style() -> None:
    """main.css replicates the old inline declarations so the VS Code
    webview keeps its exact former look (yellow/red, bold, fs-xl)."""
    css = MAIN_CSS.read_text(encoding="utf-8")
    m = re.search(r"\.rc-status\s*\{([^}]*)\}", css)
    assert m, ".rc-status rule missing from main.css"
    rule = m.group(1)
    assert "color: var(--yellow)" in rule
    assert "font-weight: 700" in rule
    assert "font-size: var(--fs-xl)" in rule
    assert "margin-bottom: 10px" in rule
    fail = re.search(r"\.rc-status\.rc-status-fail\s*\{([^}]*)\}", css)
    assert fail, ".rc-status.rc-status-fail rule missing from main.css"
    assert "color: var(--red)" in fail.group(1)
