# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_codex_mobile_layout``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import re


def _build_html() -> str:
    from kiss.server.web_server import _build_html

    return _build_html()


def test_body_keeps_remote_chat_class() -> None:
    """The remote page body keeps the remote-chat scoping class."""
    html = _build_html()
    assert '<body class="remote-chat">' in html


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
