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


def _build_html() -> str:
    from kiss.server.web_server import _build_html

    return _build_html()


def test_body_keeps_remote_chat_class() -> None:
    """The remote page body keeps the remote-chat scoping class."""
    html = _build_html()
    assert '<body class="remote-chat">' in html
