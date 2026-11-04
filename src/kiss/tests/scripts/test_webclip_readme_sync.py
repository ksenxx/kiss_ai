# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Keep the root README's Web Clip prose in step with the code it describes.

The README documents the ``"..."`` menu's **iPhone Home Screen** item,
the ``/webclip.mobileconfig`` route and every option of
``python -m kiss.scripts.webclip``.  These checks read the option names,
the route path, the menu label and the signer file names from the code,
so a renamed flag or route fails here instead of drifting silently.
"""

from __future__ import annotations

import re
from pathlib import Path

from kiss.scripts import webclip
from kiss.server import web_server

_REPO_ROOT = Path(__file__).resolve().parents[4]
_README = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
_CHAT_HTML = (_REPO_ROOT / "src" / "kiss" / "webapp" / "media" / "chat.html").read_text(
    encoding="utf-8"
)


def test_readme_names_every_webclip_cli_option() -> None:
    """Each ``--option`` of ``webclip.build_parser()`` appears in the README."""
    options = [
        action.option_strings[0]
        for action in webclip.build_parser()._actions
        if action.option_strings and action.option_strings[0] != "-h"
    ]
    assert options, "build_parser() lists no options"
    missing = [opt for opt in options if f"`{opt}`" not in _README]
    assert not missing, f"README.md does not mention {missing}"


def test_readme_names_the_route_the_menu_item_and_the_signer_files() -> None:
    """The route path, the menu label and the signer file names match the code."""
    assert f"`{web_server.WEBCLIP_PATH}`" in _README
    label = re.search(
        r'id="webclip-btn".*?<span class="more-item-label">([^<]+)</span>',
        _CHAT_HTML,
        re.S,
    )
    assert label is not None, "chat.html has no webclip-btn menu item"
    assert f"**{label.group(1)}**" in _README
    assert f"`{webclip.SIGNER_CERT_FILE}`" in _README
    assert f"`{webclip.SIGNER_KEY_FILE}`" in _README
    assert f"{webclip.ICON_SIZE}x{webclip.ICON_SIZE}" in _README
