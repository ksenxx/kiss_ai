# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The tips bootstrap carries the daemon's persisted "don't show tips again" choice.

``tips.js`` reads ``window.__TIPS__.optOut`` (``readOptOut``) so a
choice made on another device, or before this browser's site data was
cleared, still holds: the tips window's checkbox shows it and the
window is not opened automatically.  The daemon only sent ``show``, so
a browser without its own localStorage copy showed the checkbox
unticked although ``$KISS_HOME/TIPS_DISABLED`` existed.  Both the page
the daemon serves and the ``tipsData`` its ``ready`` handler sends now
carry ``optOut``.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from kiss.server import web_server
from kiss.server.tips import tips_data
from kiss.tests.server.conftest import environ


def _page_tips(opted_out: bool) -> tuple[dict[str, Any], dict[str, object]]:
    """The served page's ``__TIPS__`` and :func:`tips_data` under a private home."""
    with tempfile.TemporaryDirectory() as tmp:
        tips_file = Path(tmp) / "TIPS.md"
        tips_file.write_text("# Tip\n\nHello.\n")
        home = Path(tmp) / "home"
        home.mkdir()
        if opted_out:
            (home / "TIPS_DISABLED").write_text("2026-10-30T00:00:00\n")
        with environ(KISS_TIPS_PATH=str(tips_file), KISS_HOME=str(home)):
            html = web_server._build_html()  # type: ignore[attr-defined]
            data = tips_data("1.2.3")
    match = re.search(r"window\.__TIPS__\s*=\s*(\{.*?\});</script>", html)
    assert match is not None
    return json.loads(match.group(1).replace("<\\/", "</")), data


def test_opt_out_reaches_the_page_and_the_ready_bootstrap() -> None:
    page, data = _page_tips(opted_out=True)
    assert page["optOut"] is True
    assert page["show"] is False
    assert data["optOut"] is True


def test_no_opt_out_is_sent_as_false() -> None:
    page, data = _page_tips(opted_out=False)
    assert page["optOut"] is False
    assert page["show"] is True
    assert data["optOut"] is False
