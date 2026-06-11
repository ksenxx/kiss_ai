# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt round 2: ``_read_url_from_file`` crashes on non-dict JSON.

``_read_url_from_file``'s docstring promises: "Returns ``None`` on
missing file, parse error, or empty content."  The implementation
only guards the ``read_text``/``json.loads`` step — the subsequent
``data.get("tunnel")`` is executed unguarded, so a URL file whose
content is valid JSON but not an object (e.g. ``[]``, ``"..."``, a
number, ``null``) raises ``AttributeError`` instead of returning
``None``.

This matters for real callers:

* ``_print_url`` (the ``kiss-web --url`` CLI) expects ``None`` and
  prints "KISS Sorcar web server is not running." with exit code 1;
  a corrupted/foreign ``~/.kiss/remote-url.json`` instead crashes the
  CLI with an unhandled traceback.
* ``RemoteAccessServer._send_welcome_info`` runs the helper in an
  executor; a raised exception propagates through ``_handle_ready``
  into the WebSocket handler, tearing down the authenticated
  connection over a bad file.

The tests below exercise the helper against real files on disk — no
mocks, no server needed for the contract itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from kiss.agents.vscode.web_server import _read_url_from_file


class TestReadUrlFromFileNonDictJson:
    """Non-dict (but valid) JSON content must yield ``None``, not raise."""

    def test_json_array_returns_none(self, tmp_path: Path) -> None:
        url_file = tmp_path / "remote-url.json"
        url_file.write_text("[]\n")
        assert _read_url_from_file(url_file) is None

    def test_json_string_returns_none(self, tmp_path: Path) -> None:
        url_file = tmp_path / "remote-url.json"
        url_file.write_text('"https://example.trycloudflare.com"\n')
        assert _read_url_from_file(url_file) is None

    def test_json_null_returns_none(self, tmp_path: Path) -> None:
        url_file = tmp_path / "remote-url.json"
        url_file.write_text("null\n")
        assert _read_url_from_file(url_file) is None

    def test_valid_dict_still_works(self, tmp_path: Path) -> None:
        url_file = tmp_path / "remote-url.json"
        url_file.write_text(json.dumps({
            "local": "https://localhost:8787",
            "tunnel": "https://abc.trycloudflare.com",
        }))
        assert _read_url_from_file(url_file) == "https://abc.trycloudflare.com"

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert _read_url_from_file(tmp_path / "nope.json") is None
