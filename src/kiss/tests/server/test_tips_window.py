# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the fresh-install Tips window (Python side).

The chat webview template ``media/chat.html`` is shared between the
VS Code extension (``SorcarTab.buildChatHtml``) and the remote webapp
(``web_server._build_html``), so the remote builder must also
substitute the ``{{TIPS_JSON}}`` / ``{{TIPS_SRC}}`` placeholders.

Contract locked in here:

* :func:`kiss.server.tips.read_tips` parses the bundled
  ``src/kiss/TIPS.md``: every line starting with ``# Tip`` starts a
  new tip whose body is the markdown text up to the next such line
  (or EOF), trimmed.  Empty bodies are skipped; a missing file yields
  ``[]``.  The ``KISS_TIPS_PATH`` env var overrides the file location.
* ``web_server._build_html()`` injects ``window.__TIPS__`` with the
  parsed tips, ``show: true`` whenever there are tips, and the running
  ``version``: the server cannot tell which browser has seen the tips,
  so tips.js opens them once per version per browser (localStorage),
  i.e. on first use and again after every update.  It loads
  ``media/tips.js`` with a cache-buster and leaves no ``{{TIPS...}}``
  placeholder behind.
* ``media/chat.html`` and ``media/tips.js`` ship the web-component
  surface consumed by both hosts.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kiss.server import web_server
from kiss.server.tips import read_tips, tips_disabled


class TestReadTips(unittest.TestCase):
    """``read_tips`` parses ``# Tip`` sections from TIPS.md."""

    def _with_tips(self, content: str) -> list[str]:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tips_file = Path(tmp) / "TIPS.md"
            tips_file.write_text(content)
            prev = os.environ.get("KISS_TIPS_PATH")
            os.environ["KISS_TIPS_PATH"] = str(tips_file)
            try:
                return read_tips()
            finally:
                if prev is None:
                    del os.environ["KISS_TIPS_PATH"]
                else:
                    os.environ["KISS_TIPS_PATH"] = prev

    def test_parses_body_after_each_tip_line(self) -> None:
        """Every line starting with ``# Tip`` begins a new tip."""
        tips = self._with_tips(
            "# Tip\n\n## First\n- a\n\n# Tip \n\nSecond **bold**.\n"
            "\n# Tip\n\nThird.\n"
        )
        self.assertEqual(
            tips, ["## First\n- a", "Second **bold**.", "Third."]
        )

    def test_ignores_preamble_and_empty_bodies(self) -> None:
        """Text before the first ``# Tip`` and empty tips are skipped."""
        tips = self._with_tips("preamble\n\n# Tip\n\n  \n# Tip\n\nOnly.\n")
        self.assertEqual(tips, ["Only."])

    def test_does_not_split_on_lookalike_lines(self) -> None:
        """``## Tip`` and indented ``# Tip`` do not start a new tip."""
        tips = self._with_tips("# Tip\n\nbody\n## Tip\n  # Tip\nend\n")
        self.assertEqual(tips, ["body\n## Tip\n  # Tip\nend"])

    def test_missing_file_yields_empty_list(self) -> None:
        """A missing TIPS.md degrades gracefully to ``[]``."""
        import os

        prev = os.environ.get("KISS_TIPS_PATH")
        os.environ["KISS_TIPS_PATH"] = "/nonexistent/TIPS.md"
        try:
            self.assertEqual(read_tips(), [])
        finally:
            if prev is None:
                del os.environ["KISS_TIPS_PATH"]
            else:
                os.environ["KISS_TIPS_PATH"] = prev


@contextlib.contextmanager
def _environ(**values: str) -> Iterator[None]:
    """Set environment variables for the block, then restore them."""
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestTipsInRemoteHtml(unittest.TestCase):
    """``web_server._build_html`` wires the tips surface into the page."""

    def _render(self, tips_md: str, opted_out: bool = False) -> tuple[str, dict[str, Any]]:
        """Build the remote page against a private ``TIPS.md`` and
        ``$KISS_HOME`` (the session home carries the test suite's own
        opt-out marker); returns the HTML and the parsed ``__TIPS__``."""
        with tempfile.TemporaryDirectory() as tmp:
            tips_file = Path(tmp) / "TIPS.md"
            tips_file.write_text(tips_md)
            home = Path(tmp) / "home"
            home.mkdir()
            if opted_out:
                (home / "TIPS_DISABLED").write_text("2026-10-01T00:00:00\n")
            with _environ(KISS_TIPS_PATH=str(tips_file), KISS_HOME=str(home)):
                html = web_server._build_html()  # type: ignore[attr-defined]
        m = re.search(r"window\.__TIPS__\s*=\s*(\{.*?\});</script>", html)
        assert m is not None, "window.__TIPS__ must be assigned a JSON object literal"
        self.assertNotIn("</script>", m.group(1))
        return html, json.loads(m.group(1).replace("<\\/", "</"))

    def test_tips_json_never_embeds_raw_close_script(self) -> None:
        """``</script>`` inside a tip body must be escaped so it cannot
        terminate the inline ``window.__TIPS__`` script block."""
        _html, cfg = self._render("# Tip\n\nUse `</script>` carefully.\n")
        self.assertEqual(cfg["tips"], ["Use `</script>` carefully."])

    def test_auto_open_once_per_version_per_browser(self) -> None:
        """With tips and no opt-out the page says ``show: true`` and
        names the running version, which tips.js remembers per browser
        so the window opens on first use and after every update."""
        _html, cfg = self._render("# Tip\n\nHello.\n")
        self.assertEqual(
            cfg,
            {
                "tips": ["Hello."],
                "show": True,
                "version": web_server._read_version(),  # type: ignore[attr-defined]
            },
        )

    def test_no_tips_means_no_auto_open(self) -> None:
        """An empty tips file yields ``show: false`` so tips.js never
        mounts a blank window on the remote page."""
        _html, cfg = self._render("preamble only\n")
        self.assertEqual(cfg["tips"], [])
        self.assertIs(cfg["show"], False)

    def test_shared_opt_out_marker_keeps_the_remote_tips_closed(self) -> None:
        """``$KISS_HOME/TIPS_DISABLED`` (written by the ``tipsOptOut`` API
        from either surface) turns ``show`` off on the remote page
        too, so an opt-out made on one surface holds on every surface."""
        _html, cfg = self._render("# Tip\n\nHello.\n", opted_out=True)
        self.assertEqual(cfg["tips"], ["Hello."])
        self.assertIs(cfg["show"], False)

    def test_tips_disabled_follows_the_marker(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            with _environ(KISS_HOME=home):
                self.assertFalse(tips_disabled())
                (Path(home) / "TIPS_DISABLED").write_text("x\n")
                self.assertTrue(tips_disabled())


if __name__ == "__main__":
    unittest.main()
