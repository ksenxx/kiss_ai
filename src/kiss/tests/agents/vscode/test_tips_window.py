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

* :func:`kiss.agents.vscode.tips.read_tips` parses the bundled
  ``src/kiss/TIPS.md``: every line starting with ``# Tip`` starts a
  new tip whose body is the markdown text up to the next such line
  (or EOF), trimmed.  Empty bodies are skipped; a missing file yields
  ``[]``.  The ``KISS_TIPS_PATH`` env var overrides the file location.
* ``web_server._build_html()`` injects ``window.__TIPS__`` with the
  parsed tips and ``show: false`` (the remote webapp is never a fresh
  VS Code installation), loads ``media/tips.js`` with a cache-buster,
  and leaves no ``{{TIPS...}}`` placeholder behind.
* ``media/chat.html`` and ``media/tips.js`` ship the web-component
  surface consumed by both hosts.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from kiss.agents.vscode import web_server
from kiss.agents.vscode.tips import read_tips


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

    def test_bundled_tips_md_yields_tips(self) -> None:
        """The bundled ``src/kiss/TIPS.md`` produces non-empty tips."""
        tips = read_tips()
        self.assertGreater(len(tips), 0)
        for tip in tips:
            self.assertTrue(tip.strip())


class TestTipsInRemoteHtml(unittest.TestCase):
    """``web_server._build_html`` wires the tips surface into the page."""

    def test_html_injects_tips_config_and_script(self) -> None:
        """The served page defines ``window.__TIPS__`` and loads
        ``tips.js`` with a content-hash cache-buster; no ``{{TIPS...}}``
        placeholder survives substitution."""
        html = web_server._build_html()  # type: ignore[attr-defined]
        m = re.search(r"window\.__TIPS__\s*=\s*(\{.*?\});</script>", html)
        self.assertIsNotNone(m, "window.__TIPS__ must be defined")
        assert m is not None
        cfg = json.loads(m.group(1).replace("<\\/", "</"))
        self.assertEqual(sorted(cfg), ["show", "tips"])
        self.assertFalse(cfg["show"], "remote webapp never auto-shows tips")
        self.assertEqual(cfg["tips"], read_tips())
        self.assertRegex(html, r'src="/media/tips\.js\?v=[0-9a-f]{16}"')
        self.assertNotIn("{{TIPS", html)

    def test_tips_json_never_embeds_raw_close_script(self) -> None:
        """``</script>`` inside a tip body must be escaped so it cannot
        terminate the inline ``window.__TIPS__`` script block."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tips_file = Path(tmp) / "TIPS.md"
            tips_file.write_text("# Tip\n\nUse `</script>` carefully.\n")
            prev = os.environ.get("KISS_TIPS_PATH")
            os.environ["KISS_TIPS_PATH"] = str(tips_file)
            try:
                html = web_server._build_html()  # type: ignore[attr-defined]
            finally:
                if prev is None:
                    del os.environ["KISS_TIPS_PATH"]
                else:
                    os.environ["KISS_TIPS_PATH"] = prev
        m = re.search(r"window\.__TIPS__\s*=\s*(\{.*?\});</script>", html)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertNotIn("</script>", m.group(1))
        cfg = json.loads(m.group(1).replace("<\\/", "</"))
        self.assertEqual(cfg["tips"], ["Use `</script>` carefully."])

    def test_media_ships_tips_component(self) -> None:
        """``media/tips.js`` defines the ``<kiss-tips-panel>`` web
        component and ``chat.html`` loads it via the placeholders."""
        media = Path(web_server.MEDIA_DIR)
        tips_js = (media / "tips.js").read_text()
        self.assertIn("kiss-tips-panel", tips_js)
        self.assertIn("customElements.define", tips_js)
        chat_html = (media / "chat.html").read_text()
        self.assertIn("{{TIPS_JSON}}", chat_html)
        self.assertIn("{{TIPS_SRC}}", chat_html)


if __name__ == "__main__":
    unittest.main()
