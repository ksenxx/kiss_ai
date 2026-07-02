# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 8 (group A): CR line endings leak into ghost-text suggestions.

``clip_autocomplete_suggestion`` promises ("Stops at newlines.") that a
multi-line suggestion is clipped to its first line before being shown
as ghost text.  It clipped with ``s.split("\\n")[0]``, which handles
only LF: a suggestion whose source text uses CRLF line endings (e.g. a
history task pasted from a Windows editor, or chat-context text stored
verbatim in the database) left a trailing ``"\\r"`` on the ghost text —
and a lone-CR line break (legacy Mac / raw terminal capture) was not
clipped at all, so the "single line" ghost contained an embedded
carriage-return control character.  Accepting such a completion typed
the invisible control character into the chat input.
"""

from __future__ import annotations

import unittest

from kiss.agents.vscode.helpers import clip_autocomplete_suggestion


class TestCrlfGhostClipping(unittest.TestCase):
    """Ghost suggestions must never contain carriage returns."""

    def test_crlf_suggestion_has_no_trailing_cr(self) -> None:
        """CRLF-terminated first line must be clipped without the CR."""
        out = clip_autocomplete_suggestion("run", " tests\r\nsecond line")
        assert out == " tests", repr(out)

    def test_lone_cr_is_a_line_boundary(self) -> None:
        """A bare CR (legacy line break) must also stop the ghost text."""
        out = clip_autocomplete_suggestion("fix", " the bug\rmore text")
        assert out == " the bug", repr(out)

    def test_lf_behaviour_unchanged(self) -> None:
        """Plain-LF clipping keeps working exactly as before."""
        out = clip_autocomplete_suggestion("fix", " the bug\nmore text")
        assert out == " the bug", repr(out)


if __name__ == "__main__":
    unittest.main()
