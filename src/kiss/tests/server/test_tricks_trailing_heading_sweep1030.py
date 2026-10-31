# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: a promptlet whose last line is a bare ``##`` is refused.

The Inject promptlet panel's Add and edit buttons store the trimmed
body as a ``## Trick`` section of a real ``MY_INJECTION.md``
(``KISS_HOME`` pinned to a temp dir).  The reader splits sections at
every line that starts with ``##`` and whitespace, a newline included,
so a body ending in ``\\n## `` (trimmed to ``\\n##``) used to be stored
and read back cut short, the ``##`` left behind as an empty section.
It is refused like any other ``##`` heading line; a ``###`` line, which
the reader does not split at, is still stored as typed.
"""

from __future__ import annotations

from kiss.server import tricks
from kiss.tests.server.test_tricks_add_promptlet import _TricksHome

_HEADING_ERROR = "Promptlet must not contain a line starting with '## '"


class TestTrailingHeadingLine(_TricksHome):
    """A bare ``##`` last line is a heading the reader would split at."""

    def test_add_refuses_a_body_ending_in_a_bare_heading(self) -> None:
        self.assertEqual(
            tricks.append_my_injection_trick("Check the docs\n## "), _HEADING_ERROR,
        )
        self.assertFalse(self.user_file.exists())
        self.assertNotIn("Check the docs", tricks.read_tricks_data()["tricks"])

    def test_edit_refuses_a_body_ending_in_a_bare_heading(self) -> None:
        self.assertIsNone(tricks.append_my_injection_trick("Write tests first."))
        before = self.user_text()
        self.assertEqual(
            tricks.edit_my_injection_trick("Write tests first.", "Check the docs\n##"),
            _HEADING_ERROR,
        )
        self.assertEqual(self.user_text(), before)

    def test_a_deeper_heading_line_is_stored_as_typed(self) -> None:
        self.assertIsNone(tricks.append_my_injection_trick("Check the docs\n###"))
        self.assertIn("Check the docs\n###", tricks.read_tricks_data()["tricks"])
