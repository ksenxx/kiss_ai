# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt 6: Edit must not rewrite a CRLF file's line endings.

``UsefulTools.Edit`` used to read the file with universal-newline
translation (``Path.read_text()`` turns ``\\r\\n`` into ``\\n``) and
write the edited text back as-is — so a one-line edit on a CRLF file
silently converted EVERY line ending in the file to LF, producing
massive spurious diffs (and breaking CRLF-required files such as
``.bat`` scripts).
"""

import tempfile
import unittest
from pathlib import Path

from kiss.agents.sorcar.useful_tools import UsefulTools


class TestEditCrlfPreservation(unittest.TestCase):
    """Edit on CRLF files must only change the edited region."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tools = UsefulTools(work_dir=self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_single_line_edit_preserves_crlf(self) -> None:
        """Editing one line must not strip CRLF from untouched lines."""
        p = self.dir / "f.txt"
        p.write_bytes(b"line one\r\nline two\r\nline three\r\n")
        result = self.tools.Edit(str(p), "line two", "LINE 2")
        self.assertIn("Successfully replaced 1", result)
        self.assertEqual(
            p.read_bytes(), b"line one\r\nLINE 2\r\nline three\r\n",
        )

    def test_multiline_lf_old_string_matches_crlf_file(self) -> None:
        """old_string with LF (as the model sees via Read) must match a
        CRLF file, and the replacement must be written with CRLF."""
        p = self.dir / "f.txt"
        p.write_bytes(b"alpha\r\nbeta\r\ngamma\r\ndelta\r\n")
        # Read() shows the model LF-translated text, so the model passes
        # LF newlines inside old_string/new_string.
        result = self.tools.Edit(str(p), "beta\ngamma", "BETA\nGAMMA")
        self.assertIn("Successfully replaced 1", result)
        self.assertEqual(
            p.read_bytes(), b"alpha\r\nBETA\r\nGAMMA\r\ndelta\r\n",
        )

    def test_replace_all_preserves_crlf(self) -> None:
        """replace_all on a CRLF file must keep CRLF everywhere."""
        p = self.dir / "f.txt"
        p.write_bytes(b"x = 1\r\ny = 1\r\nz = 1\r\n")
        result = self.tools.Edit(str(p), "= 1", "= 2", replace_all=True)
        self.assertIn("Successfully replaced 3", result)
        self.assertEqual(p.read_bytes(), b"x = 2\r\ny = 2\r\nz = 2\r\n")

    def test_crlf_uniqueness_check_with_lf_old_string(self) -> None:
        """A multiline LF old_string occurring twice in a CRLF file must
        still trigger the not-unique error (not 'not found')."""
        p = self.dir / "f.txt"
        p.write_bytes(b"a\r\nb\r\nc\r\na\r\nb\r\n")
        result = self.tools.Edit(str(p), "a\nb", "X")
        self.assertIn("appears 2 times", result)
        # File untouched.
        self.assertEqual(p.read_bytes(), b"a\r\nb\r\nc\r\na\r\nb\r\n")

    def test_lf_file_unaffected_regression(self) -> None:
        """Plain-LF files behave exactly as before."""
        p = self.dir / "f.txt"
        p.write_bytes(b"one\ntwo\nthree\n")
        result = self.tools.Edit(str(p), "two", "TWO")
        self.assertIn("Successfully replaced 1", result)
        self.assertEqual(p.read_bytes(), b"one\nTWO\nthree\n")

    def test_crlf_old_string_passed_verbatim(self) -> None:
        """An old_string that already carries CRLF matches raw content
        directly (no normalisation needed)."""
        p = self.dir / "f.txt"
        p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
        result = self.tools.Edit(str(p), "one\r\ntwo", "ONE\r\nTWO")
        self.assertIn("Successfully replaced 1", result)
        self.assertEqual(p.read_bytes(), b"ONE\r\nTWO\r\nthree\r\n")

    def test_mixed_newlines_in_new_string_not_double_translated(self) -> None:
        """A new_string already carrying some CRLFs must not be corrupted
        into \\r\\r\\n by the CRLF normalisation."""
        p = self.dir / "f.txt"
        p.write_bytes(b"start\r\nmiddle\r\nend\r\n")
        result = self.tools.Edit(str(p), "middle\nend", "MID\r\nEND")
        self.assertIn("Successfully replaced 1", result)
        self.assertEqual(p.read_bytes(), b"start\r\nMID\r\nEND\r\n")

    def test_mixed_endings_only_edit_region_changes(self) -> None:
        """A file with mixed LF/CRLF endings keeps every untouched byte."""
        p = self.dir / "f.txt"
        p.write_bytes(b"lf line\ncrlf line\r\nanother lf\n")
        result = self.tools.Edit(str(p), "another lf", "ANOTHER LF")
        self.assertIn("Successfully replaced 1", result)
        self.assertEqual(
            p.read_bytes(), b"lf line\ncrlf line\r\nANOTHER LF\n",
        )


if __name__ == "__main__":
    unittest.main()
