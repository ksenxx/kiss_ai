# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""An unreadable ``MY_INJECTION.md`` is a user-facing refusal, not an exception.

The add/delete/edit helpers of :mod:`kiss.server.tricks` read the file
through ``_read_my_injection``; a read the OS refuses (a
permission-less file) must come back as the "Could not read" message
like the other refusals, instead of escaping as ``OSError``.  Real
files under a temp ``KISS_HOME``; no mocks.
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from kiss.server import tricks
from kiss.tests.conftest import is_root, posix_only


@posix_only("file permission bits")
@unittest.skipIf(is_root(), "root reads a permission-less file")
class TestUnreadableMyInjection(unittest.TestCase):
    """Every editing helper refuses an unreadable file with the same message."""

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in ("KISS_HOME", "KISS_INJECTIONS_PATH")}
        # Cleanups run after ``tearDown`` restored the file's permission bits.
        home = tempfile.TemporaryDirectory(prefix="kiss_unreadable_trick_")
        self.addCleanup(home.cleanup)
        self.kiss_dir = Path(home.name) / ".kiss"
        self.kiss_dir.mkdir(parents=True)
        bundled = self.kiss_dir / "bundled_INJECTIONS.md"
        bundled.write_text("## Trick\n\nBundled.\n", encoding="utf-8")
        os.environ["KISS_HOME"] = str(self.kiss_dir)
        os.environ["KISS_INJECTIONS_PATH"] = str(bundled)
        self.user_file = self.kiss_dir / "MY_INJECTION.md"
        self.user_file.write_text("## Trick\n\nMine.\n", encoding="utf-8")
        self.user_file.chmod(0)

    def tearDown(self) -> None:
        self.user_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_add_delete_edit_report_unreadable_file(self) -> None:
        """Each helper answers with the read refusal and leaves the file alone."""
        message = "Could not read ~/.kiss/MY_INJECTION.md"
        self.assertEqual(tricks.append_my_injection_trick("New."), message)
        self.assertEqual(tricks.delete_my_injection_trick("Mine."), message)
        self.assertEqual(tricks.edit_my_injection_trick("Mine.", "Yours."), message)
        self.user_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.assertEqual(self.user_file.read_text(encoding="utf-8"), "## Trick\n\nMine.\n")


if __name__ == "__main__":
    unittest.main()
