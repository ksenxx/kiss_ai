# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here


"""Server-only tests extracted from ``kiss.tests.agents.vscode.test_vscode_audit_3``.

Moved here because their full dependency closure touches only
kiss.core, kiss.agents.sorcar and kiss.server (task: relocate
core+sorcar+server-only test methods to tests/server).
"""


from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.server.diff_merge import _scan_files


class TestScanFilesDepthOffByOne(unittest.TestCase):
    """N3: ``_scan_files`` checks ``len(rel_root.parts) - 1 > 3``
    which was written assuming ``PurePath('.').parts == ('.',)``.
    Since ``PurePath('.').parts`` is actually ``()``, the ``- 1``
    creates an off-by-one that allows one extra level of nesting.
    """

    def test_depth_10_files_are_included(self) -> None:
        """Behavioral: files at depth 10 are included when the
        intended limit was depth 9.

        Creates a directory tree:
          root/a/b/c/d/e/f/g/h/i/shallow.txt  (depth 9)
          root/a/b/c/d/e/f/g/h/i/j/deep.txt   (depth 10)
          root/a/b/c/d/e/f/g/h/i/j/k/very_deep.txt  (depth 11)

        With the off-by-one, depth 10 is included.  Without it, only
        depth 9 should be included.
        """
        td = tempfile.mkdtemp()
        try:
            d9 = os.path.join(td, "a", "b", "c", "d", "e", "f", "g", "h", "i")
            os.makedirs(d9)
            Path(d9, "shallow.txt").write_text("ok")

            d10 = os.path.join(td, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
            os.makedirs(d10)
            Path(d10, "deep.txt").write_text("too deep?")

            d11 = os.path.join(td, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")
            os.makedirs(d11)
            Path(d11, "very_deep.txt").write_text("way too deep")

            result = _scan_files(td)
            file_results = [p for p in result if not p.endswith("/")]

            assert "a/b/c/d/e/f/g/h/i/shallow.txt" in file_results, (
                "depth-9 files should always be included"
            )
            assert "a/b/c/d/e/f/g/h/i/j/deep.txt" in file_results, (
                "N3: depth-10 files are included due to the off-by-one bug"
            )
            assert "a/b/c/d/e/f/g/h/i/j/k/very_deep.txt" not in file_results, (
                "depth-11 files should be excluded"
            )
        finally:
            shutil.rmtree(td)
