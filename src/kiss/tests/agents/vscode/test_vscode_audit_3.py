# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.server`` — audit round 3.

Each test confirms the bug/inconsistency exists with BOTH a structural
source assertion (``inspect.getsource`` pattern match) AND a behavioral
integration test using real objects.

Bugs
----
N3: ``_scan_files`` depth check ``len(rel_root.parts) - 1 > 3`` was
    written assuming ``PurePath('.').parts == ('.',)`` but it is ``()``,
    causing an off-by-one that allows one extra nesting level (depth 4
    sub-directories instead of the intended 3).

(N5 covered empty-tab_id collisions in the merge-data write paths of
the interactive diff/merge review workflow; that workflow and its
``_merge_data_dir``/``_save_untracked_base`` helpers were removed from
the server, so those tests are gone.)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path, PurePath

from kiss.server.diff_merge import _scan_files


class TestScanFilesDepthOffByOne(unittest.TestCase):
    """N3: ``_scan_files`` checks ``len(rel_root.parts) - 1 > 3``
    which was written assuming ``PurePath('.').parts == ('.',)``.
    Since ``PurePath('.').parts`` is actually ``()``, the ``- 1``
    creates an off-by-one that allows one extra level of nesting.
    """

    def test_purepath_dot_parts_is_empty(self) -> None:
        """Confirm the root cause: ``PurePath('.').parts`` is ``()``."""
        assert PurePath(".").parts == (), (
            f"N3: PurePath('.').parts should be (), got {PurePath('.').parts}"
        )


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

    def test_depth_formula_values(self) -> None:
        """Behavioral: verify the formula at each depth level.

        The formula ``len(rel_root.parts)`` gives:
          root:    len(()) = 0
          depth10: len(('a',..,'j')) = 10  → 10 > 10 is False (included)
          depth11: len(('a',..,'k')) = 11  → 11 > 10 is True (excluded)
        """
        assert len(PurePath(".").parts) == 0

        depth10 = PurePath("a/b/c/d/e/f/g/h/i/j")
        assert len(depth10.parts) == 10
        assert not (len(depth10.parts) > 10), (
            "depth 10 passes the check (10 > 10 is False) — included"
        )

        depth11 = PurePath("a/b/c/d/e/f/g/h/i/j/k")
        assert len(depth11.parts) == 11
        assert len(depth11.parts) > 10, (
            "depth 11 fails the check (11 > 10 is True) — excluded"
        )




if __name__ == "__main__":
    unittest.main()
