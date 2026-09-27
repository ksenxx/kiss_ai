# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for bugs, redundancies, and inconsistencies in
``kiss.server`` — audit round 3.

Bugs
----
N3: the ``@``-mention scan's depth check used to be an off-by-one
    (``len(rel_root.parts) - 1 > 3`` written as if ``PurePath('.').parts``
    were ``('.',)``).  The scan now lives in ``kiss.server.file_index``
    and its limit is the explicit constant ``MAX_DEPTH``: a directory at
    depth ``MAX_DEPTH`` is listed but never descended into.  The
    behavioral depth test lives in ``kiss.tests.server.test_vscode_audit_3``;
    this module pins the constant and the "listed but not descended"
    rule at the boundary.  (The former ``PurePath('.').parts`` root-cause
    tests only exercised the standard library and were dropped.)

(N5 covered empty-tab_id collisions in the merge-data write paths of
the interactive diff/merge review workflow; that workflow and its
``_merge_data_dir``/``_save_untracked_base`` helpers were removed from
the server, so those tests are gone.)
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from kiss.server.file_index import MAX_DEPTH, FileIndex


class TestScanFilesDepthBoundary(unittest.TestCase):
    """N3: the depth limit is ``MAX_DEPTH`` and is applied without an off-by-one."""

    def test_max_depth_is_twelve(self) -> None:
        """The picker descends twelve levels below the index root."""
        assert MAX_DEPTH == 12

    def test_directory_at_max_depth_is_listed_but_not_descended(self) -> None:
        """A depth-``MAX_DEPTH`` directory's subdirectory is an entry, its contents are not.

        ``d0/.../d{MAX_DEPTH-1}`` sits at depth ``MAX_DEPTH`` and is
        descended into (so ``d{MAX_DEPTH}/`` below it is an entry), but
        ``d{MAX_DEPTH}`` itself is not, so nothing inside it is listed.
        """
        td = tempfile.mkdtemp()
        try:
            names = [f"d{i}" for i in range(MAX_DEPTH + 1)]
            deepest = Path(td, *names)
            deepest.mkdir(parents=True)
            (deepest / "hidden.py").write_text("x")
            (deepest / "sub").mkdir()

            index = FileIndex.scan(td)

            listed_dir = "/".join(names) + "/"
            assert listed_dir in index.paths
            assert "/".join(names) + "/hidden.py" not in index.paths
            assert "/".join(names) + "/sub/" not in index.paths
            assert "/".join(names[:MAX_DEPTH]) in index.dirs
            assert "/".join(names) not in index.dirs
        finally:
            shutil.rmtree(td)


if __name__ == "__main__":
    unittest.main()
