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

from kiss.server.file_index import MAX_DEPTH, FileIndex


class TestScanFilesDepthBoundary(unittest.TestCase):
    """N3: the ``@``-mention scan must stop at exactly ``MAX_DEPTH``.

    The old ``_scan_files`` depth check had an off-by-one; the index
    scan now descends into directories at depth ``< MAX_DEPTH`` only,
    so a directory at depth ``MAX_DEPTH`` is listed (and its files are
    listed) but its subdirectories are never entered.
    """

    def test_files_at_max_depth_are_included_and_deeper_ones_are_not(self) -> None:
        """Behavioral: files at ``MAX_DEPTH - 1``, ``MAX_DEPTH`` and ``MAX_DEPTH + 1``.

        Creates a directory tree of nested ``dN`` directories:
          root/d0/.../d{MAX_DEPTH-2}/shallow.txt     (depth MAX_DEPTH - 1)
          root/d0/.../d{MAX_DEPTH-1}/boundary.txt    (depth MAX_DEPTH)
          root/d0/.../d{MAX_DEPTH}/too_deep.txt      (depth MAX_DEPTH + 1)
        """
        td = tempfile.mkdtemp()
        try:
            names = [f"d{i}" for i in range(MAX_DEPTH + 1)]
            shallow = os.path.join(td, *names[: MAX_DEPTH - 1])
            boundary = os.path.join(td, *names[:MAX_DEPTH])
            too_deep = os.path.join(td, *names)
            os.makedirs(too_deep)
            Path(shallow, "shallow.txt").write_text("ok")
            Path(boundary, "boundary.txt").write_text("at the limit")
            Path(too_deep, "too_deep.txt").write_text("way too deep")

            index = FileIndex.scan(td)
            file_results = [p for p in index.paths if not p.endswith("/")]

            assert "/".join(names[: MAX_DEPTH - 1]) + "/shallow.txt" in file_results, (
                f"depth-{MAX_DEPTH - 1} files should always be included"
            )
            assert "/".join(names[:MAX_DEPTH]) + "/boundary.txt" in file_results, (
                f"N3: depth-{MAX_DEPTH} files are included (the directory is listed)"
            )
            assert "/".join(names) + "/" in index.paths, (
                f"the depth-{MAX_DEPTH + 1} directory itself is listed as an entry"
            )
            assert "/".join(names) + "/too_deep.txt" not in file_results, (
                f"depth-{MAX_DEPTH + 1} files should be excluded"
            )
            assert "/".join(names[:MAX_DEPTH]) in index.dirs
            assert "/".join(names) not in index.dirs, "a listed-only directory is not descended"
        finally:
            shutil.rmtree(td)
