# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Performance stress tests for ``FileIndex.scan`` with 5000+ files.

Verifies that the file picker's index builds quickly and that a tree of a
few thousand entries is returned in full: the cap is ``_SCAN_FILES_CAP``
(1,000,000), so no realistic workspace is truncated and deep subtrees such
as ``src/`` are never crowded out by large sibling directories.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from kiss.server.file_index import _SCAN_FILES_CAP, FileIndex


class TestScanFilesPerformance:

    def test_mixed_files_and_dirs_at_scale(self):
        """Large number of files and subdirectories together."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(200):
                sub = root / f"dir{i:03d}"
                sub.mkdir()
                for j in range(25):
                    (sub / f"f{j:02d}.txt").touch()
            start = time.monotonic()
            result = FileIndex.scan(d).paths
            elapsed = time.monotonic() - start
            # 200 dirs + 200*25 files: well over the old 5000 cap, all kept.
            assert len(result) == 200 + 200 * 25
            assert len(result) <= _SCAN_FILES_CAP
            assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected < 2s"
            files = [p for p in result if not p.endswith("/")]
            dirs = [p for p in result if p.endswith("/")]
            assert len(files) > 0
            assert len(dirs) > 0
