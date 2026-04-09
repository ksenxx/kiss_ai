"""Performance stress tests for _scan_files with 5000+ files.

Verifies that the file picker completes quickly and respects the 5000
file cap without impacting responsiveness.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from kiss.agents.vscode.diff_merge import _scan_files


class TestScanFilesPerformance:

    def test_mixed_files_and_dirs_at_scale(self):
        """Large number of files and subdirectories together."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # 200 subdirs with 25 files each = 5000 files + 200 dir entries
            for i in range(200):
                sub = root / f"dir{i:03d}"
                sub.mkdir()
                for j in range(25):
                    (sub / f"f{j:02d}.txt").touch()
            start = time.monotonic()
            result = _scan_files(d)
            elapsed = time.monotonic() - start
            assert len(result) == 5000
            assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected < 2s"
            # Verify both files and directories are in results
            files = [p for p in result if not p.endswith("/")]
            dirs = [p for p in result if p.endswith("/")]
            assert len(files) > 0
            assert len(dirs) > 0
