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
    def test_5000_flat_files_completes_under_1s(self):
        """5000+ files in a flat directory should scan in under 1 second."""
        with tempfile.TemporaryDirectory() as d:
            for i in range(5500):
                (Path(d) / f"f{i:05d}.txt").touch()
            start = time.monotonic()
            result = _scan_files(d)
            elapsed = time.monotonic() - start
            assert len(result) == 5000
            assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"

    def test_deep_nested_dirs_with_many_files(self):
        """Directories nested 3 levels deep with many files total."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Create dirs: a/b/c with files at each level
            for a in range(10):
                for b in range(10):
                    for c in range(10):
                        dp = root / f"a{a}" / f"b{b}" / f"c{c}"
                        dp.mkdir(parents=True, exist_ok=True)
                        for f in range(5):
                            (dp / f"f{f}.txt").touch()
            start = time.monotonic()
            result = _scan_files(d)
            elapsed = time.monotonic() - start
            assert len(result) <= 5000
            assert elapsed < 2.0, f"Took {elapsed:.2f}s, expected < 2s"

    def test_cap_is_exactly_5000(self):
        """When more than 5000 entries exist, result is capped at 5000."""
        with tempfile.TemporaryDirectory() as d:
            for i in range(6000):
                (Path(d) / f"f{i:05d}.txt").touch()
            result = _scan_files(d)
            assert len(result) == 5000

    def test_gitignore_skips_reduce_count(self):
        """Gitignored directories are skipped, reducing total file count."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".gitignore").write_text("node_modules\nbuild\n")
            # Create non-ignored files
            for i in range(100):
                (root / f"src{i}.py").touch()
            # Create ignored dirs with many files (should be skipped)
            nm = root / "node_modules"
            nm.mkdir()
            for i in range(3000):
                (nm / f"pkg{i}.js").touch()
            bd = root / "build"
            bd.mkdir()
            for i in range(3000):
                (bd / f"out{i}.js").touch()
            start = time.monotonic()
            result = _scan_files(d)
            elapsed = time.monotonic() - start
            # Should only see the 100 src files + .gitignore, not the 6000 ignored ones
            assert len(result) < 200
            assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1s"
            # Verify no node_modules or build paths leaked through
            assert not any("node_modules" in p for p in result)
            assert not any("build" in p for p in result)

    def test_dot_dirs_skipped(self):
        """Directories starting with . are skipped."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(50):
                (root / f"visible{i}.txt").touch()
            hidden = root / ".hidden"
            hidden.mkdir()
            for i in range(5000):
                (hidden / f"h{i}.txt").touch()
            result = _scan_files(d)
            assert not any(".hidden" in p for p in result)
            assert len(result) < 100

    def test_depth_limit_pruning(self):
        """Directories deeper than the depth limit are pruned."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # The check is len(rel_root.parts) - 1 > 3, so depth 5+ is pruned.
            # a/b/c/d/e has parts len 5, 5-1=4 > 3 → pruned
            deep = root / "a" / "b" / "c" / "d" / "e"
            deep.mkdir(parents=True)
            for i in range(100):
                (deep / f"deep{i}.txt").touch()
            # Files at allowed depths
            (root / "top.txt").touch()
            (root / "a" / "mid.txt").touch()
            result = _scan_files(d)
            assert not any("deep" in p for p in result)
            assert any("top.txt" in p for p in result)

    def test_repeated_scans_consistent(self):
        """Multiple scans of the same directory produce identical results."""
        with tempfile.TemporaryDirectory() as d:
            for i in range(1000):
                (Path(d) / f"f{i:04d}.txt").touch()
            r1 = _scan_files(d)
            r2 = _scan_files(d)
            assert r1 == r2

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
