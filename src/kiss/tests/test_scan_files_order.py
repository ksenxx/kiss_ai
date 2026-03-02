"""Tests for _scan_files file-before-directory ordering."""

import os
import tempfile

from kiss.agents.sorcar.code_server import _scan_files


def test_scan_files_lists_files_before_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "alpha_dir"))
        os.makedirs(os.path.join(tmp, "beta_dir"))
        open(os.path.join(tmp, "zebra.txt"), "w").close()
        open(os.path.join(tmp, "aardvark.py"), "w").close()
        result = _scan_files(tmp)
        files = [p for p in result if not p.endswith("/")]
        dirs = [p for p in result if p.endswith("/")]
        assert len(files) > 0
        assert len(dirs) > 0
        first_file_idx = result.index(files[0])
        first_dir_idx = result.index(dirs[0])
        assert first_file_idx < first_dir_idx, (
            f"First file at index {first_file_idx} should come before "
            f"first dir at index {first_dir_idx}. Result: {result}"
        )


def test_scan_files_files_before_dirs_at_each_level():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "subdir", "nested"))
        open(os.path.join(tmp, "root.txt"), "w").close()
        open(os.path.join(tmp, "subdir", "inner.txt"), "w").close()
        result = _scan_files(tmp)
        # Root level: root.txt should come before subdir/
        assert result.index("root.txt") < result.index("subdir/")


def test_scan_files_only_files():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "a.txt"), "w").close()
        open(os.path.join(tmp, "b.txt"), "w").close()
        result = _scan_files(tmp)
        assert all(not p.endswith("/") for p in result)
        assert len(result) == 2


def test_scan_files_only_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "dir1"))
        os.makedirs(os.path.join(tmp, "dir2"))
        result = _scan_files(tmp)
        assert all(p.endswith("/") for p in result)
        assert len(result) == 2
