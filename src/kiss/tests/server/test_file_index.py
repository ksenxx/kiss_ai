# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the ``@``-mention file index.

Covers :mod:`kiss.server.file_index` (scanning rules, mtime-pruned
rescans, ranking, persistence, the registry's root resolution and
worker) and its wiring into ``AgentServer`` (``getFiles`` replies,
request tokens, post-task refresh, ``setWorkDir`` pre-warming).
Every test drives real directories on disk; nothing is mocked.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kiss.server import file_index
from kiss.server.file_index import (
    _BULK_DATA_MIN_FILES,
    MATCH_CAP,
    MAX_DEPTH,
    WIDE_DIR_MIN_CHILDREN,
    FileIndex,
    FileIndexRegistry,
    FileView,
)
from kiss.server.server import AgentServer
from kiss.tests.conftest import IS_WINDOWS, is_root, posix_only
from kiss.tests.server.conftest import EntryGate, local_view, touch

#: Scans the tree named on the command line with at most 256 MiB more memory than the
#: interpreter holds by then, and prints what a ``.gitignore`` reader raised, whether the
#: scan stayed within 64 MiB and its paths.  The cap makes a read that never ends fail
#: in this child instead of taking the machine's memory.
_SCAN_UNDER_A_MEMORY_CAP = """\
import os, resource, sys, threading
from kiss.server.file_index import FileIndex
raised = []
threading.excepthook = lambda args: raised.append(args.exc_type.__name__)
pages = int(open("/proc/self/statm").read().split()[0])
cap = pages * os.sysconf("SC_PAGE_SIZE") + 256 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
index = FileIndex.scan(sys.argv[1])
grown = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
print(raised, grown < 64 * 1024, sorted(index.paths))
"""


def _wait(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll *pred* until it holds or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def _texts(items: list[dict[str, str]]) -> list[str]:
    return [i["text"] for i in items]


def _registry(tmp_path: Path, home: Path) -> FileIndexRegistry:
    return FileIndexRegistry(home=str(home), cache_dir=tmp_path / "cache")


def _build(reg: FileIndexRegistry, work_dir: str) -> FileView:
    """Build the index covering *work_dir* and return its view."""
    done = threading.Event()
    reg.ensure(work_dir, done.set)
    assert done.wait(10.0)
    view = local_view(reg, work_dir)
    assert view is not None
    return view


def _persist(tmp_path: Path, root: Path) -> list[str]:
    """Index *root* with a registry that is then stopped, leaving its persisted listings.

    Returns the paths the scan produced.
    """
    reg = _registry(tmp_path, tmp_path / "home")
    try:
        return _build(reg, str(root)).paths
    finally:
        reg.stop()


def _settle(root: Path) -> None:
    """Date the directories at and below *root* an hour back, past a scan's racy window."""
    old = time.time_ns() - 3600 * 10**9
    for folder, _, _ in os.walk(root):
        os.utime(folder, ns=(old, old))


class TestScan:
    def test_lists_files_and_directories_skipping_dot_and_junk_dirs(self, tmp_path: Path) -> None:
        touch(
            tmp_path, "README.md", "src/main.py", ".env", ".git/HEAD",
            "node_modules/x/index.js", "__pycache__/m.pyc", "venv/bin/python",
            "src/site-packages/p.py",
        )
        paths = FileIndex.scan(str(tmp_path)).paths
        assert set(paths) == {"README.md", ".env", "src/", "src/main.py"}

    def test_empty_query_order_is_shallow_first_then_data_and_dotfiles_last(
        self, tmp_path: Path,
    ) -> None:
        touch(
            tmp_path, "b.py", "a.py", "run.log", ".hidden", "src/x.py", "src/y.json",
            "src/deep/z.py", "log", "src/bin",
        )
        index = FileIndex.scan(str(tmp_path))
        # ``log`` and ``bin`` are names, not suffixes: no data files.
        assert index.paths == [
            "a.py", "b.py", "log", "src/", "src/bin", "src/x.py", "src/deep/", "src/deep/z.py",
            ".hidden", "run.log", "src/y.json",
        ]
        assert index.dirs == frozenset({"", "src", "src/deep"})

    @pytest.mark.skipif(not IS_WINDOWS, reason="Windows hidden attribute and junctions")
    def test_windows_hidden_dirs_are_skipped_and_junctions_not_followed(
        self, tmp_path: Path,
    ) -> None:
        """``AppData`` and its ``Application Data`` junction must not be indexed.

        A user's home is the index root of every project below it; the
        hidden ``AppData`` holds most of a Windows home's files and its
        junctions point back into it, so following them recursed to
        ``MAX_DEPTH`` and a home scan took a minute.
        """
        touch(tmp_path, "a.py", "AppData/Local/cache.bin", "src/b.py")
        subprocess.run(["attrib", "+h", str(tmp_path / "AppData")], check=True)
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(tmp_path / "Application Data"),
             str(tmp_path / "AppData" / "Local")],
            check=True, capture_output=True,
        )
        index = FileIndex.scan(str(tmp_path))
        assert set(index.paths) == {"a.py", "Application Data", "src/", "src/b.py"}
        reg = FileIndexRegistry(home=str(tmp_path), cache_dir=tmp_path / "cache")
        below_hidden = str(tmp_path / "AppData" / "Local")
        assert reg.root_for(below_hidden) == (below_hidden, "")
        assert reg.root_for(str(tmp_path / "src")) == (str(tmp_path), "src/")

    def test_nested_gitignores_apply_to_dirs_and_files_below_them(self, tmp_path: Path) -> None:
        # To git a bracket expression (closed or not) and a backslash escape are
        # glob syntax: neither excludes the entry spelled like the pattern.
        spelled = ["item[12].py", "neg[!a]/f.py", "anch[^b].c", "open[.md"]
        spelled += [] if IS_WINDOWS else ["esc\\#.py"]  # no backslash in a Windows file name
        (tmp_path / ".gitignore").write_text(
            "build/\n/top-only\nsecret.txt\n!keep\n*.log\n/data/\n/cache/\n"
            "item[12].py\nneg[!a]/\n/anch[^b].c\nopen[.md\nesc\\#.py\n"
        )
        touch(
            tmp_path, "build/a.o", "top-only/x", "nested/top-only/y", "secret.txt",
            "nested/secret.txt", "keep", "run.log", "repo/.gitignore", "repo/tmp/junk",
            "repo/src/tmp/also-junk", "repo/generated/out.py", "repo/deeper/generated/in.py",
            "other/tmp/kept.py", "nested/build", "data", "cache/c.o", "nested/cache/k",
            "repo/gen", "repo/sub/gen/g.py", *spelled,
        )
        (tmp_path / "repo" / ".gitignore").write_text("tmp\n/generated\n/gen/\n/sub/gen/\n")
        paths = set(FileIndex.scan(str(tmp_path)).paths)
        assert "build/" not in paths and "build/a.o" not in paths
        assert "nested/build" in paths, "a `dir/` rule does not hide a file of that name"
        assert "top-only/" not in paths
        assert "nested/top-only/" in paths, "anchored entry only applies at the root"
        assert "secret.txt" not in paths and "nested/secret.txt" not in paths
        assert "keep" in paths and "run.log" in paths, "negations and globs are ignored"
        assert set(spelled) <= paths
        assert "repo/tmp/" not in paths and "repo/src/tmp/" not in paths
        assert "other/tmp/" in paths, "a nested ignore does not leak to siblings"
        assert "repo/generated/" not in paths
        assert "repo/deeper/generated/" in paths
        assert "repo/.gitignore" in paths
        # An anchored ``/dir/`` rule skips that directory only: not a
        # file of the same name (``git check-ignore data`` says so), nor
        # a directory of that name elsewhere.
        assert "cache/" not in paths and "cache/c.o" not in paths
        assert "nested/cache/" in paths
        assert "data" in paths, "an anchored `dir/` rule does not hide a file of that name"
        assert "repo/gen" in paths
        assert "repo/sub/gen/" not in paths and "repo/sub/gen/g.py" not in paths

    def test_rescan_reuses_unchanged_listings_and_sees_changes(self, tmp_path: Path) -> None:
        touch(tmp_path, "a/one.py", "b/two.py")
        _settle(tmp_path)
        first = FileIndex.scan(str(tmp_path))
        assert first.relisted
        same = FileIndex.scan(str(tmp_path), first.listings)
        assert not same.relisted
        assert same.paths == first.paths
        time.sleep(0.02)
        touch(tmp_path, "a/three.py")
        (tmp_path / "b" / "two.py").unlink()
        changed = FileIndex.scan(str(tmp_path), first.listings)
        assert changed.relisted
        assert "a/three.py" in changed.paths and "b/two.py" not in changed.paths
        assert changed.listings["a"][1] == ["one.py", "three.py"]
        assert changed.listings["b"] == (first.listings["b"][0], [], []) or (
            changed.listings["b"][1] == []
        )
        # A file created right behind a scan, in the clock tick its directory
        # was listed in, leaves the mtime where that scan saw it: not lost ...
        for n in range(10):
            touch(tmp_path, f"a/late{n}.py")
            changed = FileIndex.scan(str(tmp_path), {}, changed)
            assert f"a/late{n}.py" in changed.paths
        # ... nor with the mtime forced (a directory dated ahead of the clock
        # stays racy); listing it again without finding a change persists nothing.
        ahead = time.time_ns() + 3600 * 10**9
        os.utime(tmp_path / "a", ns=(ahead, ahead))
        os.utime(tmp_path / "b", ns=(ahead, ahead))
        changed = FileIndex.scan(str(tmp_path), {}, changed)
        touch(tmp_path, "a/late.py")
        os.utime(tmp_path / "a", ns=(ahead, ahead))
        changed = FileIndex.scan(str(tmp_path), {}, changed)
        assert "a/late.py" in changed.paths
        assert not FileIndex.scan(str(tmp_path), {}, changed).relisted

    def test_gitignore_edit_takes_effect_without_relisting(self, tmp_path: Path) -> None:
        touch(tmp_path, ".gitignore", "out/x.py", "src/y.py")
        _settle(tmp_path)
        first = FileIndex.scan(str(tmp_path))
        assert "out/" in first.paths
        (tmp_path / ".gitignore").write_text("out\n")
        second = FileIndex.scan(str(tmp_path), first.listings)
        assert "out/" not in second.paths and "src/" in second.paths
        # Nor is an edit lost that leaves the file's mtime where the last scan
        # saw it: one made in the same clock tick, or (the last two) with
        # the mtime forced, the file dated ahead of the clock.
        ahead = time.time_ns() + 3600 * 10**9
        for n, rule in enumerate(("src", "out") * 6):
            (tmp_path / ".gitignore").write_text(rule)
            if n >= 10:
                os.utime(tmp_path / ".gitignore", ns=(ahead, ahead))
            second = FileIndex.scan(str(tmp_path), {}, second)
            assert rule + "/" not in second.paths
        assert second.listings[""] is first.listings[""], "the directory was never listed again"

    @posix_only("symbolic links and file modes")
    @pytest.mark.skipif(is_root(), reason="root reads a mode-000 file")
    def test_gitignore_that_could_not_be_read_is_read_once_it_can(self, tmp_path: Path) -> None:
        # A link without its target, a file without the permission: when the
        # reason passes no mtime the scan compares has moved, so the failed
        # read must not count as a read.  (A ``stat`` that fails while the
        # read after it succeeds needs a stall between the two: not staged.)
        root = tmp_path / "root"
        touch(root, "linked/out/x.py", "locked/out/x.py", "locked/.gitignore")
        (root / "linked" / ".gitignore").symlink_to(tmp_path / "rules")
        locked = root / "locked" / ".gitignore"
        locked.write_text("out\n")
        locked.chmod(0o000)
        os.utime(locked, ns=(0, 0))
        _settle(root)
        first = FileIndex.scan(str(root))
        assert {"linked/out/", "locked/out/"} <= set(first.paths)
        (tmp_path / "rules").write_text("out\n")
        locked.chmod(0o644)
        second = FileIndex.scan(str(root), {}, first)
        assert "linked/out/" not in second.paths and "locked/out/" not in second.paths

    @posix_only("descriptor limits")
    def test_directory_that_could_not_be_listed_is_listed_once_it_can(self, tmp_path: Path) -> None:
        # With no descriptor left ``scandir`` fails (EMFILE), and that passes:
        # an empty listing recorded for the directory would outlive it.
        import resource

        touch(tmp_path, "src/a.py")
        _settle(tmp_path)
        first = FileIndex.scan(str(tmp_path))
        changed = first.listings["src"][0] + 60 * 10**9  # touched since, and quiet again
        os.utime(tmp_path / "src", ns=(changed, changed))
        limits = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (0, limits[1]))
        try:
            during = FileIndex.scan(str(tmp_path), {}, first)
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, limits)
        assert during.paths == ["src/"] and "src" not in during.listings
        assert FileIndex.scan(str(tmp_path), {}, during).paths == ["src/", "src/a.py"]

    def test_directories_below_max_depth_are_listed_but_not_descended(
        self, tmp_path: Path,
    ) -> None:
        deep = "/".join(f"d{i}" for i in range(MAX_DEPTH + 1))
        touch(tmp_path, f"{deep}/leaf.py")
        paths = FileIndex.scan(str(tmp_path)).paths
        at_max = "/".join(f"d{i}" for i in range(MAX_DEPTH))
        assert f"{at_max}/d{MAX_DEPTH}/" in paths
        assert f"{deep}/leaf.py" not in paths

    def test_entry_cap_bounds_the_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        touch(tmp_path, *[f"d{i // 10}/f{i}.py" for i in range(100)])
        monkeypatch.setattr(file_index, "_SCAN_FILES_CAP", 25)
        index = FileIndex.scan(str(tmp_path))
        assert len(index.paths) == 25
        assert set(index.paths) >= {f"d{i}/" for i in range(10)}, "shallow entries come first"

    def test_bulk_data_directories_collapse_to_their_entry(self, tmp_path: Path) -> None:
        dumps = [f"bench/results/r{i}.json" for i in range(_BULK_DATA_MIN_FILES)]
        touch(tmp_path, "bench/run.py", *dumps)
        index = FileIndex.scan(str(tmp_path))
        assert "bench/results/" in index.paths
        assert not any(p.startswith("bench/results/r") for p in index.paths)
        assert "bench/run.py" in index.paths
        assert "bench/results" not in index.dirs, "a collapsed directory cannot serve a tab"
        assert "bench" in index.dirs

    def test_root_is_never_a_wide_container(self, tmp_path: Path) -> None:
        touch(
            tmp_path, "src/deep/test_cli.py",
            *[f"pkg{i:02d}/mod.py" for i in range(WIDE_DIR_MIN_CHILDREN)],
            *[f"pkg00/runs/run{i:03d}/test_cli.py" for i in range(WIDE_DIR_MIN_CHILDREN)],
        )
        index = FileIndex.scan(str(tmp_path))
        assert index.wide_from == index.paths.index("pkg00/runs/run000/")
        assert _texts(index.view("").search("test_cli", {}))[0] == "src/deep/test_cli.py"

    def test_root_is_never_a_bulk_dump(self, tmp_path: Path) -> None:
        touch(tmp_path, *[f"r{i}.json" for i in range(_BULK_DATA_MIN_FILES)])
        assert len(FileIndex.scan(str(tmp_path)).paths) == _BULK_DATA_MIN_FILES

    def test_wide_container_contents_rank_last(self, tmp_path: Path) -> None:
        touch(
            tmp_path, "src/test_a.py",
            *[f"artifacts/run{i:03d}/test_a.py" for i in range(WIDE_DIR_MIN_CHILDREN)],
        )
        index = FileIndex.scan(str(tmp_path))
        view = index.view("")
        texts = _texts(view.search("test_a", {}, limit=1000))
        assert texts[0] == "src/test_a.py"
        assert len(texts) == WIDE_DIR_MIN_CHILDREN + 1
        assert index.paths.index("artifacts/") < index.paths.index("artifacts/run000/test_a.py")
        assert index.paths.index("src/test_a.py") < index.paths.index("artifacts/run000/")

    def test_unreadable_root_yields_empty_index(self, tmp_path: Path) -> None:
        index = FileIndex.scan(str(tmp_path / "missing"))
        assert index.paths == [] and index.listings == {}
        assert FileIndex.empty().dirs == frozenset({""})

    def test_view_of_subdirectory_strips_prefix_and_excludes_itself(self, tmp_path: Path) -> None:
        touch(tmp_path, "top.py", "kiss/README.md", "kiss/src/a.py", "kissing/b.py")
        index = FileIndex.scan(str(tmp_path))
        view = index.view("kiss/")
        assert view.paths == ["README.md", "src/", "src/a.py"]
        assert index.view("kiss/") is view, "views are cached per index"
        assert index.view("").paths is index.paths

    @posix_only("directory permissions (Windows ACLs are not set with chmod)")
    @pytest.mark.skipif(is_root(), reason="root lists a mode-000 directory")
    def test_refused_directory_is_a_bare_entry_and_is_never_persisted(
        self, tmp_path: Path,
    ) -> None:
        """A directory the OS will not let the daemon list stays refused for the process.

        macOS answers a daemon without the "Files and Folders" permission
        with a five-second wait per ``scandir`` of ``~/Desktop``,
        ``~/Downloads`` and dozens of ``~/Library`` directories; paying it
        on every rescan kept the index worker busy for minutes.  The
        refusal (here ``EACCES`` from a mode-000 directory) is remembered,
        the directory is indexed as its bare ``dir/`` entry, and no
        listing of it is kept: an empty listing persisted with the
        directory's mtime would stand for its contents for ever, even
        after the permission is granted.
        """
        touch(tmp_path, "a.py", "locked/secret.py", "locked/sub/deep.py")
        _settle(tmp_path)
        locked = tmp_path / "locked"
        locked.chmod(0o000)
        try:
            first = FileIndex.scan(str(tmp_path))
            assert first.paths == ["a.py", "locked/"]
            assert str(locked) in file_index._refused
            assert "locked" not in first.listings
            assert first.relisted, "the root itself was read"
            # Still refused, answered without a system call, and nothing
            # new to persist.
            second = FileIndex.scan(str(tmp_path), first.listings)
            assert second.paths == ["a.py", "locked/"]
            assert "locked" not in second.listings and not second.relisted
            locked.chmod(0o755)
            same = FileIndex.scan(str(tmp_path), second.listings)
            assert same.paths == ["a.py", "locked/"], "refused until the daemon restarts"
            # Nor does a listing another scan cached while the directory
            # was readable bring it back: the refusal is checked first.
            stale_cache = dict(first.listings)
            stale_cache["locked"] = (os.stat(locked).st_mtime_ns, ["secret.py"], ["sub"])
            from_cache = FileIndex.scan(str(tmp_path), stale_cache)
            assert from_cache.paths == ["a.py", "locked/"]
            assert "locked" not in from_cache.listings
            assert from_cache.relisted, "the dropped listing makes the persisted copy stale"
            trusted = FileIndex.scan(str(tmp_path), stale_cache, trusted=True)
            assert trusted.paths == ["a.py", "locked/"]
            file_index._refused.discard(str(locked))  # what a restart does
            granted = FileIndex.scan(str(tmp_path), same.listings)
            assert granted.paths == [
                "a.py", "locked/", "locked/secret.py", "locked/sub/", "locked/sub/deep.py",
            ]
            assert granted.listings["locked"][1] == ["secret.py"]
        finally:
            locked.chmod(0o755)
            file_index._refused.discard(str(locked))

    @posix_only("named pipes")
    def test_gitignore_that_is_no_ignore_file_is_left_unread(self, tmp_path: Path) -> None:
        """A ``.gitignore`` that is a pipe, or longer than any ignore file, is skipped at once.

        The name may be anything, or a link to anything.  The open of a
        pipe nobody writes waited out the read deadline, after which the
        directory was refused for good; a file of any length was read
        whole.  Now the file is opened without waiting and read only when
        it is a regular file of at most ``_GITIGNORE_MAX_BYTES`` (``full``
        is one, and its rule holds): its directory stays indexed, under
        the rules above it, and no reader thread is left behind.
        """
        touch(tmp_path, "pipe/out/x.py", "huge/out/x.py", "full/out/x.py", "binary/out/x.py")
        os.mkfifo(tmp_path / "pipe" / ".gitignore")
        (tmp_path / "binary" / ".gitignore").write_bytes(b"out\n\xff\n")  # not UTF-8
        for name, size in ("huge", file_index._GITIGNORE_MAX_BYTES + 1), (
            "full", file_index._GITIGNORE_MAX_BYTES,
        ):
            with open(tmp_path / name / ".gitignore", "wb") as ignore:
                ignore.write(b"out\n")
                ignore.truncate(size)  # the rest is a hole: no disk is used
        threads = threading.active_count()
        started = time.monotonic()
        paths = set(FileIndex.scan(str(tmp_path)).paths)
        assert time.monotonic() - started < file_index._READ_DEADLINE, "the scan waited for a pipe"
        assert {"pipe/out/x.py", "huge/out/x.py", "binary/out/x.py"} <= paths
        assert "full/out/" not in paths, "an ignore file of the longest length is still read"
        assert not {str(tmp_path / name) for name in ("pipe", "huge")} & file_index._refused
        assert threading.active_count() == threads, "a reader thread was left behind"

    @pytest.mark.skipif(sys.platform != "linux", reason="the child's memory is capped via /proc")
    def test_gitignore_linked_to_an_endless_device_costs_no_memory(self, tmp_path: Path) -> None:
        """A ``.gitignore`` that is a link to ``/dev/zero`` is not read.

        A clone can carry such a link.  Its read never ended: it took
        100 MiB in 40 ms and went on after the read deadline, until the
        daemon had no memory left.  The scan runs in a child whose
        memory is capped, so that a read without end fails there.
        """
        touch(tmp_path, "proj/out/x.py")
        (tmp_path / "proj" / ".gitignore").symlink_to("/dev/zero")
        child = subprocess.run(
            [sys.executable, "-c", _SCAN_UNDER_A_MEMORY_CAP, str(tmp_path)],
            capture_output=True, text=True, timeout=120,
        )
        listed = ["proj/", "proj/.gitignore", "proj/out/", "proj/out/x.py"]
        assert child.stdout.strip() == f"[] True {listed}", child.stdout + child.stderr

    def test_gitignore_read_that_never_returns_refuses_the_directory(
        self, tmp_path: Path,
    ) -> None:
        """A ``.gitignore`` read the OS holds for ever turns into a refusal.

        The daemon's home index is reloaded from persisted listings, so a
        directory inside a protected tree is reached without being
        listed in this process, and the first thing read below it is
        its ``.gitignore``.  macOS holds that ``open`` as long as the
        permission prompt the daemon cannot show stays unanswered: the
        index worker never returned and the picker never opened again.
        No file holds an ``open`` that does not wait where these tests
        run (a pipe is told apart without waiting, see above), so a
        profile gate holds the reader thread at its entry instead; the
        reader itself is not replaced.  The read runs on a
        helper thread with a deadline; when it passes the directory is
        refused: a bare ``dir/`` entry, nothing below it, no listing kept,
        and no further wait on the next scan.
        """
        touch(tmp_path, "a.py", "proj/main.py", "proj/sub/x.py", "proj/.gitignore")
        held = EntryGate(file_index._read_file_into.__code__)
        proj = str(tmp_path / "proj")
        # The listing the daemon reloaded, which lists ``.gitignore`` as a
        # plain file: the scan must read it to learn the rules below.
        listings: file_index.Listings = {
            "": (os.stat(tmp_path).st_mtime_ns, ["a.py"], ["proj"]),
            "proj": (os.stat(proj).st_mtime_ns, [".gitignore", "main.py"], ["sub"]),
            "proj/sub": (os.stat(tmp_path / "proj" / "sub").st_mtime_ns, ["x.py"], []),
        }
        threading.setprofile(held)  # carried by the reader thread the scan starts
        try:
            started = time.monotonic()
            index = FileIndex.scan(str(tmp_path), listings)
            elapsed = time.monotonic() - started
            assert held.reached.is_set(), "the scan read no .gitignore"
            assert file_index._READ_DEADLINE <= elapsed < file_index._READ_DEADLINE + 3.0
            assert index.paths == ["a.py", "proj/"]
            assert proj in file_index._refused
            assert "proj" not in index.listings
            assert "" in index.listings
            assert index.relisted, "the cached listing was dropped: persist that"
            started = time.monotonic()
            again = FileIndex.scan(str(tmp_path), index.listings)
            assert time.monotonic() - started < 1.0, "a refused directory is not read again"
            assert again.paths == ["a.py", "proj/"]
            assert not again.relisted
        finally:
            threading.setprofile(None)
            held.release.set()  # the reader thread reads the file and ends
            file_index._refused.discard(proj)


class TestSearch:
    def test_case_insensitive_substring_ranked_by_match_position(self) -> None:
        view = FileView(["docs/README.md", "README.md", "src/readme_gen.py", "x.py"])
        assert _texts(view.search("readme", {})) == [
            "docs/README.md", "README.md", "src/readme_gen.py",
        ]
        assert _texts(view.search("ReAdMe.MD", {})) == ["docs/README.md", "README.md"]
        assert view.search("nomatch", {}) == []
        assert view.search("a\nb", {}) == []
        # A lone surrogate (a client's JSON may carry one, as may an NTFS name) is text
        # like any other: it matches the names that hold it and no others.
        assert view.search("read\ud83d", {}) == []
        odd = FileView(["a\ud83d.txt", "b.txt"])
        assert "a\ud83d.txt" in odd and _texts(odd.search("\ud83d", {})) == ["a\ud83d.txt"]

    def test_empty_query_keeps_index_order_and_limit(self) -> None:
        view = FileView([f"f{i}.py" for i in range(30)])
        assert _texts(view.search("", {})) == [f"f{i}.py" for i in range(20)]
        assert _texts(view.search("", {}, limit=3)) == ["f0.py", "f1.py", "f2.py"]

    def test_frequent_entries_come_first_by_position_recency_and_count(self) -> None:
        view = FileView(["a/old.py", "b/new.py", "c/big.py", "d/other.py", "e.py"])
        usage = {"a/old.py": 5, "elsewhere.py": 9, "c/big.py": 7, "b/new.py": 1}
        items = view.search("", usage)
        assert items[:3] == [
            {"type": "frequent", "text": "b/new.py"},
            {"type": "frequent", "text": "c/big.py"},
            {"type": "frequent", "text": "a/old.py"},
        ]
        assert _texts(items[3:]) == ["d/other.py", "e.py"]
        assert all(i["type"] == "file" for i in items[3:])
        assert _texts(view.search("big", usage)) == ["c/big.py"]
        assert _texts(view.search("", {"e.py": 0}))[0] == "a/old.py", "zero counts are not frequent"
        assert _texts(view.search("", usage, limit=2)) == ["b/new.py", "c/big.py"]

    def test_match_cap_bounds_the_candidates_but_keeps_close_matches(self) -> None:
        paths = [f"dir{i:05d}/x_file.py" for i in range(MATCH_CAP + 50)]
        paths.append("late/x.py")
        view = FileView(paths)
        texts = _texts(view.search("x", {}))
        assert texts[0] == "dir00000/x_file.py"
        assert "late/x.py" not in texts, "beyond MATCH_CAP hits the query is unspecific"
        assert _texts(view.search("late/x", {})) == ["late/x.py"]

    def test_contains_requires_the_exact_spelling(self) -> None:
        view = FileView(["src/App.py", "src/", "README.md", "readme.md"])
        assert "src/App.py" in view and "src/" in view
        assert "src/app.py" not in view, "a usage row from a case-only rename is not an entry"
        assert "README.md" in view and "readme.md" in view
        assert "App.py" not in view and "src/App" not in view
        assert _texts(view.search("readme", {"Readme.md": 3})) == ["README.md", "readme.md"]

    def test_wide_container_entries_rank_after_every_other_match(self) -> None:
        view = FileView(["src/tests/test_long_name.py", "runs/r1/test.py", "runs/r2/test.py"], 1)
        assert _texts(view.search("test", {})) == [
            "src/tests/test_long_name.py", "runs/r1/test.py", "runs/r2/test.py",
        ]
        assert _texts(view.search("test", {}, limit=1)) == ["src/tests/test_long_name.py"]
        top_two = ["src/tests/test_long_name.py", "runs/r1/test.py"]
        assert _texts(view.search("", {}, limit=2)) == top_two
        many = FileView([f"src/f{i}.py" for i in range(30)] + ["runs/r/f.py"], 30)
        assert _texts(many.search("f", {})) == [f"src/f{i}.py" for i in range(20)]
        # Inside the wide region every hit still competes: the best one is
        # not among the first *limit* in index order.
        weak = [f"runs/{i:02d}/config_old.txt" for i in range(25)] + ["runs/25/config"]
        assert _texts(FileView(weak, 0).search("config", {}))[0] == "runs/25/config"
        mixed = FileView(["src/config.py", *weak], 1)
        assert _texts(mixed.search("config", {}))[:2] == ["src/config.py", "runs/25/config"]

    def test_non_ascii_names_are_searchable(self, tmp_path: Path) -> None:
        touch(tmp_path, "docs/résumé.md", "plain.py")
        index = FileIndex.scan(str(tmp_path))
        view = index.view("")
        assert "docs/résumé.md" in view.paths and "docs/résumé.md" in view
        assert _texts(view.search("RÉSUMÉ", {})) == ["docs/résumé.md"]
        assert _texts(index.view("docs/").search("", {})) == ["résumé.md"]

    @posix_only("non-UTF-8 file names (NTFS names are UTF-16)")
    def test_undecodable_names_are_searchable(self, tmp_path: Path) -> None:
        touch(tmp_path, "plain.py")
        raw = os.path.join(os.fsencode(tmp_path), b"bad_\xff.py")
        try:
            with open(raw, "wb"):
                pass
        except OSError as exc:  # APFS (macOS) refuses names that are not valid UTF-8
            if exc.errno != errno.EILSEQ:
                raise
            pytest.skip("filesystem rejects non-UTF-8 file names")
        view = FileIndex.scan(str(tmp_path)).view("")
        bad = os.fsdecode(b"bad_\xff.py")
        assert bad in view.paths and bad in view
        assert _texts(view.search("bad_", {})) == [bad]


class TestRegistry:
    def test_root_resolution(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        touch(home, "proj/a.py", "proj/.gitignore", "proj/ignored/b.py", ".hidden/c.py")
        (home / "proj" / ".gitignore").write_text("ignored\n")
        reg = _registry(tmp_path, home)
        assert reg.root_for(str(home)) == (str(home), "")
        assert reg.root_for(str(home / "proj")) == (str(home), "proj/")
        assert reg.root_for(str(home / "proj" / "ignored")) == (str(home), "proj/ignored/"), (
            "before the home index exists the gitignore cannot be known"
        )
        assert reg.root_for(str(home / ".hidden")) == (str(home / ".hidden"), "")
        assert reg.root_for(str(home / "proj" / "node_modules")) == (
            str(home / "proj" / "node_modules"), "",
        )
        assert reg.root_for(str(tmp_path / "elsewhere")) == (str(tmp_path / "elsewhere"), "")
        assert reg.root_for("/") == (str(home), "")
        assert reg.root_for(str(home) + "/proj/../proj") == (str(home), "proj/")
        _build(reg, str(home))
        assert reg.root_for(str(home / "proj" / "ignored")) == (
            str(home / "proj" / "ignored"), "",
        )
        assert reg.root_for(str(home / "proj")) == (str(home), "proj/")
        reg.stop()

    def test_ensure_builds_in_background_and_refresh_rescans(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        touch(home, "proj/a.py")
        reg = _registry(tmp_path, home)
        assert local_view(reg, str(home / "proj")) is None
        view = _build(reg, str(home / "proj"))
        assert view.paths == ["a.py"]
        reg.refresh(str(tmp_path / "unknown"))
        assert local_view(reg, str(tmp_path / "unknown")) is None, "refresh never indexes new roots"
        time.sleep(0.02)
        touch(home, "proj/b.py")
        reg.refresh(str(home / "proj"))
        _wait(lambda: (local_view(reg, str(home / "proj")) or view).paths == ["a.py", "b.py"])
        reg.stop()

    def test_stale_view_schedules_a_refresh(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")
        _build(reg, str(root))
        time.sleep(0.02)
        touch(root, "b.py")
        assert local_view(reg, str(root)).paths == ["a.py"]  # type: ignore[union-attr]
        reg._indexes[str(root)].built_at -= file_index.STALE_AFTER + 1
        assert local_view(reg, str(root)).paths == ["a.py"], "the stale view is served at once"  # type: ignore[union-attr]
        _wait(lambda: local_view(reg, str(root)).paths == ["a.py", "b.py"])  # type: ignore[union-attr]
        reg.stop()

    def test_listings_persist_across_registries(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py", "sub/b.py")
        _settle(root)
        first = _registry(tmp_path, tmp_path / "home")
        _build(first, str(root))
        first.stop()
        cache_files = list((tmp_path / "cache").glob("*.json"))
        assert len(cache_files) == 1
        data = json.loads(cache_files[0].read_text())
        assert data["root"] == str(root) and set(data["listings"]) == {"", "sub"}

        second = _registry(tmp_path, tmp_path / "home")
        view = _build(second, str(root))
        assert view.paths == ["a.py", "sub/", "sub/b.py"]
        assert not second._indexes[str(root)].relisted, "unchanged tree reused persisted listings"
        second.stop()

        for junk in ("{not json", "[]", f'{{"version": 1, "root": "{root}", "listings": []}}'):
            cache_files[0].write_text(junk)
            third = _registry(tmp_path, tmp_path / "home")
            assert _build(third, str(root)).paths == ["a.py", "sub/", "sub/b.py"]
            assert third._indexes[str(root)].relisted
            third.stop()

    def test_persisted_listings_of_another_root_or_version_are_ignored(
        self, tmp_path: Path,
    ) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")
        path = reg._cache_path(str(root))
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 999, "root": str(root), "listings": {}}))
        assert file_index._load_listings(path, str(root)) == {}
        path.write_text(json.dumps({"version": 1, "root": "/other", "listings": {}}))
        assert file_index._load_listings(path, str(root)) == {}
        truncated = {"version": 1, "root": str(root), "listings": {"": [1, ["a.py"]]}}
        path.write_text(json.dumps(truncated))
        assert file_index._load_listings(path, str(root)) == {}
        file_index._save_listings(tmp_path / "nodir" / "x" / "c.json", str(root), {})
        assert (tmp_path / "nodir" / "x" / "c.json").exists()
        file_index._save_listings(root / "a.py" / "c.json", str(root), {})

    def test_failed_scan_installs_empty_index_and_still_calls_back(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path, tmp_path / "home")
        missing = tmp_path / "missing"
        view = _build(reg, str(missing))
        assert view.paths == []
        reg.stop()

    def test_duplicate_refreshes_are_coalesced(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")
        _build(reg, str(root))
        # Block the worker so the refreshes queue up behind this job.
        gate = threading.Event()
        reg.ensure(str(root), gate.wait)
        for _ in range(5):
            reg.refresh(str(root))
        with caplog.at_level(logging.INFO, logger="kiss.server.file_index"):
            gate.set()
            done = threading.Event()
            reg.ensure(str(root), done.set)
            assert done.wait(10.0)
        builds = [r for r in caplog.records if r.getMessage().startswith("file index of")]
        assert len(builds) == 1, [r.getMessage() for r in builds]
        reg.stop()

    def test_stale_requests_during_a_refresh_do_not_rescan_again(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Picker requests keep arriving while the refresh a stale index
        # triggered is still walking the tree; the index they are served
        # is still the stale one, which must not queue a second walk
        # behind the running one (nobody asked for newer content).
        root = tmp_path / "root"
        touch(root, *(f"d{i}/f.py" for i in range(2000)))
        reg = _registry(tmp_path, tmp_path / "home")
        _build(reg, str(root))
        stale = reg._indexes[str(root)]
        stale.built_at -= file_index.STALE_AFTER + 1

        with caplog.at_level(logging.INFO, logger="kiss.server.file_index"):
            deadline = time.monotonic() + 10.0
            while reg._indexes[str(root)] is stale:
                assert time.monotonic() < deadline, "no refresh"
                reg.picker_for(str(root))
                time.sleep(0.001)
            # Queued behind any job the requests above left: when it
            # has run, so has every rescan of ``root``.
            idle = threading.Event()
            reg.ensure(str(tmp_path / "other"), idle.set)
            assert idle.wait(10.0)
        builds = [r for r in caplog.records if r.getMessage().startswith(f"file index of {root}")]
        assert len(builds) == 1, [r.getMessage() for r in builds]
        reg.stop()

    def test_callback_exception_does_not_kill_the_worker(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")

        def boom() -> None:
            raise RuntimeError("callback failed")

        reg.ensure(str(root), boom)
        assert _build(reg, str(root)).paths == ["a.py"]
        reg.stop()

    def test_stop_ignores_later_requests_and_queued_callbacks(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")
        gate = threading.Event()
        reg.ensure(str(root), gate.wait)
        ran = threading.Event()
        assert reg.ensure(str(root), ran.set)
        reg.stop()
        gate.set()
        reg._worker.join(5.0)  # type: ignore[union-attr]
        assert not ran.is_set(), "a job queued before stop() must not run after it"
        assert reg.ensure(str(tmp_path), lambda: pytest.fail("must not run")) is False
        assert local_view(reg, str(tmp_path)) is None
        reg.stop()

    def test_refresh_queued_during_a_scan_is_not_skipped(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")
        _build(reg, str(root))
        index = reg._indexes[str(root)]
        # A refresh queued while the scan that produced ``index`` was
        # already running must not be treated as satisfied by it.
        reg._jobs.put((str(root), index.built_at + 1e-6, None))
        time.sleep(0.02)
        touch(root, "b.py")
        done = threading.Event()
        reg.ensure(str(root), done.set)
        assert done.wait(10.0)
        assert reg._indexes[str(root)] is not index

    @posix_only("named pipes")
    def test_refresh_absorbed_by_a_job_older_than_the_running_scan_is_not_lost(
        self, tmp_path: Path,
    ) -> None:
        # A FIFO in place of the persisted listings holds the worker between
        # taking the first job and starting its scan, where the second job is
        # queued: older than that scan, it would be skipped as satisfied by
        # it, but the refresh it absorbs comes after the scan.
        root = tmp_path / "root"
        touch(root, "a.py")
        reg = _registry(tmp_path, tmp_path / "home")
        fifo = reg._cache_path(str(root))
        fifo.parent.mkdir()
        os.mkfifo(fifo)

        def change_behind_the_scan() -> None:
            touch(root, "b.py")
            reg.refresh(str(root))

        try:
            reg.ensure(str(root), change_behind_the_scan)
            fd = None
            deadline = time.monotonic() + 10.0
            while fd is None:
                assert time.monotonic() < deadline, "the worker never read the listings"
                try:
                    fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                except OSError:  # ENXIO until the worker's read has opened it
                    time.sleep(0.01)
            reg.ensure(str(root))
            os.close(fd)  # end of file: no listings, so the scan starts
            idle = threading.Event()
            reg.ensure(str(tmp_path / "other"), idle.set)
            assert idle.wait(10.0)
            assert local_view(reg, str(root)).paths == ["a.py", "b.py"]  # type: ignore[union-attr]
        finally:
            reg.stop()

    def test_default_cache_dir_follows_kiss_home(self, tmp_path: Path) -> None:
        from kiss.core.config import kiss_home

        reg = FileIndexRegistry(home=str(tmp_path))
        assert reg._cache_path("/x").parent == kiss_home() / "file-index"

    def test_first_build_serves_persisted_listings_then_rescans(self, tmp_path: Path) -> None:
        """After a restart the picker answers from the persisted listings at once.

        The rescan that follows may wait on the OS (five seconds per
        protected directory on macOS, minutes on a network share); the
        first ``@`` of the session must not.  The index built from the
        listings alone is published first and the callback runs; the
        rescan is queued behind it and publishes the fresh index, with
        the ``.gitignore`` rules the trusted build could not read.
        """
        root = tmp_path / "root"
        touch(root, "old.py", "proj/.gitignore", "proj/out/gen.py", "proj/src/a.py")
        (root / "proj" / ".gitignore").write_text("out\n")
        assert "proj/out/" not in _persist(tmp_path, root)
        time.sleep(0.02)
        touch(root, "new.py")
        (root / "old.py").unlink()

        second = _registry(tmp_path, tmp_path / "home")
        served: list[FileIndex] = []

        def on_ready() -> None:
            # On the worker thread, before it takes the next job: what the
            # callback of the first request is answered from.
            served.append(second._indexes[str(root)])

        try:
            second.ensure(str(root), on_ready)
            _wait(lambda: bool(served))
            trusted = served[0]
            assert not trusted.relisted, "nothing was read for the first index"
            assert trusted.built_at == float("-inf"), "stale from birth"
            assert "old.py" in trusted.paths and "new.py" not in trusted.paths
            assert "proj/src/a.py" in trusted.paths
            _wait(lambda: second._indexes[str(root)] is not trusted)
            fresh = second._indexes[str(root)]
            assert fresh.relisted
            assert "new.py" in fresh.paths and "old.py" not in fresh.paths
            assert "proj/out/" not in fresh.paths, "the rescan read the .gitignore"
            assert "proj/src/a.py" in fresh.paths
            assert fresh.built_at > 0
        finally:
            second.stop()

    def test_request_queued_while_listings_load_still_rescans(self, tmp_path: Path) -> None:
        """A request that arrives while the persisted listings load gets a real scan.

        The worker has taken the first request off ``_pending`` when the
        second one arrives, so the second is queued with the root pending
        again, and the rescan the trusted publish asks for folds into it.
        That second job was queued before the trusted index existed; the
        worker skips a job older than the index it finds, so the trusted
        index must count as older than any request (``built_at`` -inf)
        or the second request would be answered from the listings and no
        real scan would run until the index went stale.
        """
        root = tmp_path / "root"
        touch(root, "old.py")
        _persist(tmp_path, root)
        time.sleep(0.02)
        touch(root, "new.py")
        (root / "old.py").unlink()

        second = _registry(tmp_path, tmp_path / "home")
        gate = threading.Event()
        answered: list[FileIndex] = []

        def on_ready() -> None:
            answered.append(second._indexes[str(root)])

        try:
            # Hold the worker so the two requests below queue up behind
            # this job: the second stands in for a request that arrived
            # while the listings loaded, stamped before the trusted index.
            second.ensure(str(tmp_path / "other"), gate.wait)
            second.ensure(str(root), on_ready)
            second._jobs.put((str(root), time.monotonic(), on_ready))
            gate.set()
            _wait(lambda: len(answered) == 2)
            assert "old.py" in answered[0].paths, "the first request is answered from the listings"
            assert "new.py" in answered[1].paths and "old.py" not in answered[1].paths, (
                "the second request waited for the real scan"
            )
            idle = threading.Event()
            second.ensure(str(tmp_path / "idle"), idle.set)
            assert idle.wait(10.0)
            assert second._indexes[str(root)] is answered[1], "no further rescan was needed"
        finally:
            gate.set()
            second.stop()

    def test_persisted_listing_of_a_refused_directory_is_dropped_from_the_cache(
        self, tmp_path: Path,
    ) -> None:
        """A refusal recorded before the listings load is written back to the cache.

        The trusted build drops the refused directory's listing; the rescan
        starts from that filtered index and finds nothing else to persist,
        so the trusted build itself has to save, or the persisted cache
        would keep (and every later start would briefly publish) the
        refused directory's contents.
        """
        root = tmp_path / "root"
        touch(root, "a.py", "locked/secret.py")
        # Spelled as the scanner spells a sub-directory, with the host's separator.
        locked = str(root / "locked")
        assert "locked/secret.py" in _persist(tmp_path, root)
        file_index._refused.add(locked)  # what a scan of another root left behind
        reg = _registry(tmp_path, tmp_path / "home")
        served: list[FileIndex] = []

        def on_ready() -> None:
            served.append(reg._indexes[str(root)])

        try:
            reg.ensure(str(root), on_ready)
            _wait(lambda: bool(served))
            assert served[0].paths == ["a.py", "locked/"]
            _wait(lambda: reg._indexes[str(root)] is not served[0])
            assert reg._indexes[str(root)].paths == ["a.py", "locked/"]
            cache = json.loads(reg._cache_path(str(root)).read_text())
            assert set(cache["listings"]) == {""}, "the refused listing is gone from disk"
        finally:
            file_index._refused.discard(locked)
            reg.stop()

    def test_home_index_skips_the_macos_library_tree(self, tmp_path: Path) -> None:
        """On macOS ``~/Library`` is left out of the home index, like Windows' ``AppData``.

        It is the application state tree (a third of a home directory's
        entries here, dozens of them behind the privacy protection that
        stalls a daemon's ``scandir`` for five seconds each).  A
        ``Library`` directory anywhere else is an ordinary directory, and
        a work dir inside the skipped tree (iCloud Drive) is indexed as a
        root of its own, the way a dot-directory is.
        """
        home = tmp_path / "home"
        touch(
            home, "work/a.py", "work/Library/keep.py",
            "Library/Application Support/app/state.plist",
            "Library/Mobile Documents/com~apple~CloudDocs/notes.md",
        )
        skipped = sys.platform == "darwin"
        reg = _registry(tmp_path, home)
        paths = _build(reg, str(home)).paths
        assert ("Library/" in paths) is not skipped
        assert ("Library/Application Support/app/state.plist" in paths) is not skipped
        assert "work/Library/" in paths and "work/Library/keep.py" in paths
        assert reg.root_for(str(home / "work")) == (str(home), "work/")
        icloud = str(home / "Library" / "Mobile Documents")
        if skipped:
            assert reg.root_for(icloud) == (icloud, "")
            assert _build(reg, icloud).paths == [
                "com~apple~CloudDocs/", "com~apple~CloudDocs/notes.md",
            ]
        else:
            assert reg.root_for(icloud) == (str(home), "Library/Mobile Documents/")
        reg.stop()
        # Outside the home root the name means nothing (a Unity or Xcode
        # project keeps its own ``Library``).
        touch(tmp_path, "proj/Library/x.py", "proj/main.py")
        assert "Library/x.py" in FileIndex.scan(str(tmp_path / "proj")).paths


class _Server:
    """A ``AgentServer`` with a recording printer and a private registry."""

    def __init__(self, tmp_path: Path, home: Path, work_dir: str) -> None:
        self.server = AgentServer()
        self.server.work_dir = work_dir
        self.server._file_index.stop()
        self.server._file_index = _registry(tmp_path, home)
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.server.printer.broadcast = self._record  # type: ignore[method-assign]

    def _record(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(dict(event))

    def files_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e.get("type") == "files"]

    def populated(self) -> list[dict[str, Any]]:
        return [e for e in self.files_events() if not e.get("loading")]

    def get_files(
        self, prefix: str, work_dir: str = "", conn_id: str = "c1", tab_id: str = "t1",
    ) -> None:
        self.server._handle_command({
            "type": "getFiles", "prefix": prefix, "workDir": work_dir,
            "connId": conn_id, "tabId": tab_id,
        })

    def wait_populated(self, n: int = 1) -> dict[str, Any]:
        _wait(lambda: len(self.populated()) >= n)
        return self.populated()[n - 1]

    def stop(self) -> None:
        self.server._file_index.stop()


class TestServerWiring:
    def test_cold_request_emits_loading_then_populated_reply(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "alpha.py", "sub/beta.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        s.get_files("be", str(root))
        first = s.files_events()[0]
        assert first["loading"] is True and first["files"] == []
        assert first["prefix"] == "be" and first["connId"] == "c1" and first["tabId"] == "t1"
        reply = s.wait_populated()
        assert _texts(reply["files"]) == ["./sub/beta.py"]
        assert reply["prefix"] == "be" and reply["connId"] == "c1" and reply["tabId"] == "t1"
        assert s.server._files_latest_request == {}
        s.stop()

    def test_cold_request_on_a_stopped_registry_releases_its_token(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        s.server._file_index.stop()
        s.get_files("", str(root))
        assert [e["loading"] for e in s.files_events()] == [True]
        assert s.server._files_latest_request == {}, "nobody will answer, so no token may linger"

    def test_warm_request_is_answered_synchronously(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "alpha.py", "sub/beta.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        _build(s.server._file_index, str(root))
        s.get_files("", str(root))
        assert [e.get("loading") for e in s.files_events()] == [None]
        assert _texts(s.files_events()[0]["files"]) == ["./alpha.py", "./sub/", "./sub/beta.py"]
        assert s.server._files_latest_request == {}
        s.stop()

    def test_empty_work_dir_falls_back_to_the_daemon_work_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "only.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        s.get_files("", "")
        assert _texts(s.wait_populated()["files"]) == ["./only.py"]
        s.stop()

    def test_non_string_prefix_is_treated_as_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "only.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        s.server._handle_command({"type": "getFiles", "prefix": {"a": 1}, "workDir": str(root)})
        reply = s.wait_populated()
        assert reply["prefix"] == "" and _texts(reply["files"]) == ["./only.py"]
        s.stop()

    def test_tab_below_home_is_served_by_the_home_index(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        touch(home, "notes.md", "proj/a.py", "proj/src/b.py", "other/c.py")
        s = _Server(tmp_path, home, str(home))
        s.get_files("", str(home / "proj"))
        items = s.wait_populated()["files"]
        assert _texts(items[:3]) == ["./a.py", "./src/", "./src/b.py"]
        assert {(i["type"], i["text"]) for i in items[3:]} == {
            ("home", "~/notes.md"), ("home", "~/other/"), ("home", "~/other/c.py"),
        }, "the rest of home follows the work dir, without the work dir's own entries"
        # Warm now: another tab below home answers synchronously.
        s.get_files("", str(home / "other"), tab_id="t2")
        assert _texts(s.populated()[1]["files"])[0] == "./c.py"
        assert "~/proj/src/b.py" in _texts(s.populated()[1]["files"])
        assert s.populated()[1]["tabId"] == "t2"
        assert set(s.server._file_index._indexes) == {str(home)}
        s.stop()

    def test_gitignored_tab_below_home_gets_its_own_index_in_a_second_round(
        self, tmp_path: Path,
    ) -> None:
        home = tmp_path / "home"
        touch(home, ".gitignore", "scratch/work.py", "proj/a.py")
        (home / ".gitignore").write_text("scratch\n")
        s = _Server(tmp_path, home, str(home))
        s.get_files("", str(home / "scratch"))
        reply = s.wait_populated()
        assert _texts(reply["files"]) == ["./work.py", "~/proj/", "~/proj/a.py", "~/.gitignore"]
        assert set(s.server._file_index._indexes) == {str(home), str(home / "scratch")}
        assert s.server._files_latest_request == {}
        s.stop()

    def test_superseded_cold_request_is_dropped(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "abc.py", "abd.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        gate = threading.Event()
        s.server._file_index.ensure(str(tmp_path / "unrelated"), gate.wait)
        s.get_files("ab", str(root))
        s.get_files("abd", str(root))
        assert len(s.files_events()) == 2 and all(e["loading"] for e in s.files_events())
        gate.set()
        reply = s.wait_populated()
        time.sleep(0.2)
        assert len(s.populated()) == 1
        assert reply["prefix"] == "abd" and _texts(reply["files"]) == ["./abd.py"]
        assert s.server._files_latest_request == {}
        s.stop()

    def test_superseded_second_round_is_abandoned(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        touch(home, ".gitignore", "scratch/work.py")
        (home / ".gitignore").write_text("scratch\n")
        s = _Server(tmp_path, home, str(home))
        gate = threading.Event()
        s.server._file_index.ensure(str(tmp_path / "unrelated"), gate.wait)
        # Both requests queue behind the gate; the second supersedes the
        # first before the worker builds anything, so when the first
        # round finds ``scratch`` uncovered it must not start a second.
        s.get_files("w", str(home / "scratch"))
        s.get_files("", str(home))
        gate.set()
        reply = s.wait_populated()
        time.sleep(0.2)
        assert len(s.populated()) == 1
        assert reply["prefix"] == "" and _texts(reply["files"]) == ["./.gitignore"]
        assert str(home / "scratch") not in s.server._file_index._indexes
        assert s.server._files_latest_request == {}
        s.stop()

    def test_connections_have_independent_tokens(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        s.get_files("", str(root), conn_id="c1")
        s.get_files("", str(root), conn_id="c2")
        s.wait_populated(2)
        assert {e["connId"] for e in s.populated()} == {"c1", "c2"}
        s.stop()

    def test_refresh_after_task_picks_up_new_files(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        touch(root, "a.py")
        s = _Server(tmp_path, tmp_path / "home", str(root))
        s.server._refresh_files_after_task(str(root))
        time.sleep(0.1)
        assert local_view(s.server._file_index, str(root)) is None, "never-indexed roots stay alone"
        _build(s.server._file_index, str(root))
        time.sleep(0.02)
        touch(root, "new_test.py")
        s.server._refresh_files_after_task(str(root))
        reg = s.server._file_index
        _wait(lambda: "new_test.py" in (local_view(reg, str(root)) or FileView([])).paths)
        assert s.files_events() == [], "no unsolicited files event"
        s.get_files("new", str(root))
        assert _texts(s.files_events()[0]["files"]) == ["./new_test.py"]
        s.stop()

    def test_set_work_dir_prewarms_the_new_directory(self, tmp_path: Path) -> None:
        old, new = tmp_path / "old", tmp_path / "new"
        touch(old, "o.py")
        touch(new, "n.py")
        s = _Server(tmp_path, tmp_path / "home", str(old))
        s.server._handle_command({"type": "setWorkDir", "workDir": str(new)})
        assert s.server.work_dir == str(new)
        _wait(lambda: local_view(s.server._file_index, str(new)) is not None)
        s.get_files("", "")
        assert _texts(s.files_events()[0]["files"]) == ["./n.py"]
        s.stop()


def test_home_default_is_the_users_home() -> None:
    assert FileIndexRegistry().home == os.path.abspath(str(Path.home()))
