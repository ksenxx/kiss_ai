"""Round-3 regressions for the gpt-5.6-sol review findings in
``tmp/review3-sorcar.md``.

Two defects, each reproduced against the real production objects
(no mocks, patches, or fakes):

1. ``persistence._RWLock`` — a stranded acquisition whose exception
   traceback is retained (an ``except KeyboardInterrupt`` handler, a
   stored exception) kept its ownership token alive forever, blocking
   every later writer/reader despite the weakref design and the 1 s
   recheck.  Fixed by recording each acquisition's own generator on its
   token: a closed generator's ``gi_frame`` is ``None`` regardless of
   who retains the frame, so ``_token_active`` pronounces the
   acquisition over without garbage collection.

2. ``git_worktree._reclaim_process_lock`` — an injected stop that
   skipped the marker-withdrawal cleanup left the thread-local re-entry
   marker SET after the kernel flock was released, so every later call
   on that thread silently bypassed cross-process locking.  Fixed by
   fusing marker and flock lifetime into one token that owns the open
   handle: the marker is live exactly while the kernel lock is held.

The injected-stop boundaries exercised here (the first line of a
``finally`` block, skipping its body) are driven with a real exception
raised by ``sys.settrace`` at that exact line — the same boundary class
``PyThreadState_SetAsyncExc`` can hit — because a timed async injection
cannot deterministically land on one bytecode.  No production code is
altered or replaced: the trace only chooses *when* the exception
arrives.
"""

from __future__ import annotations

import gc
import inspect
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as persistence
from kiss.agents.sorcar import git_worktree
from kiss.agents.sorcar.git_worktree import (
    _held_reclaim_locks,
    _reclaim_process_lock,
)


def _make_repo(path: Path) -> Path:
    """Create a git repo on branch ``main`` with one committed file."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        capture_output=True, check=True,
    )
    for key, val in (("user.email", "t@t.com"), ("user.name", "T")):
        subprocess.run(
            ["git", "-C", str(path), "config", key, val],
            capture_output=True, check=True,
        )
    (path / "f.txt").write_text("f0\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."], capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return path


def _reclaim_lock_path(repo: Path) -> Path:
    """Return the repo's ``kiss-reclaim.lock`` path."""
    common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    common_dir = Path(common)
    if not common_dir.is_absolute():
        common_dir = (repo / common_dir).resolve()
    return common_dir / "kiss-reclaim.lock"


def _finally_line(func: Any, needle: str) -> int:
    """Return the absolute line number of *needle* inside *func*'s source.

    Used to aim the trace-injected ``KeyboardInterrupt`` at the first
    statement of a ``finally`` block, which skips the block's body —
    the exact boundary the round-3 review identified.
    """
    lines, start = inspect.getsourcelines(inspect.unwrap(func))
    for i, line in enumerate(lines):
        if needle in line:
            return start + i
    raise AssertionError(f"{needle!r} not found in {func!r}")


def _run_with_finally_skipped(
    code_name: str, target_line: int, body: Any
) -> BaseException:
    """Run *body* with a trace that raises at *target_line* of *code_name*.

    Returns:
        The escaped exception, WITH its traceback still attached — the
        retention scenario from the review.
    """
    escaped: list[BaseException] = []

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if (
            event == "line"
            and frame.f_lineno == target_line
            and frame.f_code.co_name == code_name
        ):
            raise KeyboardInterrupt("injected at finally entry")
        return tracer

    sys.settrace(tracer)
    try:
        body()
    except BaseException as exc:  # noqa: BLE001 — the injected stop
        escaped.append(exc)
    finally:
        sys.settrace(None)
    assert escaped, "the injected exception never escaped"
    assert isinstance(escaped[0], KeyboardInterrupt)
    return escaped[0]


class TestRWLockTracebackRetention:
    """Finding 1: retained tracebacks must not prolong lock ownership."""

    def test_retained_reader_traceback_does_not_block_writer(self) -> None:
        """A reader acquisition whose teardown was skipped by an
        injected stop, and whose exception is still being HELD (as a
        stop handler holds it), must stop blocking writers within one
        recheck tick."""
        lock = persistence._RWLock()
        target = _finally_line(
            persistence._RWLock._read_lock_gen,
            "self._teardown(ref, token, writer=False)",
        )

        def acquire_and_die() -> None:
            with lock.read_lock():
                pass

        exc = _run_with_finally_skipped("_read_lock_gen", target, acquire_and_die)
        # The stranded registration is present and its token is
        # traceback-retained — the exact pre-fix wedge.
        assert lock._reader_refs, "injection did not strand the registration"

        acquired = threading.Event()

        def writer() -> None:
            with lock.write_lock():
                acquired.set()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            assert acquired.wait(timeout=5.0), (
                "writer stayed blocked by a traceback-retained reader token"
            )
        finally:
            del exc  # release the retained traceback
            t.join(timeout=10)

    def test_retained_writer_traceback_does_not_block_reader(self) -> None:
        """Symmetric case: a stranded writer whose token survives in a
        retained traceback must stop excluding readers."""
        lock = persistence._RWLock()
        target = _finally_line(
            persistence._RWLock._write_lock_gen,
            "self._teardown(ref, token, writer=True)",
        )

        def acquire_and_die() -> None:
            with lock.write_lock():
                pass

        exc = _run_with_finally_skipped("_write_lock_gen", target, acquire_and_die)
        assert lock._writer_ref is not None, (
            "injection did not strand the writer publication"
        )

        acquired = threading.Event()

        def reader() -> None:
            with lock.read_lock():
                acquired.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            assert acquired.wait(timeout=5.0), (
                "reader stayed blocked by a traceback-retained writer token"
            )
        finally:
            del exc
            t.join(timeout=10)

    def test_normal_acquisitions_still_exclude_each_other(self) -> None:
        """The generator-liveness check must not weaken normal mutual
        exclusion: a live writer still blocks a reader until release."""
        lock = persistence._RWLock()
        in_write = threading.Event()
        release = threading.Event()
        read_done = threading.Event()

        def writer() -> None:
            with lock.write_lock():
                in_write.set()
                release.wait(timeout=30)

        def reader() -> None:
            with lock.read_lock():
                read_done.set()

        wt = threading.Thread(target=writer, daemon=True)
        wt.start()
        assert in_write.wait(timeout=10)
        rt = threading.Thread(target=reader, daemon=True)
        rt.start()
        assert not read_done.wait(timeout=1.5), (
            "reader entered while a live writer held the lock"
        )
        release.set()
        assert read_done.wait(timeout=10)
        wt.join(timeout=10)
        rt.join(timeout=10)


class TestReclaimMarkerFlockConsistency:
    """Finding 2: the re-entry marker must never outlive the flock."""

    def _flock_probe_blocked(self, lock_path: Path) -> bool:
        """Return True when a fresh process CANNOT take the flock now."""
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "from kiss.core.file_lock import lock_exclusive\n"
                    "h = open(sys.argv[1], 'a+')\n"
                    "sys.exit(0 if lock_exclusive(h, blocking=False) else 3)\n"
                ),
                str(lock_path),
            ],
            capture_output=True,
            timeout=60,
        )
        return probe.returncode == 3

    def test_skipped_cleanup_stays_fail_closed_then_heals(self) -> None:
        """Skip the marker-withdrawal ``finally`` with an injected stop
        while its exception (and so the acquisition frame, token and
        handle) is retained.  At every stage marker and kernel flock
        must agree — pre-fix the marker said 'held' while the flock was
        already released, silently bypassing cross-process locking on
        every later call of this thread."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            lock_path = _reclaim_lock_path(repo)
            target = _finally_line(
                git_worktree._reclaim_process_lock, "held.pop(lock_path, None)"
            )

            def enter_and_die() -> None:
                with _reclaim_process_lock(repo):
                    pass

            exc = _run_with_finally_skipped(
                "_reclaim_process_lock", target, enter_and_die
            )
            try:
                held = _held_reclaim_locks()
                ref = held.get(str(lock_path))
                marker_live = ref is not None and ref() is not None
                flock_held = self._flock_probe_blocked(lock_path)
                # Fail-closed consistency: if the marker claims the
                # hold, the kernel flock must really be held (the
                # leaked token still owns the open handle).
                assert marker_live, "marker was withdrawn despite the skip"
                assert flock_held, (
                    "re-entry marker claims a hold the kernel no longer "
                    "grants — cross-process locking would be bypassed"
                )
            finally:
                del exc
            gc.collect()
            # Token gone: handle closed, flock released, marker dead.
            ref = _held_reclaim_locks().get(str(lock_path))
            assert ref is None or ref() is None, "marker outlived its token"
            assert not self._flock_probe_blocked(lock_path), (
                "flock still held after the leaked token died"
            )
            # And the same thread re-acquires for real (no bypass).
            with _reclaim_process_lock(repo):
                assert self._flock_probe_blocked(lock_path), (
                    "re-acquisition after healing did not take the flock"
                )
            assert not self._flock_probe_blocked(lock_path)

    def test_reentry_still_works_within_one_hold(self) -> None:
        """Normal nesting still yields immediately without a second
        descriptor (which would self-deadlock)."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp) / "repo")
            done = threading.Event()

            def nested() -> None:
                with _reclaim_process_lock(repo):
                    with _reclaim_process_lock(repo):
                        done.set()

            t = threading.Thread(target=nested, daemon=True)
            t.start()
            assert done.wait(timeout=30), "re-entrant take self-deadlocked"
            t.join(timeout=10)
