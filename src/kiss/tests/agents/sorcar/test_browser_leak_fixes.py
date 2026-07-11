# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E tests reproducing (and verifying the fix for) Chromium browser leaks.

Long-horizon tasks accumulated open Chromium windows because:

1. ``WebUseTool`` never recorded the browser OS PID, and a failed
   graceful ``close()`` (wedged driver, cross-thread greenlet error)
   was silently swallowed — the Chromium process leaked forever.
2. The LLM agent had no tool to close the browser mid-task
   (``get_tools()`` excluded ``close``).
3. Every parallel sub-agent launched its own visible Chromium in an
   escalated ``browser_profile_N`` directory that was never cleaned.

These tests use a REAL headless Chromium (no mocks) and assert on the
actual OS process state via ``os.kill(pid, 0)``.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool

# ``os.kill(pid, 0)`` liveness probes, signal escalation, and
# SingletonLock symlinks all need POSIX semantics.
_posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process/signal semantics required"
)


def _pid_alive(pid: int) -> bool:
    """Return True iff the OS process *pid* exists."""
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:  # pragma: no cover — foreign-owned process
        return True
    except OSError:
        return False


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    """Poll until *pid* exits, returning True if it died within *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return not _pid_alive(pid)


def _dead_pid() -> int:
    """Return a PID that is guaranteed dead (spawn-and-reap a child)."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _launch(tool: WebUseTool) -> int:
    """Open a page so the browser launches; return its recorded OS PID."""
    result = tool.go_to_url("about:blank")
    assert not result.startswith("Error"), result
    pid = tool._browser_pid
    assert pid is not None, "browser PID was not recorded at launch"
    assert _pid_alive(pid), "recorded browser PID is not a live process"
    return pid


@_posix_only
class TestBrowserProcessKilledOnClose:
    """close() must guarantee the Chromium OS process actually exits."""

    def test_close_kills_chromium_process(self, tmp_path):
        tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
        try:
            pid = _launch(tool)
        finally:
            tool.close()
        assert _wait_dead(pid), f"Chromium (pid {pid}) leaked after close()"

    def test_cross_thread_close_kills_process(self, tmp_path):
        """Reproduces leak cause: cross-thread close.

        Sync-Playwright objects are greenlet-bound to their creating
        thread, so ``context.close()`` from another thread raises
        ``greenlet.error`` — previously swallowed at debug level,
        leaking the Chromium process (this is exactly what happened on
        the atexit path for browsers created in ThreadPoolExecutor
        workers).  The PID-kill fallback must still terminate it.
        """
        box: dict = {}

        def _worker() -> None:
            tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
            box["pid"] = _launch(tool)
            box["tool"] = tool

        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=120)
        assert not t.is_alive(), "worker thread hung"
        assert "pid" in box, "browser failed to launch in worker thread"
        # Close from the MAIN thread: graceful close fails cross-thread.
        box["tool"].close()
        assert _wait_dead(box["pid"]), (
            f"Chromium (pid {box['pid']}) leaked after cross-thread close()"
        )

    def test_ensure_browser_relaunch_kills_previous_process(self, tmp_path):
        """A crash-recovery relaunch must not leave the old process behind."""
        tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
        try:
            old_pid = _launch(tool)
            # Simulate the renderer-crash state that forces a full teardown
            # + relaunch in _ensure_browser (page cleared, context kept).
            tool._on_page_crash(tool._page)
            assert tool.go_to_url("about:blank").startswith("Page:")
            new_pid = tool._browser_pid
            assert new_pid is not None
            assert _wait_dead(old_pid), (
                f"old Chromium (pid {old_pid}) leaked across relaunch"
            )
        finally:
            tool.close()
        assert _wait_dead(new_pid)

    def test_kill_refuses_mismatched_process_identity(self):
        """PID-reuse guard: never signal a process that is not our browser.

        Points the recorded browser PID at a real, live, unrelated child
        process with a non-matching identity fingerprint; the kill
        fallback must refuse to signal it.
        """
        victim = subprocess.Popen(["sleep", "30"])
        tool = WebUseTool(user_data_dir=None, headless=True)
        try:
            tool._browser_pid = victim.pid
            tool._browser_identity = "bogus-identity-recorded-long-ago"
            tool._kill_browser_process()
            assert victim.poll() is None, (
                "unrelated process was killed despite identity mismatch"
            )
            assert tool._browser_pid is None
        finally:
            victim.terminate()
            victim.wait()
            tool.close()


@_posix_only
class TestCloseBrowserTool:
    """The agent must be able to close the browser mid-task via a tool."""

    def test_close_browser_tool_is_registered(self, tmp_path):
        tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
        try:
            names = [t.__name__ for t in tool.get_tools()]
            assert "close_browser" in names
            assert "close" not in names
        finally:
            tool.close()

    def test_close_browser_kills_process_and_relaunches(self, tmp_path):
        tool = WebUseTool(user_data_dir=str(tmp_path / "prof"), headless=True)
        try:
            pid = _launch(tool)
            msg = tool.close_browser()
            assert "closed" in msg.lower()
            assert _wait_dead(pid), (
                f"Chromium (pid {pid}) survived close_browser()"
            )
            # The next web tool call lazily relaunches a fresh browser.
            new_pid = _launch(tool)
            assert new_pid != pid
        finally:
            tool.close()
        assert _wait_dead(new_pid)


@_posix_only
class TestEphemeralProfile:
    """Sub-agent browsers use throwaway profiles removed on close."""

    def test_ephemeral_profile_removed_on_close(self):
        tool = WebUseTool(headless=True, ephemeral=True)
        profile = tool.user_data_dir
        assert profile is not None and os.path.isdir(profile)
        try:
            pid = _launch(tool)
        finally:
            tool.close()
        assert _wait_dead(pid)
        assert not os.path.exists(profile), "ephemeral profile dir leaked"

    def test_ephemeral_close_without_launch(self):
        tool = WebUseTool(headless=True, ephemeral=True)
        profile = tool.user_data_dir
        assert profile is not None and os.path.isdir(profile)
        tool.close()
        assert not os.path.exists(profile)

    def test_ephemeral_profile_removed_again_after_revival(self):
        """close() → revive (relaunch) → close() must delete the profile
        both times: the ephemeral marker survives the first close."""
        tool = WebUseTool(headless=True, ephemeral=True)
        profile = tool.user_data_dir
        assert profile is not None
        try:
            pid1 = _launch(tool)
            tool.close()
            assert _wait_dead(pid1)
            assert not os.path.exists(profile)
            # Revive: the next web tool call relaunches and recreates the dir.
            pid2 = _launch(tool)
            assert os.path.isdir(profile)
        finally:
            tool.close()
        assert _wait_dead(pid2)
        assert not os.path.exists(profile), (
            "ephemeral profile dir leaked after revival"
        )


@_posix_only
class TestStaleEscalationDirCleanup:
    """Stale ``<profile>_N`` escalation dirs (dead lock PID) are removed."""

    def test_stale_escalation_dirs_cleaned_at_launch(self, tmp_path):
        base = tmp_path / "profile"
        base.mkdir()
        dead = _dead_pid()
        for i in (1, 2):
            stale = tmp_path / f"profile_{i}"
            stale.mkdir()
            os.symlink(f"testhost-{dead}", str(stale / "SingletonLock"))
        tool = WebUseTool(user_data_dir=str(base), headless=True)
        try:
            pid = _launch(tool)
            assert not (tmp_path / "profile_1").exists(), (
                "stale profile_1 not cleaned"
            )
            assert not (tmp_path / "profile_2").exists(), (
                "stale profile_2 not cleaned"
            )
        finally:
            tool.close()
        assert _wait_dead(pid)

    def test_live_escalation_dir_is_preserved(self, tmp_path):
        base = tmp_path / "profile"
        base.mkdir()
        live = tmp_path / "profile_1"
        live.mkdir()
        os.symlink(f"testhost-{os.getpid()}", str(live / "SingletonLock"))
        tool = WebUseTool(user_data_dir=str(base), headless=True)
        try:
            pid = _launch(tool)
            assert (tmp_path / "profile_1").exists(), (
                "live escalation dir was deleted"
            )
        finally:
            tool.close()
        assert _wait_dead(pid)


class TestSubagentBrowserPolicy:
    """Parallel sub-agents get headless, ephemeral-profile browsers."""

    def test_subagent_gets_headless_ephemeral_browser(self):
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        agent = ChatSorcarAgent("subagent-browser-policy-test")
        agent._subagent_info = {"parent_task_id": "", "parent_tab_id": ""}
        try:
            agent._get_tools()
            tool = agent.web_use_tool
            assert tool is not None
            assert tool._headless is True
            assert tool._ephemeral_dir is not None
            assert tool.user_data_dir == tool._ephemeral_dir
        finally:
            if agent.web_use_tool is not None:
                agent.web_use_tool.close()

    def test_top_level_agent_keeps_visible_persistent_browser(self):
        from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent

        agent = ChatSorcarAgent("toplevel-browser-policy-test")
        try:
            agent._get_tools()
            tool = agent.web_use_tool
            assert tool is not None
            assert tool._headless is False
            assert tool._ephemeral_dir is None
        finally:
            if agent.web_use_tool is not None:
                agent.web_use_tool.close()
