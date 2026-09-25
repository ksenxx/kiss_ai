# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The subprocess reaper plugin stops every process a test run leaves behind.

Each test writes a small pytest project into ``tmp_path`` (outside this
repository, so the root ``conftest.py`` is not picked up), runs pytest
on it in a fresh interpreter with ``-p kiss.tests.subprocess_reaper``,
and then checks from the outside that the processes the inner tests
started -- and deliberately never stopped -- are gone, and that the run
reported who leaked them.

Not exercised here: the Windows branches of ``_leads_own_group`` /
``kill_process_group`` (POSIX host), and the ``ProcessLookupError`` /
``OSError`` fallbacks that only fire when a child exits in the
microseconds between ``poll()`` and the signal.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from kiss.core.processes import pid_alive
from kiss.tests.conftest import posix_only

_SLEEPER = "import time; time.sleep(600)"

_LEAK_TEST = f"""
import subprocess, sys
from pathlib import Path

def test_leak():
    proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}], start_new_session=True)
    Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))
"""

# ``pid_alive`` rather than ``os.kill(pid, 0)``: on Windows the latter is
# not a probe but ``TerminateProcess`` (and raises ``WinError 87`` for a
# pid that is gone).
_ASSERT_DEAD = """
from pathlib import Path
from kiss.core.processes import pid_alive

def test_earlier_process_is_gone():
    assert not pid_alive(int(Path(__file__).with_name("leaked.pid").read_text()))
"""


_ASSERT_ALIVE = """
from pathlib import Path
from kiss.core.processes import pid_alive

def test_earlier_process_is_still_running():
    assert pid_alive(int(Path(__file__).with_name("leaked.pid").read_text()))
"""


def _run_pytest(project: Path) -> subprocess.CompletedProcess[str]:
    """Run pytest on *project* in a fresh interpreter with the reaper plugin loaded."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", "kiss.tests.subprocess_reaper",
            "-p", "no:cacheprovider", "-q", str(project),
        ],
        capture_output=True, text=True, env=env, cwd=project, timeout=180, check=False,
    )


def _write(project: Path, name: str, source: str) -> None:
    (project / name).write_text(textwrap.dedent(source), encoding="utf-8")


def _leaked_pid(project: Path, name: str = "leaked.pid") -> int:
    return int((project / name).read_text())


def test_process_leaked_by_a_test_is_gone_before_the_next_test(tmp_path: Path) -> None:
    _write(tmp_path, "test_a_leak.py", _LEAK_TEST)
    _write(tmp_path, "test_b_check.py", _ASSERT_DEAD)

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "LeakedSubprocessWarning: test test_a_leak.py::test_leak left 1 subprocess(es)" in (
        result.stdout
    )
    assert _SLEEPER in result.stdout
    assert not pid_alive(_leaked_pid(tmp_path))


def test_process_that_exits_on_its_own_is_not_reported(tmp_path: Path) -> None:
    _write(
        tmp_path, "test_quick.py", """
        import subprocess, sys

        def test_run_to_completion():
            subprocess.run([sys.executable, "-c", "pass"], check=True)
            with subprocess.Popen([sys.executable, "-c", "pass"]) as proc:
                proc.wait()
        """,
    )

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "LeakedSubprocessWarning" not in result.stdout
    assert "subprocess_reaper" not in result.stdout


def test_module_fixture_process_lives_exactly_as_long_as_its_module(tmp_path: Path) -> None:
    _write(
        tmp_path, "test_a_shared.py", f"""
        import os, subprocess, sys
        import pytest
        from pathlib import Path

        @pytest.fixture(scope="module")
        def server():
            proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
            Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))
            yield proc

        def test_one(server):
            assert server.poll() is None

        def test_two(server):
            assert server.poll() is None
        """,
    )
    _write(tmp_path, "test_b_check.py", _ASSERT_DEAD)

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert "module-scoped fixture 'server' (test_a_shared.py) left 1 subprocess(es)" in (
        result.stdout
    )
    assert not pid_alive(_leaked_pid(tmp_path))


def test_setupclass_process_lives_exactly_as_long_as_its_class(tmp_path: Path) -> None:
    _write(
        tmp_path, "test_a_class.py", f"""
        import subprocess, sys, unittest
        from pathlib import Path

        class Server(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
                Path(__file__).with_name("leaked.pid").write_text(str(cls.proc.pid))

            def test_one(self):
                self.assertIsNone(self.proc.poll())

            def test_two(self):
                self.assertIsNone(self.proc.poll())
        """,
    )
    _write(tmp_path, "test_b_check.py", _ASSERT_DEAD)

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert "class-scoped fixture '_unittest_setUpClass_fixture_Server'" in result.stdout
    assert not pid_alive(_leaked_pid(tmp_path))


def test_process_started_outside_any_test_is_terminated_at_session_end(tmp_path: Path) -> None:
    _write(
        tmp_path, "conftest.py", f"""
        import subprocess, sys
        from pathlib import Path

        proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
        Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))
        """,
    )
    _write(tmp_path, "test_noop.py", "def test_noop():\n    pass\n")

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "subprocess_reaper: the session left 1 subprocess(es) running; terminated:" in (
        result.stdout
    )
    assert not pid_alive(_leaked_pid(tmp_path))


def test_owned_asyncio_subprocess_is_left_to_its_owner_until_session_end(tmp_path: Path) -> None:
    """A child something still references (here: a module global holding the
    asyncio ``Process``) is not the test's leak: it survives into the next
    test and is swept when the run ends."""
    _write(
        tmp_path, "test_a_async.py", f"""
        import asyncio, sys
        from pathlib import Path

        KEEP = []

        async def start():
            proc = await asyncio.create_subprocess_exec(sys.executable, "-c", {_SLEEPER!r})
            KEEP.append(proc)
            Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))

        def test_start():
            asyncio.run(start())
        """,
    )
    _write(tmp_path, "test_b_check.py", _ASSERT_ALIVE)

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "LeakedSubprocessWarning" not in result.stdout
    assert "subprocess_reaper: the session left 1 subprocess(es) running; terminated:" in (
        result.stdout
    )
    assert not pid_alive(_leaked_pid(tmp_path))


def test_process_started_on_first_use_of_a_shared_fixture_is_not_killed(tmp_path: Path) -> None:
    """The Playwright shape: a module-scoped fixture yields a tool that
    starts its driver lazily inside the first test.  The driver belongs to
    the tool, so it must survive that test's teardown, and the fixture's
    own teardown stopping it leaves nothing to report."""
    _write(
        tmp_path, "test_lazy.py", f"""
        import subprocess, sys
        import pytest
        from pathlib import Path

        class Tool:
            proc = None

            def use(self):
                if self.proc is None:
                    self.proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
                    Path(__file__).with_name("leaked.pid").write_text(str(self.proc.pid))
                assert self.proc.poll() is None

        @pytest.fixture(scope="module")
        def tool():
            tool = Tool()
            yield tool
            tool.proc.terminate()
            tool.proc.wait()

        def test_first_use_starts_the_driver(tool):
            tool.use()

        def test_second_use_finds_it_running(tool):
            tool.use()
        """,
    )

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "LeakedSubprocessWarning" not in result.stdout
    assert "subprocess_reaper" not in result.stdout
    assert not pid_alive(_leaked_pid(tmp_path))


@posix_only("SIG_IGN")
def test_child_that_ignores_sigterm_is_killed(tmp_path: Path) -> None:
    _write(
        tmp_path, "test_stubborn.py", """
        import subprocess, sys
        from pathlib import Path

        STUBBORN = (
            "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "open(sys.argv[1], 'w').write('ready'); time.sleep(600)"
        )

        def test_leak():
            ready = Path(__file__).with_name("ready")
            proc = subprocess.Popen([sys.executable, "-c", STUBBORN, str(ready)])
            Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))
            while not ready.exists() or not ready.read_text():
                pass
        """,
    )

    started = time.monotonic()
    result = _run_pytest(tmp_path)
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stdout + result.stderr
    # SIGTERM was ignored, so the sweep had to sit out the grace period
    # before SIGKILL -- and must not have waited for the 600 s sleep.
    assert 5 < elapsed < 60, elapsed
    assert "left 1 subprocess(es)" in result.stdout
    assert "(still running)" not in result.stdout
    assert not pid_alive(_leaked_pid(tmp_path))


@posix_only("process groups")
def test_group_leader_is_terminated_with_its_descendants(tmp_path: Path) -> None:
    _write(
        tmp_path, "test_tree.py", f"""
        import subprocess, sys
        from pathlib import Path

        # The leader ignores SIGTERM (so the group gets SIGKILL too) and
        # starts a grandchild that survives the leader's death unless the
        # whole group is signalled.
        LEADER = '''
        import signal, subprocess, sys, time
        from pathlib import Path
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
        Path(sys.argv[1]).write_text(str(child.pid))
        time.sleep(600)
        '''

        def test_leak():
            grandchild_pid = Path(__file__).with_name("grandchild.pid")
            proc = subprocess.Popen(
                [sys.executable, "-c", LEADER, str(grandchild_pid)], start_new_session=True
            )
            Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))
            while not grandchild_pid.exists() or not grandchild_pid.read_text():
                pass
        """,
    )

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "left 1 subprocess(es)" in result.stdout
    assert not pid_alive(_leaked_pid(tmp_path))
    assert not pid_alive(_leaked_pid(tmp_path, "grandchild.pid"))


@posix_only("process groups")
def test_group_survivors_are_killed_after_the_leader_exits(tmp_path: Path) -> None:
    """The double-fork daemon shape: the leader forks a SIGTERM-ignoring
    grandchild into its own process group and exits at once, so at sweep
    time only the group -- not the tracked child -- is alive."""
    _write(
        tmp_path, "test_daemon.py", """
        import subprocess, sys
        from pathlib import Path

        GRANDCHILD = (
            "import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "open(sys.argv[1], 'w').write('ready'); time.sleep(600)"
        )
        LEADER = (
            "import subprocess, sys; "
            "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]); "
            "open(sys.argv[3], 'w').write(str(child.pid))"
        )

        def test_leak():
            ready = Path(__file__).with_name("ready")
            grandchild_pid = Path(__file__).with_name("grandchild.pid")
            proc = subprocess.Popen(
                [sys.executable, "-c", LEADER, GRANDCHILD, str(ready), str(grandchild_pid)],
                start_new_session=True,
            )
            assert proc.wait(timeout=60) == 0  # the leader is gone before the sweep
            while not ready.exists() or not ready.read_text():
                pass
        """,
    )

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "left 1 subprocess(es)" in result.stdout
    assert "(still running)" not in result.stdout
    assert not pid_alive(_leaked_pid(tmp_path, "grandchild.pid"))


def test_fixture_and_teardown_cleanup_run_before_the_sweep(tmp_path: Path) -> None:
    """A fixture or ``tearDown`` that stops its own process is not reported:
    the sweep runs after them and finds nothing left."""
    _write(
        tmp_path, "test_tidy.py", f"""
        import subprocess, sys, unittest
        import pytest

        @pytest.fixture(scope="module")
        def server():
            proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
            yield proc
            proc.terminate()
            proc.wait()

        def test_uses_server(server):
            assert server.poll() is None

        class Tidy(unittest.TestCase):
            def setUp(self):
                self.proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])

            def tearDown(self):
                self.proc.kill()
                self.proc.wait()

            def test_uses_process(self):
                self.assertIsNone(self.proc.poll())
        """,
    )

    result = _run_pytest(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "LeakedSubprocessWarning" not in result.stdout


def test_run_cut_short_by_pytest_exit_is_swept_at_interpreter_exit(tmp_path: Path) -> None:
    """``pytest.exit`` skips the teardown phase, so the test's bucket is
    still open when the session ends; the ``atexit`` sweep takes it."""
    _write(
        tmp_path, "test_exit.py", f"""
        import subprocess, sys
        import pytest
        from pathlib import Path

        def test_leak_then_exit():
            proc = subprocess.Popen([sys.executable, "-c", {_SLEEPER!r}])
            Path(__file__).with_name("leaked.pid").write_text(str(proc.pid))
            pytest.exit("stop right here")
        """,
    )

    result = _run_pytest(tmp_path)

    assert "stop right here" in result.stdout
    assert "subprocess_reaper: the process left 1 subprocess(es) running; terminated:" in (
        result.stderr
    )
    assert not pid_alive(_leaked_pid(tmp_path))
