# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for useful_tools.py module."""

import errno
import inspect
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import useful_tools
from kiss.agents.sorcar.useful_tools import (
    _MAX_BINARY_READ_BYTES,
    NOT_STARTED,
    UsefulTools,
    _truncate_output,
)
from kiss.core import tool_interrupt
from kiss.core.processes import pid_alive
from kiss.core.tool_verdict import ToolRedirect
from kiss.tests.conftest import host_sed_is_gnu, is_root, posix_only
from kiss.tests.server.pause import Pause


@pytest.fixture
def temp_test_dir():
    test_dir = Path(tempfile.mkdtemp()).resolve()
    original_dir = Path.cwd()
    os.chdir(test_dir)
    yield test_dir
    os.chdir(original_dir)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def tools(temp_test_dir):
    return UsefulTools(), temp_test_dir


class TestUsefulTools:

    def test_write_to_directory_path(self, tools):
        ut, test_dir = tools
        subdir = test_dir / "subdir"
        subdir.mkdir()
        result = ut.Write(str(subdir), "content")
        assert "Error:" in result


@pytest.fixture
def streaming_tools(temp_test_dir):
    streamed: list[str] = []
    ut = UsefulTools(stream_callback=streamed.append)
    return ut, temp_test_dir, streamed


@pytest.fixture(params=[False, True], ids=["nonstreaming", "streaming"])
def any_tools(request, temp_test_dir):
    if request.param:
        return UsefulTools(stream_callback=lambda _: None), temp_test_dir
    return UsefulTools(), temp_test_dir
@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test (signals, bash scripts)")
class TestAdversarial:
    """Adversarial tests to try to break the Popen/killpg changes."""

    def test_interrupt_kills_child(self, any_tools):
        """KeyboardInterrupt must kill the child process group."""
        import _thread
        import threading
        import time

        ut, test_dir = any_tools
        pid_file = test_dir / "interrupt_child.pid"
        script = test_dir / "interrupt_target.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nsleep 100\n"
        )
        script.chmod(0o755)

        child_pid = None

        def send_interrupt():
            nonlocal child_pid
            for _ in range(20):
                time.sleep(0.1)
                if pid_file.exists():
                    # Poll for parseable content, not mere existence:
                    # the shell redirect creates the file empty before
                    # echo writes the PID.
                    pid_text = pid_file.read_text().strip()
                    if pid_text.isdigit():
                        child_pid = int(pid_text)
                        break
            if child_pid:
                _thread.interrupt_main()

        t = threading.Thread(target=send_interrupt, daemon=True)
        t.start()
        try:
            ut.Bash(str(script), "interruptible", timeout_seconds=30)
        except KeyboardInterrupt:
            pass
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.3)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived KeyboardInterrupt!"

def _texts(root: Path, names: tuple[str, ...]) -> list[str]:
    """Return the text of each file of *names* under *root*."""
    return [(root / name).read_text() for name in names]


def _bash_exit(command: str, cwd: Path) -> int:
    """Return the exit code of *command* run by the host's bash in *cwd*."""
    return subprocess.run(["bash", "-c", command], cwd=cwd, timeout=60, capture_output=True,
                          stdin=subprocess.DEVNULL).returncode


class TestBugs:
    """Tests that expose bugs in useful_tools.py."""

    def test_truncate_output_tiny_limit(self):
        big = "X" * 200
        result = _truncate_output(big, 5)
        assert len(result) <= 5

    def test_read_refuses_an_oversized_binary_without_loading_it(self, temp_test_dir):
        """Read checks the binary size cap before reading, so an oversized PNG costs no memory.

        A sparse file one byte over the cap with a PNG signature (invalid
        UTF-8) used to be read whole by the text attempt before the
        binary branch measured it (A10-7).
        """
        big = temp_test_dir / "big.png"
        with open(big, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
            handle.truncate(_MAX_BINARY_READ_BYTES + 1)
        ut = UsefulTools(work_dir=str(temp_test_dir))
        tracemalloc.start()
        try:
            result = ut.Read(str(big), max_lines=1)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert result.startswith("Error: Binary file") and "too large" in result, result
        assert peak < _MAX_BINARY_READ_BYTES // 4, f"Read allocated {peak} bytes to refuse the file"

    def test_read_of_a_missing_binary_path_suggests_the_close_name(self, temp_test_dir):
        """A mistyped image name gets the same ``Did you mean`` answer as a mistyped text name.

        The size pre-check of the binary branch ran before the ``try``
        whose ``FileNotFoundError`` handler builds the suggestion, so a
        missing ``.png`` answered with the raw errno text (A14-1).
        """
        (temp_test_dir / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (temp_test_dir / "printer.py").write_text("x = 1\n")
        ut = UsefulTools(work_dir=str(temp_test_dir))
        for typo, close in (("screenshoot.png", "screenshot.png"), ("printr.py", "printer.py")):
            result = ut.Read(typo)
            assert result.startswith(f"Error: File not found: {typo}."), result
            assert close in result and "Did you mean" in result, result

    @pytest.mark.skipif(
        sys.platform == "win32", reason="the Windows volume rejects this 255-byte component",
    )
    def test_write_and_edit_take_a_file_name_of_the_full_length(self, temp_test_dir):
        """Write and Edit reach a file whose name is as long as the file system allows.

        The sibling temporary file carried the file's name plus 38
        characters, so a name over 217 bytes answered ``File name too
        long`` although the file itself is legal.
        """
        ut = UsefulTools(work_dir=str(temp_test_dir))
        path = temp_test_dir / ("n" * 251 + ".txt")
        assert ut.Write(str(path), "a = 1\n").startswith("Successfully wrote")
        assert ut.Edit(str(path), "a = 1", "a = 2").startswith("Successfully replaced 1")
        assert path.read_text() == "a = 2\n"
        assert os.listdir(temp_test_dir) == [path.name]

    @pytest.mark.skipif(is_root(), reason="root writes a read-only file")
    @posix_only("the rename over a read-only file is refused on Windows")
    def test_write_and_edit_refuse_a_read_only_file(self, temp_test_dir):
        """A file made read-only is refused, as the shell's ``>`` refuses it.

        The rename of a sibling temporary file over it takes the
        directory's permission only, so both tools replaced the file.
        Review 2026-11-11 S4-3: a ``sed -i`` does replace it that way,
        so it runs as written, not as the ``Edit`` that is refused.
        """
        ut = UsefulTools(work_dir=str(temp_test_dir))
        path = temp_test_dir / "locked.txt"
        path.write_text("a = 1\n")
        path.chmod(0o444)
        ut.Read(str(path))
        assert "Permission denied" in ut.Edit(str(path), "a = 1", "a = 2")
        assert "Permission denied" in ut.Write(str(path), "a = 3\n")
        assert path.read_text() == "a = 1\n"
        assert os.listdir(temp_test_dir) == ["locked.txt"]
        in_place = "sed -i" if host_sed_is_gnu() else "sed -i ''"
        assert ut.Bash(f"{in_place} 's/a = 1/a = 2/' locked.txt", "sed renames") == ""
        assert path.read_text() == "a = 2\n"

    def test_write_and_edit_reach_a_file_that_is_a_mount_point(self, temp_test_dir):
        """A file bound over another is written in place, as the shell's ``>`` writes it.

        Review 2026-11-11 S4r3 (X2): the rename of the sibling temporary
        file over a mount point is refused as busy, and only a
        ``PermissionError`` sent the write down the in-place path, so
        ``Write`` and ``Edit`` answered "Device or resource busy" for a
        single file bound into a sandbox or a container.  The file is
        bound for real, in a throwaway mount namespace.  Any other
        failure of the temporary file still leaves the old text: on a
        full file system an in-place write would have emptied the file.
        """
        in_namespace = ["unshare", "-Urm"]
        if (
            shutil.which("unshare") is None
            or subprocess.run([*in_namespace, "true"], capture_output=True).returncode
        ):
            pytest.skip("no unprivileged namespaces here")
        (temp_test_dir / "source.txt").write_text("a = 1\n")
        (temp_test_dir / "bound.txt").write_text("under the mount\n")
        (temp_test_dir / "full").mkdir()
        script = (
            "import os, subprocess, sys\n"
            "from kiss.agents.sorcar.useful_tools import UsefulTools\n"
            "mounts = (['--bind', 'source.txt', 'bound.txt'],\n"
            "          ['-t', 'tmpfs', '-o', 'size=64k', 'tmpfs', 'full'])\n"
            "if any(subprocess.run(['mount', *mount]).returncode for mount in mounts):\n"
            "    sys.exit(3)\n"
            "tools = UsefulTools(work_dir='.')\n"
            "tools.Read('bound.txt')\n"
            "print(tools.Edit('bound.txt', 'a = 1', 'a = 2'))\n"
            "print(tools.Write('bound.txt', 'a = 3\\n'))\n"
            "print(tools.Write('full/kept.txt', 'kept\\n'))\n"
            "print(tools.Write('full/kept.txt', 'x' * 1_000_000))\n"
            "print(os.listdir('full'), open('full/kept.txt').read().split())\n"
        )
        run = subprocess.run(
            [*in_namespace, sys.executable, "-c", script], cwd=temp_test_dir,
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        if run.returncode == 3:
            pytest.skip("nothing can be mounted here")
        said = run.stdout.splitlines()
        assert said[0].startswith("Successfully replaced 1"), run.stdout + run.stderr
        assert [line[:18] for line in said[-4:-2]] == ["Successfully wrote"] * 2, run.stdout
        assert "No space left on device" in said[-2] and said[-1] == "['kept.txt'] ['kept']"
        assert (temp_test_dir / "source.txt").read_text() == "a = 3\n"
        assert (temp_test_dir / "bound.txt").read_text() == "under the mount\n"
        assert sorted(os.listdir(temp_test_dir)) == ["bound.txt", "full", "source.txt"]

    @posix_only("the names Windows refuses are others")
    def test_a_rewrite_of_a_name_the_system_refuses_runs_as_written(self, temp_test_dir):
        """A ``sed -i`` or python rewrite becomes an ``Edit`` only where the system finds its file.

        Review 2026-11-11 S4r2: ``Path.resolve`` drops a ``..`` along with
        a name that is no directory, so ``sed -i s/a/b/ nosub/../f`` (sed:
        no such file) changed ``f``; a name the system refuses (too long,
        a lone surrogate, in a directory it may not search) raised
        ``OSError`` / ``UnicodeEncodeError`` out of ``Bash``.
        """
        ut = UsefulTools(work_dir=str(temp_test_dir))
        path = temp_test_dir / "f"
        path.write_text("a\n")
        (temp_test_dir / "g").write_text("a\n")
        locked = temp_test_dir / "locked"
        locked.mkdir()
        (locked / "f").write_text("a\n")
        ut.Read(str(path))
        sed = ("sed -i" if host_sed_is_gnu() else "sed -i ''") + " 's/a/b/' "
        python = (
            "python3 - <<'EOF'\ns = open({0}).read()\ns = s.replace('a', 'b')\n"
            "open({0}, 'w').write(s)\nEOF"
        )
        cases = [
            (sed + "nosub/../f", "No such file or directory"),
            (sed + "g/../f", "Not a directory"),
            (python.format("'nosub/../f'"), "No such file or directory"),
            (sed + "p" * 5000, "File name too long"),
            (python.format(repr("p" * 5000)), "File name too long"),
            (python.format("'\\ud800'"), "surrogates not allowed"),
            (sed + "\ud800", "surrogates not allowed"),
        ]
        if not is_root():
            cases.append((sed + "locked/f", "Permission denied"))
        locked.chmod(0)
        try:
            for command, said in cases:
                out = ut.Bash(command, "fails as written")
                assert out.startswith("Error") and said in out, out
                assert path.read_text() == "a\n"
        finally:
            locked.chmod(0o700)
        applied = ut.Bash(python.format("'locked/../f'"), "a real directory")
        assert applied.startswith("Applied as Edit(") and path.read_text() == "b\n", applied

    @posix_only("a symbolic link, and the cd of the host's bash")
    def test_a_rewrite_after_a_cd_is_applied_only_where_bash_walks_to_the_directory(
        self, temp_test_dir,
    ):
        """``cd DIR && <rewrite>`` becomes a ``Write``/``Edit`` only where bash's ``cd`` gets there.

        Review 2026-11-11 S4r3: the ``cd`` was judged by its normalised
        path, so after ``cd nosub/..`` (bash: no such directory, nothing
        runs) the rewrite was applied, and after ``cd link/../gone/../both``,
        which bash enters through the link, the ``both`` beside the link
        was edited.  Bash drops a ``..`` only after a directory.
        """
        ut = UsefulTools(work_dir=str(temp_test_dir))
        names = ("f", "sub/h", "both/f", "other/both/f", "other/gone/f", "other/sub/f")
        for name in names:
            path = temp_test_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("a\n")
            ut.Read(str(path))
        (temp_test_dir / "link").symlink_to("other/sub")
        sed = ("sed -i" if host_sed_is_gnu() else "sed -i ''") + " 's/a/b/' {0}"
        python = (
            "python3 - <<'EOF'\ns = open('{0}').read()\ns = s.replace('a', 'b')\n"
            "open('{0}', 'w').write(s)\nEOF"
        )
        rewrites = (sed, sed + " && echo ran > out", python, "cat > {0} <<'EOF'\nb\nEOF",
                    "tee -a {0} <<'EOF' > /dev/null\nb\nEOF")
        for operand, target in (("nosub/..", "f"), ("f/..", "f"), ("nosub/../sub", "h")):
            refusal = useful_tools.CD_VERDICT.format(cwd=f"{temp_test_dir}/{operand}")
            for rewrite in rewrites:
                command = f"cd {operand} && " + rewrite.format(target)
                assert ut.Bash(command, "the cd fails") == refusal, command
                assert _bash_exit(command, temp_test_dir) != 0, command
                assert _texts(temp_test_dir, names) == ["a\n"] * 6, command
                assert os.listdir(temp_test_dir / "sub") == ["h"], command
        through_the_link = ("link/../gone/../both", "link/../gone")
        for operand in through_the_link:
            out = ut.Bash(f"cd {operand} && " + sed.format("f"), "only bash knows where")
            assert out == useful_tools.CD_LINK_VERDICT.format(
                cwd=f"{temp_test_dir}/{operand}"), out
            assert _texts(temp_test_dir, names) == ["a\n"] * 6, operand
        for operand in through_the_link:  # bash enters the directory beyond the link
            assert _bash_exit(f"cd {operand} && " + sed.format("f"), temp_test_dir) == 0
        assert _texts(temp_test_dir, names) == ["a\n", "a\n", "a\n", "b\n", "b\n", "a\n"]
        applied = ut.Bash(f"cd sub/.. && {sed.format('f')} && echo ran > out", "bash walks it")
        assert applied.startswith("Applied as Edit(") and (temp_test_dir / "out").exists(), applied
        applied = ut.Bash("cd link/../both && cat > f <<'EOF'\nb\nEOF", "beside the link")
        assert applied.startswith("Applied as Write("), applied
        assert _texts(temp_test_dir, names) == ["b\n", "a\n", "b\n", "b\n", "b\n", "a\n"]
        # Two operands: bash 4.4 and later fail the cd, an older one enters the first.
        command = "cd sub extra && " + sed.format("h")
        assert not ut.Bash(command, "as written").startswith("Applied"), command
        edited = (temp_test_dir / "sub" / "h").read_text() == "b\n"
        assert edited == (_bash_exit(command, temp_test_dir) == 0)

    def test_a_command_that_cannot_be_spawned_is_an_error_on_every_launch_path(
        self, temp_test_dir,
    ):
        """A command ``Popen`` refuses (a lone surrogate or a NUL byte) is answered.

        Windows accepts a lone surrogate in its native command-line string,
        so that case is exercised only where the OS encoder rejects it.
        The job starter caught ``OSError`` only, so its ``ValueError``
        left ``Bash`` for a background call and for a foreground one over
        the detach grace (the default), and each failed start left its
        empty log.
        """
        ut = UsefulTools(work_dir=str(temp_test_dir))
        commands = ("echo a\0b",) if sys.platform == "win32" else ("echo \ud800", "echo a\0b")
        for command in commands:
            for out in (
                ut.Bash(command, "over the detach grace"),
                ut.Bash(command, "under the detach grace", timeout_seconds=60),
                ut.Bash(command, "as a job", background=True),
                ut.Bash(command, "as a job at the deadline", on_timeout="background"),
            ):
                assert out.startswith("Error: "), out
        assert ut._jobs == {} and os.listdir(temp_test_dir / "tmp" / "bash_jobs") == []

    def test_bash_reads_no_input_from_the_process_that_runs_it(self, temp_test_dir):
        """A command that reads its input gets end of file, as a background job does.

        The foreground shell inherited the stdin of the process running
        the tool: under an open pipe, or the terminal of the CLI,
        ``cat``, ``ssh`` or a prompt waited for input until the timeout.
        """
        script = (
            "import sys\n"
            "from kiss.agents.sorcar.useful_tools import UsefulTools\n"
            "tools = UsefulTools(work_dir=sys.argv[1])\n"
            "print(tools.Bash('cat; echo done', 'reads its input', timeout_seconds=20))\n"
        )
        read_end, write_end = os.pipe()  # never written: its reader waits
        try:
            run = subprocess.run(
                [sys.executable, "-c", script, str(temp_test_dir)],
                stdin=read_end, capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            )
        finally:
            os.close(read_end)
            os.close(write_end)
        assert run.stdout.strip() == "done", run.stdout + run.stderr

    @posix_only("a file has no group on Windows")
    def test_write_and_edit_keep_the_group_of_the_file(self, temp_test_dir):
        """The file a Write or an Edit leaves has the group of the one it replaced.

        The temporary file renamed over it had the group a new file
        gets, so the file changed group (and owner, for another
        user's file) with every edit.
        """
        groups = [group for group in os.getgroups() if group != temp_test_dir.stat().st_gid]
        if not groups:
            pytest.skip("the user is in one group")
        ut = UsefulTools(work_dir=str(temp_test_dir))
        path = temp_test_dir / "shared.txt"
        path.write_text("a = 1\n")
        os.chown(path, -1, groups[0])
        path.chmod(0o664)
        ut.Read(str(path))
        assert ut.Edit(str(path), "a = 1", "a = 2").startswith("Successfully replaced 1")
        assert ut.Write(str(path), "a = 3\n").startswith("Successfully wrote")
        assert path.read_text() == "a = 3\n"
        assert (path.stat().st_gid, path.stat().st_mode & 0o777) == (groups[0], 0o664)

    @pytest.mark.skipif(sys.platform == "win32", reason="bash is found by path on Windows")
    def test_bash_runs_bash_not_the_posix_shell(self, temp_test_dir):
        """``Bash`` runs bash, so brace expansion and arrays work in every launch path.

        ``shell=True`` ran ``/bin/sh`` (dash on Debian), which expands no
        braces and has no arrays, so the tool named Bash failed on, or
        silently changed, valid bash commands (A14-3 fable).
        """
        ut = UsefulTools(work_dir=str(temp_test_dir))
        braces = "printf '<%s>\\n' {a,b}"
        array = "a=(first second); printf '%s\\n' \"${a[1]}\""
        assert ut.Bash(braces, "braces", timeout_seconds=30).startswith("<a>\n<b>\n")
        assert ut.Bash(array, "array", timeout_seconds=30).startswith("second\n")
        job = re.search(r"job (j[0-9a-f]{8})", ut.Bash(braces, "bg", background=True))
        assert job is not None
        assert ut.bash_job(job.group(1), action="wait", timeout_seconds=30).endswith("<a>\n<b>\n")
        parallel = ut.run_commands_parallel(json.dumps([braces, array]), description="both")
        assert "<a>\n<b>\n" in parallel and parallel.endswith("\nsecond"), parallel


class TestConcurrentEdits:
    """Two agents (two tool instances, two threads) editing one file lose no change.

    ``Edit`` is a read-modify-write of the whole file; without a lock a
    sibling's write landing between another's read and write was undone
    by that write, both calls reporting success (A10-2).  Edit has no
    synchronisation point a test could hold without a test double, so
    the race is provoked by volume: forty edits per thread on a 4 MB
    file, every one of which must survive.
    """

    def test_concurrent_edits_of_one_file_keep_every_change(self, temp_test_dir):
        path = temp_test_dir / "shared.txt"
        edits_per_thread = 40
        lines = [f"{prefix}{i}=0\n" for prefix in "ab" for i in range(edits_per_thread)]
        path.write_text("".join(lines) + ("x" * 200 + "\n") * 20000)
        tools = [UsefulTools(work_dir=str(temp_test_dir)) for _ in "ab"]
        for tool in tools:
            assert "a0=0" in tool.Read(str(path), max_lines=2)
        results: dict[str, list[str]] = {}

        def edit(prefix: str, tool: UsefulTools) -> None:
            results[prefix] = [
                tool.Edit(str(path), f"{prefix}{i}=0\n", f"{prefix}{i}=1\n")
                for i in range(edits_per_thread)
            ]

        threads = [
            threading.Thread(target=edit, args=(prefix, tool))
            for prefix, tool in zip("ab", tools, strict=True)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert all(r.startswith("Successfully") for rs in results.values() for r in rs), results
        text = path.read_text()
        lost = [
            f"{prefix}{i}" for prefix in "ab" for i in range(edits_per_thread)
            if f"{prefix}{i}=1\n" not in text
        ]
        assert lost == [], f"edits undone by a concurrent Edit: {lost}"


def _stop_at_entry(frame_locals: dict[str, Any]) -> bool:
    """Set the Stop of the tools whose method is entered, without holding the thread."""
    frame_locals["self"].stop_event.set()
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test (signals, bash scripts)")
class TestStopEvent:
    """Tests that stop_event kills child processes promptly."""

    def test_stop_event_kills_streaming_child(self, temp_test_dir):
        """When stop_event is set, the streaming Bash child must be killed."""
        import threading
        import time

        stop_event = threading.Event()
        streamed: list[str] = []
        ut = UsefulTools(stream_callback=streamed.append, stop_event=stop_event)

        pid_file = temp_test_dir / "stop_child.pid"
        script = temp_test_dir / "stop_target.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nwhile true; do echo tick; sleep 0.2; done\n"
        )
        script.chmod(0o755)

        child_pid = None

        def set_stop_after_start():
            nonlocal child_pid
            for _ in range(50):
                time.sleep(0.1)
                if pid_file.exists():
                    # Poll for parseable content, not mere existence:
                    # the shell redirect creates the file empty before
                    # echo writes the PID.
                    pid_text = pid_file.read_text().strip()
                    if pid_text.isdigit():
                        child_pid = int(pid_text)
                        break
            if child_pid:
                stop_event.set()

        t = threading.Thread(target=set_stop_after_start, daemon=True)
        t.start()
        ut.Bash(str(script), "stoppable", timeout_seconds=30)
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.5)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived stop_event!"

    def test_stop_event_kills_nonstreaming_child(self, temp_test_dir):
        """When stop_event is set, the non-streaming Bash child must be killed."""
        import threading
        import time

        stop_event = threading.Event()
        ut = UsefulTools(stop_event=stop_event)

        pid_file = temp_test_dir / "stop_child_ns.pid"
        script = temp_test_dir / "stop_target_ns.sh"
        script.write_text(
            f"#!/bin/bash\necho $$ > {pid_file}\nsleep 100\n"
        )
        script.chmod(0o755)

        child_pid = None

        def set_stop_after_start():
            nonlocal child_pid
            for _ in range(50):
                time.sleep(0.1)
                if pid_file.exists():
                    # Poll for parseable content, not mere existence:
                    # the shell redirect creates the file empty before
                    # echo writes the PID.
                    pid_text = pid_file.read_text().strip()
                    if pid_text.isdigit():
                        child_pid = int(pid_text)
                        break
            if child_pid:
                stop_event.set()

        t = threading.Thread(target=set_stop_after_start, daemon=True)
        t.start()
        ut.Bash(str(script), "stoppable", timeout_seconds=30)
        t.join(timeout=5)

        if child_pid is None:
            pytest.skip("Script didn't start in time")

        time.sleep(0.5)
        alive = False
        try:
            os.kill(child_pid, 0)
            alive = True
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        assert not alive, f"Child {child_pid} survived stop_event!"

    def test_stop_set_before_a_parallel_call_starts_no_command(self, temp_test_dir):
        """A pool entered with the task's stop event already set starts nothing (A10-5).

        The workers used to be submitted before the stop was looked at,
        so a quick command ran (and was then killed) after the Stop.
        """
        stop_event = threading.Event()
        stop_event.set()
        marker = temp_test_dir / "started.marker"
        ut = UsefulTools(stop_event=stop_event, work_dir=str(temp_test_dir))
        report = ut.run_commands_parallel(json.dumps([f"touch {marker}", f"touch {marker}"]))
        assert report.count(NOT_STARTED) == 2, report
        assert not marker.exists(), "a command ran after the task was stopped"

    def test_stop_before_a_bash_launch_starts_no_command(self, temp_test_dir):
        """A Bash call starts nothing and leaves no job once a Stop precedes its launch (X-STOP-1).

        Each launch path spawned its shell before the thread owning it looked at the Stop: under
        load the command ran first, and the paths that start a job returned it as running.
        """
        stopped = f"Error (exit code -1):\n{NOT_STARTED}"
        stop_event = threading.Event()
        stop_event.set()
        ut = UsefulTools(stop_event=stop_event, work_dir=str(temp_test_dir))
        launches: list[dict[str, Any]] = [{}, {"timeout_seconds": 30}, {"background": True},
                                          {"until": "x"}, {"on_timeout": "background"}]
        for options in launches:
            assert ut.Bash("touch started.marker", "touch", **options) == stopped, options
        panel = UsefulTools(work_dir=str(temp_test_dir))  # the tool panel's Stop raises
        token = tool_interrupt.new_tool_call("Bash")
        tool_interrupt.register_tool_call(token)
        try:
            tool_interrupt.interrupt_tool_call(token.thread_ident, call_id=token.call_id)
            with pytest.raises(tool_interrupt.ToolCallInterrupted):
                panel.Bash("touch started.marker", "touch", background=True)
        finally:
            tool_interrupt.unregister_tool_call(token)
        assert not any(temp_test_dir.iterdir()) and not ut._jobs and not panel._jobs
        # A Stop that lands as a long sleep's wait on the running job starts: no rest either.
        waiting = UsefulTools(stop_event=threading.Event(), work_dir=str(temp_test_dir))
        waiting.Bash("sleep 30", "job", background=True)
        with Pause(UsefulTools._wait_for_job.__code__, _stop_at_entry):
            assert waiting.Bash("sleep 40; touch started.marker", "rest") == stopped
        assert not (temp_test_dir / "started.marker").exists() and len(waiting._jobs) == 1

    def test_stop_kills_a_background_jobs_orphans_after_its_shell_exited(self, temp_test_dir):
        """A job's descendant that outlives the job's shell is killed by a later task Stop (A10-6).

        ``sleep 100 &`` stays in the job's process group after the shell
        exits; the watcher used to end with the shell, leaving nothing to
        kill the group when the Stop came.
        """
        stop_event = threading.Event()
        ut = UsefulTools(stop_event=stop_event, work_dir=str(temp_test_dir))
        pid_file = temp_test_dir / "orphan.pid"
        script = temp_test_dir / "orphan.sh"
        script.write_text(f"sleep 100 &\necho $! > {pid_file}\n")
        started = ut.Bash(f"bash {script}", "orphan", background=True)
        match = re.search(r"job (j[0-9a-f]+)", started)
        assert match is not None, started
        job_id = match.group(1)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if "exited with code 0" in ut.bash_job(job_id) and pid_file.is_file():
                break
            time.sleep(0.05)
        assert "exited with code 0" in ut.bash_job(job_id)
        orphan = int(pid_file.read_text().strip())
        assert pid_alive(orphan)
        try:
            stop_event.set()
            deadline = time.monotonic() + 10
            while pid_alive(orphan) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not pid_alive(orphan), f"the job's orphan {orphan} survived the task Stop"
        finally:
            with suppress(OSError):
                os.kill(orphan, signal.SIGKILL)


class TestStreamingTimeout:
    """Test that streaming Bash correctly reports timeout."""

    def test_streaming_bash_timeout(self, temp_test_dir):
        """Streaming Bash must return timeout error when command exceeds timeout."""
        streamed: list[str] = []
        ut = UsefulTools(stream_callback=streamed.append)
        result = ut.Bash("sleep 10", "should timeout", timeout_seconds=0.5)
        assert "timeout" in result.lower()

    def test_timer_firing_after_completion_keeps_output(self, temp_test_dir):
        """A timer that fires after the command already finished must not clobber output.

        The command completes almost instantly, but a slow stream callback
        delays the loop past the timeout, so the Timer deterministically
        fires in the gap between command completion and ``timer.cancel()``.
        The successful output must be preserved, not replaced with a
        timeout error.
        """
        def slow_stream(_line: str) -> None:
            time.sleep(1.0)

        ut = UsefulTools(stream_callback=slow_stream)
        result = ut.Bash("echo hi", "quick command", timeout_seconds=0.5)
        assert "timeout" not in result.lower()
        assert "hi" in result


class TestBashUntil:
    """``Bash(until=<regex>)`` re-runs the command inside one call until the regex matches."""

    COUNT = "n=$(cat n); n=$((n+1)); echo $n > n; echo run $n"
    # The elapsed time a report prints: each call's own monotonic bounds check its
    # size; a loaded Windows box (49 suite splits at once) adds seconds to a run.
    SECONDS = r"\d+\.\d s"

    def test_until_polls_until_the_regex_matches_the_output(self, temp_test_dir, monkeypatch):
        (temp_test_dir / "n").write_text("0")
        spools = temp_test_dir / "spools"  # where the searched output is spooled for the child
        spools.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(spools))
        streamed: list[str] = []
        ut = UsefulTools(stream_callback=streamed.append, work_dir=str(temp_test_dir))
        before = time.monotonic()
        out = ut.Bash(self.COUNT, "poll", until=r"run 2$", poll_seconds=1, timeout_seconds=30)
        elapsed = time.monotonic() - before
        assert 1.0 <= elapsed < 10, elapsed  # one pause of poll_seconds between the runs
        assert re.fullmatch(
            rf"run 2\n\n\[until 'run 2\$' matched after 2 run\(s\) and {self.SECONDS}\]", out,
        ), out
        assert (temp_test_dir / "n").read_text() == "2\n"  # no run after the match
        assert "".join(streamed).count("run ") == 2  # each run streams its output
        assert list(spools.iterdir()) == []  # each run's spool file is removed
        # The regex is matched against the command's output, not the exit-code line.
        out = ut.Bash("echo nope >&2; exit 3", "fail", until="exit code", timeout_seconds=2.5)
        assert out.startswith("Error (exit code 3):\nnope\n")
        assert "did not match after" in out

    def test_until_ends_at_the_deadline_with_the_last_run_shown(self, temp_test_dir):
        ut = UsefulTools(work_dir=str(temp_test_dir))
        before = time.monotonic()
        # poll_seconds below 1 is raised to 1: at most three runs fit in 3 s, not thirty (a
        # fourth cannot start before the deadline with 1 s pauses; a third does only while a
        # run and its search cost under 0.5 s, which a loaded Windows box's bash spawn misses;
        # there the third run can also start late and be killed at the deadline).
        out = ut.Bash("echo nope", "poll", until="never", poll_seconds=0.1, timeout_seconds=3)
        assert 2.9 <= time.monotonic() - before < 10
        assert re.fullmatch(
            rf"(nope\n|Error: Command execution timeout.*)\n\[until 'never' did not match "
            rf"after [23] run\(s\) and {self.SECONDS}: the call's deadline \(timeout_seconds\) "
            r"passed\]", out, re.DOTALL,
        ), out
        # A run killed at the deadline is the last one, its partial output kept.  Its
        # timeout is the time left, to the hundredth (1.49 s once the run starts late).
        out = ut.Bash("echo part; sleep 10", "slow", until="never", timeout_seconds=1.5)
        killed = re.match(r"Error: Command execution timeout after ([\d.]+)s\. Output before", out)
        assert killed is not None and 1.0 <= float(killed.group(1)) <= 1.5, out
        assert re.search(
            rf"after 1 run\(s\) and {self.SECONDS}: the call's deadline \(timeout_seconds\) "
            r"passed\]$", out,
        ), out
        # A long sleep with a job running is spent waiting on the job, then the poll runs
        # (both notes in the result); the poll's deadline is the call's, absolute, so the
        # wait counts against it: a poll handed a deadline 1.2 s away ends there, not after
        # poll_seconds.
        ut.Bash("sleep 1", "job", background=True)
        before = time.monotonic()
        out = ut.Bash("sleep 60; echo after", "poll", until="after", timeout_seconds=100)
        assert time.monotonic() - before < 10
        assert "[until 'after' matched after 1 run(s)" in out and "Waited on job " in out, out
        before = time.monotonic()
        out = ut._poll_until(
            "echo nope", re.compile("never"), 30, before + 1.2, 8000, str(temp_test_dir),
        ).text
        assert 1.1 <= time.monotonic() - before < 5
        assert re.search(
            rf"did not match after 1 run\(s\) and {self.SECONDS}: the call's deadline "
            r"\(timeout_seconds\) passed\]$", out,
        ), out
        # The search itself is bounded by the deadline: a pattern that backtracks for
        # minutes runs in a child the poll kills at the deadline (as bash_job's wait does).
        before = time.monotonic()
        out = ut.Bash(f"printf '%s' {'a' * 60}!", "regex", until=r"(a+)+$", timeout_seconds=1)
        assert time.monotonic() - before < 4
        assert re.search(
            rf"did not match after 1 run\(s\) and {self.SECONDS}: the call's deadline", out,
        ), out

    def test_until_does_not_search_an_output_too_long_to_keep_whole(self, temp_test_dir):
        """Above the capture budget only a head and a tail are kept: no match is claimed."""
        ut = UsefulTools(work_dir=str(temp_test_dir))
        fill = "head -c 1600000 /dev/zero | tr '\\0' {}"
        command = f"{fill.format('a')}; echo MIDDLE; {fill.format('b')}"
        out = ut.Bash(command, "big", until="MIDDLE|ab", poll_seconds=1, timeout_seconds=30)
        assert re.search(
            r"\[until 'MIDDLE\|ab' was not searched after 1 run\(s\) and \d+\.\d s: the output "
            r"is too long to keep whole \([\d,]+ characters dropped\); poll with a command that "
            r"prints less \(grep, tail\)\]$", out,
        ), out[-300:]

    def test_until_stops_with_the_stop_event(self, temp_test_dir):
        stop = threading.Event()
        ut = UsefulTools(stop_event=stop, work_dir=str(temp_test_dir))
        threading.Timer(0.5, stop.set).start()
        before = time.monotonic()
        out = ut.Bash("echo once", "poll", until="never", poll_seconds=30, timeout_seconds=60)
        assert time.monotonic() - before < 5
        assert out.startswith("once\n\n[until 'never' did not match after 1 run(s) and 0.")
        assert "deadline" not in out

    def test_until_is_refused_for_a_background_call_a_bad_regex_or_a_rewrite(self, temp_test_dir):
        ut = UsefulTools(work_dir=str(temp_test_dir))
        out = ut.Bash("touch ran", "bg", background=True, until="x")
        assert out == (
            "Error: until= polls a foreground command; with background=True follow the "
            "job's log with bash_job(id, \"wait\", until=<regex>) instead."
        )
        assert not (temp_test_dir / "ran").exists()
        out = ut.Bash("touch ran", "bad", until="(")
        assert out.startswith("Error: until is not a valid regular expression: missing )")
        # A NUL cannot reach the search child as a program argument: every run "did not match".
        out = ut.Bash("touch ran", "bad", until="x|\x00", poll_seconds=1, timeout_seconds=2)
        assert out == "Error: until is not a valid regular expression: embedded null byte"
        assert not (temp_test_dir / "ran").exists()
        # ...before a shell rewrite of a file is applied as the Edit it stands for.
        f = temp_test_dir / "f.txt"
        f.write_text("old\n")
        ut.Read(str(f))
        assert ut.Bash("sed -i 's/old/new/' f.txt", "edit", until="(").startswith("Error: until")
        out = ut.Bash("sed -i 's/old/new/' f.txt", "edit", until="new", background=True)
        assert out.startswith("Error: until= polls")
        assert f.read_text() == "old\n"
        # A valid regex polls a command run as written: one applied as the Write/Edit it
        # stands for is refused before any edit, with or without a rest to poll (the rest
        # ran once, unpolled, and the redirect's follow-up lost until= and poll_seconds=:
        # S9/F2); under an agent (redirect) too, where the refusal is returned, not raised.
        # The file is named as the shell path maps back to the host (``host_path``:
        # ``C:/Users/...`` on Windows, where Git Bash's ``/c/Users/...`` ran the command).
        shown = json.dumps(f.as_posix())
        before = time.monotonic()
        out = ut.Bash(
            "sed -i 's/old/new/' f.txt && echo READY", "edit", until="NEVER", poll_seconds=1,
            timeout_seconds=3,
        )
        assert time.monotonic() - before < 1, out
        assert out == (
            "Error: until= re-runs a command as written, but this one is applied as "
            f'Edit(file_path={shown}, old_string="old", new_string="new") instead of '
            'running: make that call, then poll "echo READY" with until= in a Bash call '
            "of its own."
        )
        out = ut.Bash("sed -i 's/old/new/' f.txt", "edit", until="new", timeout_seconds=3)
        assert out.startswith("Error: until= re-runs") and out.endswith("make that call."), out
        ut.redirect = True
        out = ut.Bash(
            "cat >> f.txt <<'EOF' && echo READY\nmore\nEOF", "append", until="READY",
            timeout_seconds=3,
        )
        ut.redirect = False
        assert out.startswith("Error: until= re-runs a command as written, but this one is")
        assert f"applied as Write(file_path={shown}, content=" in out, out
        assert out.endswith('then poll "echo READY" with until= in a Bash call of its own.')
        assert f.read_text() == "old\n"
        # A rewrite the guards let run as written (a sed -i through a hard link is not
        # converted) is polled whole, as documented.  BSD sed (macOS) wants the backup
        # suffix of -i as a word of its own; Git Bash's sed is GNU whatever the Windows
        # PATH holds.
        os.link(f, temp_test_dir / "g.txt")
        in_place = "sed -i" if sys.platform == "win32" or host_sed_is_gnu() else "sed -i ''"
        out = ut.Bash(
            f"{in_place} 's/old/new/' f.txt && cat f.txt", "edit", until="new", timeout_seconds=3,
        )
        assert re.search(r"\[until 'new' matched after 1 run\(s\) and 0\.\d s\]$", out), out
        assert f.read_text() == "new\n"


_REWRITE_FILES = (str(useful_tools.__file__), os.path.dirname(str(pathlib.__file__)))


def _in_tool_code(frame: Any) -> bool:
    """Whether *frame* runs the tool's own code or ``pathlib`` (the in-place ``write_text``)."""
    return bool(frame.f_code.co_filename.startswith(_REWRITE_FILES))


def _run_injecting_at(
    run: Callable[[], str], nth: int, counted: Callable[[Any], bool] = _in_tool_code,
) -> int:
    """Run *run* raising ``KeyboardInterrupt`` at its *nth* line event on this thread.

    Only the line events of the frames *counted* accepts count, by
    default the tool's own code and ``pathlib``: a trace event comes BEFORE its line
    runs, so an injection at the first line of a ``finally`` body skips
    that body, which a real injection (delivered after a call returns)
    cannot do to the lock helper's release.  ``nth`` past the last event
    injects nothing.

    Returns:
        The number of line events *run* produced before it returned or raised.
    """
    count = 0

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal count
        if event == "line" and counted(frame):
            count += 1
            if count == nth:
                raise KeyboardInterrupt(f"injected Stop at line event {nth}")
        return tracer

    sys.settrace(tracer)
    try:
        run()
    except KeyboardInterrupt:
        pass
    finally:
        sys.settrace(None)
    return count


def _swap_for_fifo_at_open(
    path: Path,
    run: Callable[[], str],
    outcome: dict[str, Any],
    nth: int = 1,
    reader: bool = False,
) -> None:
    """Run *run* on this thread, replacing *path* with a FIFO right before its *nth* ``open``.

    The swap happens in a profile hook, on this thread and on the one
    the tool reads or writes the file on, at the ``c_call`` of an open
    (``io.open`` or ``os.open``) whose calling frame holds *path* as its
    ``resolved`` (the tool's own helpers) or ``self`` (``pathlib``),
    after every ``is_file`` check *run* made and before the open takes
    effect: the interval another actor needs to swap the file.  Opens
    of other files (of ``.git`` while a path is resolved, of a spool)
    do not count.  *nth* skips the
    opens before the one to race (a binary Read's text attempt makes
    two, the descriptor and its text wrapper, before the binary read's
    own).  With *reader* the FIFO gets a reader, so an open for writing
    succeeds instead of failing at once (``ENXIO``) and only the
    descriptor's type check can refuse it.  *outcome* gets the result
    or exception of *run*, the name of the function whose open the swap
    preceded (``swapped_in``) and the reader's descriptor.
    """
    opens = 0

    def hook(frame: Any, event: str, arg: Any) -> None:
        nonlocal opens
        if event != "c_call" or getattr(arg, "__name__", "") != "open":
            return
        if frame.f_locals.get("resolved", frame.f_locals.get("self")) != path:
            return
        opens += 1
        if opens == nth:
            path.unlink()
            os.mkfifo(path)
            outcome["swapped_in"] = frame.f_code.co_name
            if reader:
                outcome["reader"] = os.open(path, os.O_RDONLY | os.O_NONBLOCK)

    sys.setprofile(hook)
    threading.setprofile(hook)
    try:
        outcome["result"] = run()
    except BaseException as e:  # noqa: BLE001 — reported to the test
        outcome["exc"] = e
    finally:
        threading.setprofile(None)
        sys.setprofile(None)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file semantics")
class TestInterruptedWrites:
    """A Stop landing in or around Write/Edit leaves the file whole and is not deferred."""

    @pytest.mark.parametrize("tool", ["Write", "Edit"])
    def test_a_stop_at_any_check_point_leaves_the_old_text_or_the_new(
        self, temp_test_dir: Path, tool: str,
    ) -> None:
        """Injected at every line event of a rewrite in turn, a Stop never leaves a truncated file.

        ``Path.write_text`` truncated the file at its open and wrote at
        its next statement, so a Stop injected between the two emptied
        the file for good and a concurrent Read saw ``(file is empty)``
        (A14-2).  No stray file may be left beside it either.
        """
        path = temp_test_dir / "whole.txt"
        old, new = "old text\n" * 50, "new text\n" * 50
        ut = UsefulTools(work_dir=str(temp_test_dir))
        path.write_text(old)
        assert "old text" in ut.Read(str(path))

        def run() -> str:
            if tool == "Write":
                return ut.Write(str(path), new)
            return ut.Edit(str(path), "old", "new", replace_all=True)

        total = _run_injecting_at(run, 0)
        assert path.read_text() == new and total > 0
        for nth in range(1, total + 1):
            path.write_text(old)
            _run_injecting_at(run, nth)
            text = path.read_text()
            assert text in (old, new), f"check point {nth}/{total} left {text[:30]!r}..."
            assert [p.name for p in temp_test_dir.iterdir()] == [path.name], (
                f"check point {nth}/{total} left a stray file"
            )

    @pytest.mark.parametrize("tool", ["Read", "Read binary", "Write"])
    def test_a_stop_at_any_check_point_leaves_no_descriptor_open(
        self, temp_test_dir: Path, tool: str,
    ) -> None:
        """Injected at every line event of a file tool in turn, a Stop leaves the file closed.

        The file was opened to a bare descriptor that only a local
        variable knew until a ``with`` took it over: a Stop between the
        two left it open for the life of the daemon (B/S13-F2).  A
        failing ``fstat`` leaked it the same way; it cannot be produced
        without a test double, as it never fails on a descriptor just opened.
        """
        path = temp_test_dir / ("shot.png" if tool == "Read binary" else "text.txt")
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64) if tool == "Read binary" else b"old\n")
        os.link(path, temp_test_dir / "other-name")  # a Write then goes in place, through the link
        ut = UsefulTools(work_dir=str(temp_test_dir))
        ut.Read(str(path))

        def run() -> str:
            if tool == "Write":
                return ut.Write(str(path), "new\n")
            return ut.Read(str(path), force=True)

        def descriptors() -> int:
            target, count = path.stat(), 0
            for name in os.listdir("/dev/fd"):
                with suppress(OSError):  # the listing's own descriptor is closed by now
                    held = os.fstat(int(name))
                    count += (held.st_dev, held.st_ino) == (target.st_dev, target.st_ino)
            return count

        total = _run_injecting_at(run, 0)
        assert total > 0 and descriptors() == 0
        for nth in range(1, total + 1):
            _run_injecting_at(run, nth)
            assert descriptors() == 0, f"check point {nth}/{total} left the file open"

    def test_a_stop_in_the_start_of_a_file_thread_leaves_no_write_to_come(
        self, temp_test_dir: Path,
    ) -> None:
        """A Stop that takes a Write out of the start of its file thread leaves nothing behind it.

        The file is read and written on a thread of its own (B/S13-F2).
        Stopped right after that thread was created, the call returned
        and the thread, which had not begun, wrote the file afterwards,
        with the file lock released.
        """
        path = temp_test_dir / "handed.txt"
        path.write_text("old\n")
        os.link(path, temp_test_dir / "other-name")  # the Write goes in place, through the link
        ut = UsefulTools(work_dir=str(temp_test_dir))
        ut.Read(str(path))
        lines, first = inspect.getsourcelines(threading.Thread.start)
        created = first + next(i for i, line in enumerate(lines) if "_started.wait()" in line)

        def thread_created(frame: Any) -> bool:
            return frame.f_code is threading.Thread.start.__code__ and frame.f_lineno == created

        def run() -> str:
            return ut.Write(str(path), "new\n")

        total = _run_injecting_at(run, 0, thread_created)
        assert total > 0 and path.read_text() == "new\n"
        for nth in range(1, total + 1):
            path.write_text("old\n")
            threads = set(threading.enumerate())
            _run_injecting_at(run, nth, thread_created)
            at_return = path.read_text()
            deadline = time.monotonic() + 10
            while set(threading.enumerate()) - threads and time.monotonic() < deadline:
                time.sleep(0.01)
            assert path.read_text() == at_return, (
                f"thread start {nth}/{total}: the write landed after the stopped call returned"
            )

    def test_a_stop_reaches_a_write_parked_on_the_held_lock(self, temp_test_dir: Path) -> None:
        """A Write waiting for the file lock another task's write holds still gets its Stop.

        The lock was a plain ``Lock`` taken with ``with``: a thread parked
        in its untimed ``acquire`` reached no check point, so the
        injected ``KeyboardInterrupt`` waited for the holder to release,
        however long its write took (A14-3).
        """
        path = temp_test_dir / "parked.txt"
        ut = UsefulTools(work_dir=str(temp_test_dir))
        lock = useful_tools._FILE_WRITE_LOCK
        parked = threading.Event()
        outcome: dict[str, Any] = {}

        def hook(frame: Any, event: str, arg: Any) -> None:
            # The profile hook runs inside the CALL, right before the C
            # ``acquire`` (``Lock.__enter__`` is the same function) runs:
            # an injection made once ``parked`` is set is pending while
            # the thread is inside it.
            if (
                event == "c_call"
                and getattr(arg, "__name__", "") in ("acquire", "__enter__")
                and getattr(arg, "__self__", None) is lock
            ):
                parked.set()

        def write() -> None:
            sys.setprofile(hook)
            try:
                outcome["result"] = ut.Write(str(path), "x\n")
            except BaseException as e:  # noqa: BLE001 — the injected Stop is expected
                outcome["exc"] = e
            finally:
                sys.setprofile(None)

        writer = threading.Thread(target=write, daemon=True)
        assert lock.acquire(timeout=5)
        try:
            writer.start()
            assert parked.wait(10), "the Write never reached the lock"
            assert writer.ident is not None
            assert tool_interrupt.inject_async_exception(writer.ident, KeyboardInterrupt) == 1
            writer.join(timeout=3)
            assert not writer.is_alive(), "the Stop waited for the lock holder"
        finally:
            lock.release()
            writer.join(timeout=10)
        assert isinstance(outcome.get("exc"), KeyboardInterrupt), outcome
        assert not path.exists()

    @pytest.mark.parametrize(
        ("tool", "nth", "reader", "expected"),
        [
            ("Read", 1, False, "could block forever"),
            ("Read binary", 3, False, "could block forever"),
            ("Edit", 1, False, "Error: File not found: "),
            ("Edit unread", 1, False, "has not been read in this session"),
            ("Write", 1, True, "is not a regular file (FIFO/device/socket); refusing to write"),
            ("Write append", 1, False, os.strerror(errno.ENXIO)),
            ("sed -i", 1, False, "not shown in this session. Read("),
            ("cat >>", 1, False, "refusing to append to it."),
        ],
    )
    def test_a_fifo_swapped_in_after_the_check_does_not_block(
        self, temp_test_dir: Path, tool: str, nth: int, reader: bool, expected: str,
    ) -> None:
        """A path swapped for a FIFO between the type check and the open is refused, not waited for.

        ``open`` of a FIFO with no writer (no reader, for a write) blocks
        in C for good, where no Stop lands; under the file lock that held
        every Write and Edit of the daemon (A14-2 fable).  Every open of
        a file the model names is the descriptor-checked one: the text
        and binary reads of Read, Edit's read (of an unread file too, for
        the region its refusal shows), the in-place write of a file with
        another hard link, the append a redirected ``>>`` Write makes, and
        the reads a ``sed -i`` or a ``cat >>`` here-document is converted
        with (S9/F1).
        """
        path = temp_test_dir / ("swap.png" if tool == "Read binary" else "swap.txt")
        if tool == "Read binary":
            path.write_bytes(b"\x89PNG\r\n\x1a\n\xff")
        else:
            path.write_text("old\n")
        if tool == "Write":
            os.link(path, temp_test_dir / "other.txt")  # written in place, through the link
        ut = UsefulTools(work_dir=str(temp_test_dir))
        if tool not in ("Edit unread", "sed -i"):
            assert not ut.Read(str(path)).startswith("Error")
        append = "cat >> swap.txt <<'EOF'\nmore\nEOF"
        arguments: dict[str, Any] = {}
        if tool == "Write append":
            ut.redirect = True
            with pytest.raises(ToolRedirect) as raised:
                ut.Bash(append, "append")
            ut.redirect = False
            arguments = raised.value.arguments
        outcome: dict[str, Any] = {}

        def run() -> str:
            if tool.startswith("Read"):
                return ut.Read(str(path), force=True)
            if tool.startswith("Edit"):
                return ut.Edit(str(path), "old", "new")
            if tool == "Write":
                return ut.Write(str(path), "new\n")
            if tool == "Write append":
                return ut.Write(**arguments)
            if tool == "sed -i":
                return ut.Bash("sed -i 's/old/new/' swap.txt", "edit")
            return ut.Bash(append, "append")

        worker = threading.Thread(
            target=_swap_for_fifo_at_open, args=(path, run, outcome, nth, reader), daemon=True,
        )
        worker.start()
        worker.join(timeout=5)
        try:
            assert not worker.is_alive(), f"{tool} blocked on the FIFO"
            assert path.is_fifo(), "the swap never happened"
            assert outcome.get("swapped_in") == "_open_regular", outcome
            assert useful_tools._FILE_WRITE_LOCK.acquire(timeout=5), "the lock stayed held"
            useful_tools._FILE_WRITE_LOCK.release()
            result = outcome.get("result", "")
            assert expected in result and (
                result.startswith("Error:") or tool == "sed -i" and result.startswith("REFUSED")
            ), outcome
            if tool == "Write":
                assert (temp_test_dir / "other.txt").read_text() == "old\n"
        finally:
            if "reader" in outcome:
                os.close(outcome["reader"])
            if worker.is_alive():
                with suppress(OSError):
                    os.close(os.open(path, os.O_RDWR | os.O_NONBLOCK))
                worker.join(timeout=10)
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
