# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``kiss-service-cgroup-escape`` block of the installers.

The remote webapp's Terminal tab is a shell the kiss-web daemon forks,
so on Linux it lives in the ``kiss-web.service`` control group, and so
does an ``./install.sh`` typed into it.  The script ends by running
``systemctl --user stop kiss-web.service``, which SIGTERMs every process
of that group: the install killed itself.  The block re-executes the
script in a transient scope of its own before any of that.  The curl
bootstrap ``scripts/install.sh`` (which holds the update lock and waits
for the root script) carries the same block; piped in by ``curl``, it
re-executes as ``bash -s`` reading the rest of itself from stdin.

The block is extracted verbatim and run under bash.  Most tests drive
it with a stub ``systemd-run`` (recording its argv and ``exec``-ing the
payload like the real ``--scope`` does) and a cgroup file of their own;
the last one reproduces the bug under the real systemd user manager
and is skipped where there is none.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

# The pty module itself is POSIX-only (a top-level ``import pty`` fails
# collection on Windows), so it is imported where it is used.
pytestmark = posix_only("pty, bash scripts, os.symlink and systemd-run")

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = [REPO / "install.sh", REPO / "scripts" / "install.sh"]
BEGIN = "# BEGIN: kiss-service-cgroup-escape"
END = "# END: kiss-service-cgroup-escape"


def _block(script: Path) -> str:
    text = script.read_text(encoding="utf-8")
    start = text.index(BEGIN)
    end = text.index(END, start) + len(END)
    return text[start:end] + "\n"


def test_both_installers_carry_the_same_block() -> None:
    assert _block(SCRIPTS[0]) == _block(SCRIPTS[1])
    # The bootstrap's copy runs before anything else, so a re-exec'd
    # ``bash -s`` resuming after it misses nothing.
    boot = SCRIPTS[1].read_text(encoding="utf-8")
    assert all(line.startswith("#") or not line for line in boot[: boot.index(BEGIN)].splitlines())


# Prints what a re-exec must preserve: the escaped unit, the pid (systemd-run
# ``exec``s in place) and the arguments.
_REPORT = 'echo "host=${_KISS_HOST_SERVICE:-} pid=$$ args=[$*]"\n'

# ``systemd-run`` stand-in: logs its argv, then runs the payload after
# ``--`` in place.  ``$PROBE_RC`` lets a test fail the ``true`` probe.
_SYSTEMD_RUN_STUB = """\
#!/bin/bash
printf '%s\\n' "$*" >> "$STUB_LOG"
while [ "$1" != "--" ]; do shift; done
shift
if [ "$1" = true ]; then exit "${PROBE_RC:-0}"; fi
exec "$@"
"""

KISS_WEB_CGROUP = "0::/user.slice/user-1006.slice/user@1006.service/app.slice/kiss-web.service\n"
PROBE = "--user --scope --quiet --collect -- true"
REEXEC = (
    "--user --scope --quiet --collect "
    "--description=install.sh (moved out of kiss-web.service) -- bash"
)


class Harness:
    """A copy of the block whose default cgroup file is the test's own."""

    def __init__(
        self,
        tmp_path: Path,
        cgroup_text: str | None,
        script: Path = SCRIPTS[0],
        with_stub: bool = True,
    ) -> None:
        self.dir = tmp_path
        self.cgroup = tmp_path / "cgroup"
        if cgroup_text is not None:
            self.cgroup.write_text(cgroup_text, encoding="utf-8")
        block = _block(script)
        assert block.count("${1:-/proc/self/cgroup}") == 1
        self.script = tmp_path / "harness.sh"
        self.script.write_text(
            block.replace("${1:-/proc/self/cgroup}", "${1:-" + str(self.cgroup) + "}") + _REPORT,
            encoding="utf-8",
        )
        self.log = tmp_path / "systemd-run.log"
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        if with_stub:
            stub = self.bin / "systemd-run"
            stub.write_text(_SYSTEMD_RUN_STUB, encoding="utf-8")
            stub.chmod(0o755)
        for tool in ("bash", "sed", "head"):
            real = shutil.which(tool)
            assert real is not None
            os.symlink(real, self.bin / tool)

    def run(self, *args: str, env: dict[str, str] | None = None, stdin: str = "none"):
        """Run the harness with *args*; returns (CompletedProcess, pid it ran as).

        *stdin* is how the script reaches bash: ``"none"`` as a file
        argument, ``"file"`` / ``"pipe"`` as ``bash -s`` input (seekable
        or not), ``"tty"`` the ``bash -c "$(curl ...)"`` style, with a
        terminal on stdin.
        """
        full_env = {"PATH": str(self.bin), "STUB_LOG": str(self.log), **(env or {})}
        text = self.script.read_text(encoding="utf-8")
        master = -1
        if stdin == "none":
            argv, input_fd = ["bash", str(self.script), *args], subprocess.DEVNULL
        elif stdin == "file":
            argv, input_fd = ["bash", "-s", *args], os.open(self.script, os.O_RDONLY)
        elif stdin == "pipe":
            argv, input_fd = ["bash", "-s", *args], subprocess.PIPE
        else:
            import pty

            master, slave = pty.openpty()
            argv, input_fd = ["bash", "-c", text, "bash", *args], slave
        proc = subprocess.Popen(
            argv,
            stdin=input_fd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=full_env,
            cwd=self.dir,
        )
        out, err = proc.communicate(text if stdin == "pipe" else None, timeout=30)
        for fd in (master, input_fd):
            if isinstance(fd, int) and fd >= 0:
                os.close(fd)
        return subprocess.CompletedProcess(argv, proc.returncode, out, err), proc.pid

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("script", SCRIPTS, ids=["root", "bootstrap"])
def test_reexecs_in_a_scope_when_a_service_hosts_the_script(tmp_path: Path, script: Path) -> None:
    h = Harness(tmp_path, KISS_WEB_CGROUP, script)
    res, pid = h.run("a b", "--non-interactive")
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host=kiss-web.service pid={pid} args=[a b --non-interactive]\n"
    assert h.calls() == [PROBE, f"{REEXEC} {h.script} a b --non-interactive"]


@pytest.mark.parametrize("stdin", ["pipe", "file"])
def test_a_script_on_stdin_reexecs_as_bash_s_and_resumes_after_the_block(
    tmp_path: Path, stdin: str
) -> None:
    """``curl ... | bash`` has no ``$0`` file; the re-exec'd ``bash -s``
    reads the rest of the script from the inherited stdin."""
    h = Harness(tmp_path, KISS_WEB_CGROUP, SCRIPTS[1])
    res, pid = h.run("x y", "z", stdin=stdin)
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host=kiss-web.service pid={pid} args=[x y z]\n"
    assert h.calls() == [PROBE, f"{REEXEC} -s x y z"]


def test_an_inline_script_with_a_terminal_on_stdin_stays_put(tmp_path: Path) -> None:
    # bash -c "$(curl ...)": nothing to re-run, and stdin is not the script.
    h = Harness(tmp_path, KISS_WEB_CGROUP)
    res, pid = h.run("q", stdin="tty")
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host= pid={pid} args=[q]\n"
    assert h.calls() == []


def test_cgroup_v1_lines_and_scopes_do_not_count(tmp_path: Path) -> None:
    # A v1 controller line ends at user@UID.service; the systemd line of
    # an already escaped script ends in .scope: neither names a host.
    h = Harness(
        tmp_path,
        "12:pids:/user.slice/user-1006.slice/user@1006.service\n"
        "1:name=systemd:/user.slice/user-1006.slice/user@1006.service/app.slice/run-r1.scope\n",
    )
    res, pid = h.run("x")
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host= pid={pid} args=[x]\n"
    assert h.calls() == []


def test_no_cgroup_file_means_no_reexec(tmp_path: Path) -> None:
    h = Harness(tmp_path, None)  # macOS has no /proc/self/cgroup.
    res, pid = h.run()
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host= pid={pid} args=[]\n"
    assert h.calls() == []


def test_a_failing_scope_probe_leaves_the_script_where_it_is(tmp_path: Path) -> None:
    h = Harness(tmp_path, KISS_WEB_CGROUP)
    res, pid = h.run(env={"PROBE_RC": "1"})
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host= pid={pid} args=[]\n"
    assert h.calls() == [PROBE]


def test_without_systemd_run_there_is_nothing_to_do(tmp_path: Path) -> None:
    h = Harness(tmp_path, KISS_WEB_CGROUP, with_stub=False)
    res, pid = h.run()
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host= pid={pid} args=[]\n"
    assert h.calls() == []


def test_an_escaped_script_does_not_loop(tmp_path: Path) -> None:
    h = Harness(tmp_path, KISS_WEB_CGROUP)
    res, pid = h.run(env={"_KISS_HOST_SERVICE": "kiss-web.service"})
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"host=kiss-web.service pid={pid} args=[]\n"
    assert h.calls() == []


# --------------------------------------------------------------------------
# The real thing: a transient user service hosts the script, which stops it.
# --------------------------------------------------------------------------


def _user_scopes_work() -> bool:
    if not shutil.which("systemd-run"):
        return False
    probe = subprocess.run(
        ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--", "true"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return probe.returncode == 0


def _unit_active(unit: str) -> bool:
    res = subprocess.run(
        ["systemctl", "--user", "is-active", unit], capture_output=True, text=True, check=False
    )
    return res.stdout.strip() in ("active", "activating", "deactivating")


@pytest.mark.skipif(not _user_scopes_work(), reason="no systemd user manager")
@pytest.mark.parametrize(
    ("escape", "launch"),
    [(True, "bash {script} &"), (True, "cat {script} | bash -s &"), (False, "bash {script} &")],
    ids=["file", "piped", "no-escape"],
)
def test_survives_stopping_the_service_that_hosts_it(
    tmp_path: Path, escape: bool, launch: str
) -> None:
    """A script that a user service's child started (as kiss-web starts
    the Terminal tab's shell) and that stops that very service only
    lives to write its marker when the block moved it into a scope
    first.  ``no-escape`` presets ``_KISS_HOST_SERVICE`` to disable the
    block and shows the kill it prevents."""
    unit = f"kiss-test-host-{uuid.uuid4().hex[:8]}.service"
    marker = tmp_path / "marker"
    script = tmp_path / "harness.sh"
    script.write_text(
        _block(SCRIPTS[0])
        + textwrap.dedent(
            f"""\
            systemctl --user stop {unit}
            cat /proc/self/cgroup > {marker}
            """
        ),
        encoding="utf-8",
    )
    # The service's main process is a stand-in daemon; systemd signals a
    # main pid wherever it is, so the script must not be it.
    host = tmp_path / "host.sh"
    host.write_text(launch.format(script=script) + "\nsleep 300\n", encoding="utf-8")
    cmd = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}"]
    if not escape:
        cmd.append("--setenv=_KISS_HOST_SERVICE=preset")
    cmd += ["--", "bash", str(host)]
    subprocess.run(cmd, check=True, timeout=30)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not _unit_active(unit) and (marker.exists() or not escape):
                break
            time.sleep(0.1)
        assert not _unit_active(unit)
        time.sleep(0.5)  # a doomed script would have had ample time to write
        if escape:
            cgroup = marker.read_text(encoding="utf-8").strip()
            assert cgroup.endswith(".scope") and unit not in cgroup, cgroup
        else:
            assert not marker.exists()
    finally:
        for verb in ("stop", "reset-failed"):
            subprocess.run(["systemctl", "--user", verb, unit], capture_output=True, check=False)
