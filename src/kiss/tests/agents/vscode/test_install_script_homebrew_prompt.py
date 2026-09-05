# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` questions: only the Homebrew install asks, and never fatally.

Run from a terminal, ``install.sh`` asks ``[Y/n]`` before installing
Homebrew ("no" skips it and continues).  ``--non-interactive`` or
``KISS_NONINTERACTIVE=1`` answers the question with its default (Yes)
without touching the terminal, and so does the absence of a terminal
altogether (the kiss-web daemon, cron, the detached re-exec).

The script deliberately has NO update/upgrade logic for third-party
tools: an installed git, uv, Node.js or VS Code is used as-is, whatever
its version.  Several tests below run the script against stub binaries
reporting very old versions and assert that no upgrade question is
asked, nothing is upgraded, and the run still reaches the build step.

History the tests guard against
===============================
An earlier prompt implementation crashed under ``set -eo pipefail``:

* an unguarded ``read ... </dev/tty`` returned non-zero on EOF and killed
  the script silently, right at the question;
* the non-interactive path hard-exited on a failing ``brew``, so the
  Update button never completed.

The tests run the real ``install.sh`` inside a hermetic sandbox (stub
``git``/``brew``/``sudo``/``node``/``npm``/``code``/... binaries, throwaway
``$HOME``) and assert the script gets *past* the question all the way to
the extension build step (a stub ``npm run package`` prints a marker and
stops the run deterministically).  Terminal runs use ``script(1)`` to
attach a real controlling pty and feed the answers through it.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

NPM_MARKER = "NPM-PACKAGE-MARKER"
NPM_EXIT = 7

# Deliberately ancient stub versions: the script must accept them as-is.
OLD_GIT_VERSION = "2.30.0"
OLD_UV_VERSION = "0.5.0"
OLD_NODE_VERSION = "v18.0.0"
OLD_CODE_VERSION = "1.90.0"

BREW_QUESTION = "Install Homebrew now? [Y/n]"
BREW_SKIPPED = "Skipping the Homebrew install"
BREW_INSTALLING = "Installing Homebrew..."

CTRL_C = "\x03"
CTRL_D = "\x04"


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    """Create an executable bash stub named *name* in *bin_dir*."""
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def make_sandbox(
    root: Path,
    log_perl: bool = False,
    darwin: bool = False,
) -> dict:
    """Build a hermetic install.sh sandbox under *root*.

    Returns a dict with the script path, the environment to run it with
    and *root*.  All external tools that install.sh probes are stubbed
    so no install can touch the real system: ``brew`` and ``sudo`` log
    their arguments and fail, ``curl`` succeeds with empty output, and
    the stub ``npm run package`` stops the run with :data:`NPM_MARKER`.
    Every stub reports an ancient version so the tests prove the script
    accepts installed tools as-is instead of upgrading them.

    Args:
        root: Empty directory to build the sandbox in.
        log_perl: Add a ``perl`` stub that records the new-session
            re-exec block's ``POSIX::setsid`` probe in ``root/perl.log``
            and fails it (so the block is attempted but never re-execs),
            while delegating every other invocation — the update lock's
            ``flock`` call — to the real perl.
        darwin: Add a ``uname`` stub reporting ``Darwin``/``x86_64`` and
            omit ``brew`` so the macOS Homebrew question is reached.
    """
    kiss_ai = root / "kiss_ai"
    home = root / "home"
    stub_bin = root / "stubbin"
    clt = root / "clt"
    for d in (kiss_ai, home, stub_bin, clt / "usr" / "bin"):
        d.mkdir(parents=True)
    (clt / "usr" / "bin" / "git").touch()

    script = kiss_ai / "install.sh"
    script.write_bytes(INSTALL_SCRIPT.read_bytes())
    script.chmod(0o755)

    src_dir = kiss_ai / "src" / "kiss" / "agents" / "vscode"
    (src_dir / "src").mkdir(parents=True)
    # Version-source fixtures with impossibly high thresholds.  install.sh
    # no longer reads these files, but a REINTRODUCED version-gated upgrade
    # path (the removed logic extracted required versions from exactly
    # these two files) would see every stub tool as hopelessly outdated
    # and trip the no-upgrade assertions instead of passing vacuously on
    # empty extracted versions.
    (src_dir / "src" / "DependencyInstaller.ts").write_text(
        "const UV_VERSION = '99.0.0';\nconst GIT_VERSION = '99.0.0';\n",
        encoding="utf-8",
    )
    (src_dir / "package.json").write_text(
        '{"engines": {"vscode": "^99.0.0"}}\n', encoding="utf-8",
    )

    _write_stub(
        stub_bin,
        "git",
        'for a in "$@"; do\n'
        f'  [ "$a" = "--version" ] && {{ echo "git version {OLD_GIT_VERSION}"; exit 0; }}\n'
        '  [ "$a" = "rev-parse" ] && exit 1\n'
        "done\nexit 0\n",
    )
    if not darwin:
        _write_stub(stub_bin, "brew", f'echo "brew $*" >> "{root}/brew.log"\nexit 1\n')
    _write_stub(stub_bin, "sudo", f'echo "sudo $*" >> "{root}/sudo.log"\nexit 1\n')
    _write_stub(stub_bin, "apt-get", f'echo "apt-get $*" >> "{root}/aptget.log"\nexit 0\n')
    _write_stub(stub_bin, "curl", f'echo "curl $*" >> "{root}/curl.log"\nexit 0\n')
    _write_stub(stub_bin, "uv", f'echo "uv {OLD_UV_VERSION} (stub)"\n')
    _write_stub(stub_bin, "node", f'echo "{OLD_NODE_VERSION}"\n')
    _write_stub(stub_bin, "npx", "exit 0\n")
    _write_stub(
        stub_bin,
        "npm",
        '[ "$1" = "ci" ] && exit 0\n'
        'if [ "$1" = "run" ]; then\n'
        f'  [ "$2" = "package" ] && {{ echo "{NPM_MARKER}"; exit {NPM_EXIT}; }}\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
    )
    _write_stub(stub_bin, "code", f'printf "{OLD_CODE_VERSION}\\nabcdef\\nstub\\n"\n')
    _write_stub(
        stub_bin,
        "xcode-select",
        f'[ "$1" = "-p" ] && {{ echo "{clt}"; exit 0; }}\nexit 1\n',
    )
    # Belt and braces: nothing in the sandbox should reach osascript, but
    # never let a stray call touch the real VS Code on a macOS host.
    _write_stub(stub_bin, "osascript", "exit 0\n")
    if log_perl:
        # Fail only the re-exec block's ``use POSIX qw(setsid)`` probe
        # (recording it), so the run stays attached to the pty; every
        # other perl call — the update lock's flock — needs the real
        # perl, resolved on the host's PATH before the stub shadows it.
        real_perl = shutil.which("perl") or "/usr/bin/perl"
        _write_stub(
            stub_bin,
            "perl",
            'case "$*" in\n'
            f'  *POSIX*setsid*) echo "perl $*" >> "{root}/perl.log"; exit 1 ;;\n'
            f'  *) exec "{real_perl}" "$@" ;;\n'
            "esac\n",
        )
    if darwin:
        _write_stub(
            stub_bin,
            "uname",
            '[ "$1" = "-s" ] && { echo Darwin; exit 0; }\n'
            '[ "$1" = "-m" ] && { echo x86_64; exit 0; }\n'
            "echo Darwin\n",
        )

    env = {
        "HOME": str(home),
        "PATH": f"{stub_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
    }
    return {"script": script, "env": env, "root": root}


def _pty_command(sandbox: dict, args: tuple[str, ...]) -> list[str]:
    """Return the ``script(1)`` command line that runs install.sh on a pty."""
    inner = " ".join([f"bash '{sandbox['script']}'", *args])
    if sys.platform == "darwin":
        return ["script", "-q", "/dev/null", "bash", "-c", inner]
    return ["script", "-qec", inner, "/dev/null"]


def run_install(
    sandbox: dict,
    use_pty: bool,
    args: tuple[str, ...] = (),
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run the sandboxed install.sh, optionally under a pty.

    ``use_pty=False`` mirrors the kiss-web daemon's update endpoint
    (detached, no controlling terminal — ``start_new_session`` guarantees
    that even when pytest itself runs from a terminal).  ``use_pty=True``
    uses ``script(1)`` to attach a real controlling pty; *input_text* is
    typed into that pty (``script`` forwards its stdin), so ``"n\\n"`` is
    what a user answering "n" produces and :data:`CTRL_D` is Ctrl-D.
    """
    env = dict(sandbox["env"])
    if extra_env:
        env.update(extra_env)
    if use_pty:
        cmd = _pty_command(sandbox, args)
    else:
        cmd = ["bash", str(sandbox["script"]), *args]
    return subprocess.run(
        cmd,
        cwd=str(sandbox["script"].parent),
        env=env,
        input=input_text,
        stdin=subprocess.DEVNULL if input_text is None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        timeout=120,
        text=True,
        errors="replace",
        check=False,
    )


def _assert_reached_build(result: subprocess.CompletedProcess) -> None:
    """The run must have reached the stub ``npm run package`` and stopped there.

    Under a pty the exit status is checked only where ``script(1)`` forwards
    it (``-e``, util-linux); BSD ``script`` on macOS always exits 0, so
    there the marker plus the absence of the next install step (the
    launcher install that follows the packaging step) is the proof.
    """
    assert NPM_MARKER in result.stdout, (
        f"install.sh did not reach the packaging step:\n{result.stdout}"
    )
    assert "Installing rsorcar" not in result.stdout, result.stdout
    if not (sys.platform == "darwin" and result.args[0] == "script"):
        assert result.returncode == NPM_EXIT, result.stdout


def _assert_nothing_upgraded(
    result: subprocess.CompletedProcess, root: Path, allow_curl: bool = False
) -> None:
    """Old tools must be accepted as-is: no question, no upgrade attempt.

    Besides the console wording, every stub that a resurrected upgrade
    path would have to run (``sudo``, ``apt-get``, ``brew``, ``curl``)
    logs its invocation; those logs must not exist.  *allow_curl* is for
    the macOS test that answers "yes" to the Homebrew question, whose
    legitimate bootstrap runs ``curl``.
    """
    out = result.stdout
    assert "Upgrade" not in out, out
    assert "Upgrading" not in out, out
    assert "older than the required version" not in out, out
    assert f"git {OLD_GIT_VERSION} ready" in out, out
    assert f"uv {OLD_UV_VERSION} ready" in out, out
    assert f"node {OLD_NODE_VERSION} ready" in out, out
    assert f"(v{OLD_CODE_VERSION})" in out, out
    for log_name in ("sudo.log", "aptget.log", "brew.log"):
        assert not (root / log_name).exists(), (
            f"{log_name} written — a system-mutating command ran:\n"
            + (root / log_name).read_text(encoding="utf-8")
        )
    if not allow_curl:
        assert not (root / "curl.log").exists(), (
            "curl ran — a download was attempted:\n"
            + (root / "curl.log").read_text(encoding="utf-8")
        )


# ---------------------------------------------------------------------------
# Third-party tools are never upgraded
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_no_terminal_accepts_old_tools_without_upgrading(tmp_path: Path) -> None:
    """Daemon update path: no tty, ancient git/uv/node/code stubs.

    The script must accept every installed tool as-is — no version
    check, no upgrade, no question — and continue to the build step.
    """
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=False)
    assert "Mode: non-interactive" in result.stdout, result.stdout
    assert "[Y/n]" not in result.stdout, result.stdout
    _assert_nothing_upgraded(result, tmp_path)
    _assert_reached_build(result)


@pytest.mark.slow
def test_terminal_accepts_old_tools_without_asking(tmp_path: Path) -> None:
    """Interactive runs ask nothing on Linux: there is no upgrade question.

    With the answer channel open (a real pty) and every stub tool
    ancient, the script must not ask a single ``[Y/n]`` question — the
    Homebrew question is macOS-only and upgrade questions no longer
    exist.  The setsid re-exec must be skipped in interactive mode,
    which the failing ``perl`` stub would record.
    """
    sandbox = make_sandbox(tmp_path, log_perl=True)
    result = run_install(sandbox, use_pty=True, input_text="n\n")
    assert "Mode: interactive" in result.stdout, result.stdout
    assert "[Y/n]" not in result.stdout, result.stdout
    _assert_nothing_upgraded(result, tmp_path)
    assert not (tmp_path / "perl.log").exists(), (
        "interactive runs must skip the setsid re-exec:\n" + result.stdout
    )
    _assert_reached_build(result)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("args", "extra_env"),
    [
        (("--non-interactive",), None),
        ((), {"KISS_NONINTERACTIVE": "1"}),
    ],
    ids=["flag", "env"],
)
def test_non_interactive_opt_out_at_a_terminal(
    tmp_path: Path, args: tuple[str, ...], extra_env: dict[str, str] | None
) -> None:
    """``--non-interactive`` / ``KISS_NONINTERACTIVE=1`` silence the questions on a pty.

    This is what the Update button and the Docker entrypoint pass.  The
    "n" typed into the terminal must be ignored and the new-session
    re-exec must still be attempted, which the ``perl`` stub records.
    """
    sandbox = make_sandbox(tmp_path, log_perl=True)
    result = run_install(sandbox, use_pty=True, args=args, input_text="n\n", extra_env=extra_env)
    assert "Mode: non-interactive" in result.stdout, result.stdout
    assert "[Y/n]" not in result.stdout, result.stdout
    _assert_nothing_upgraded(result, tmp_path)
    assert (tmp_path / "perl.log").exists(), (
        "non-interactive runs must keep the setsid re-exec:\n" + result.stdout
    )
    _assert_reached_build(result)


# ---------------------------------------------------------------------------
# The Homebrew question (macOS, the one remaining interactive question)
# ---------------------------------------------------------------------------


def _darwin_sandbox(tmp_path: Path) -> dict:
    """A Darwin sandbox, skipping when a real Homebrew would preempt the question."""
    if any(Path(p).exists() for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")):
        pytest.skip("a real Homebrew is installed; install.sh would find it before asking")
    return make_sandbox(tmp_path, darwin=True)


@pytest.mark.slow
@pytest.mark.parametrize("brew_answer", ["n", "y"])
def test_macos_asks_before_installing_homebrew(tmp_path: Path, brew_answer: str) -> None:
    """On macOS without Homebrew the install is a question.

    A ``uname`` stub makes install.sh take its Darwin path on any host;
    the stubbed ``curl`` returns an empty installer so "y" runs a no-op
    Homebrew bootstrap that ends in the existing "did not complete"
    warning.  Either way the run must continue to the build step, with
    no further questions.
    """
    sandbox = _darwin_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True, input_text=f"{brew_answer}\n")
    out = result.stdout
    assert "OS: Darwin" in out, out
    assert BREW_QUESTION in out, out
    if brew_answer == "n":
        assert BREW_SKIPPED in out, out
        assert BREW_INSTALLING not in out, out
    else:
        assert BREW_INSTALLING in out, out
        assert "WARNING: Homebrew install did not complete" in out, out
    _assert_nothing_upgraded(result, tmp_path, allow_curl=brew_answer == "y")
    _assert_reached_build(result)
    # Question and echoed answer travel the same tee pipe, so the log has
    # the complete line.
    log = (tmp_path / "home" / ".kiss" / "install.log").read_text(encoding="utf-8")
    assert f"{BREW_QUESTION} {brew_answer}\n" in log, log


@pytest.mark.slow
def test_terminal_invalid_answer_is_asked_again(tmp_path: Path) -> None:
    """Anything but y/yes/n/no/Enter re-asks the question."""
    sandbox = _darwin_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True, input_text="maybe\nNO\n")
    assert "Please answer y or n." in result.stdout, result.stdout
    assert result.stdout.count(BREW_QUESTION) == 2, result.stdout
    assert BREW_SKIPPED in result.stdout, result.stdout
    _assert_reached_build(result)


@pytest.mark.slow
def test_terminal_eof_takes_the_default(tmp_path: Path) -> None:
    """Ctrl-D at the question is EOF on /dev/tty: default to yes, do not die.

    This is the exact ``read`` failure that once killed the script under
    ``set -e``; a ``/dev/tty`` that can no longer be opened fails the same
    ``read`` the same way.
    """
    sandbox = _darwin_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True, input_text=CTRL_D)
    assert BREW_QUESTION in result.stdout, result.stdout
    assert "(no answer from the terminal; assuming yes)" in result.stdout, result.stdout
    assert BREW_INSTALLING in result.stdout, result.stdout
    _assert_reached_build(result)


def _read_until(proc: subprocess.Popen[bytes], needle: str, deadline_s: float) -> str:
    """Read *proc*'s stdout until *needle* appears or the deadline passes."""
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    buf = b""
    deadline = time.monotonic() + deadline_s
    while needle.encode() not in buf:
        remaining = deadline - time.monotonic()
        so_far = buf.decode(errors="replace")
        if remaining <= 0:
            raise AssertionError(f"timed out waiting for {needle!r}; output so far:\n{so_far}")
        ready, _, _ = select.select([fd], [], [], remaining)
        if ready:
            chunk = os.read(fd, 65536)
            if not chunk:
                raise AssertionError(f"stdout closed before {needle!r}; output so far:\n{so_far}")
            buf += chunk
    return buf.decode(errors="replace")


@pytest.mark.slow
def test_terminal_ctrl_c_at_the_question_asks_again(tmp_path: Path) -> None:
    """A single Ctrl-C while the question is pending re-asks it.

    ``handle_interrupt`` deliberately ignores one Ctrl-C (it prints a
    notice and aborts only on a second one within 3 s).  bash 5 then
    leaves the ``read`` waiting, so the trap re-prints the pending
    question (``CONFIRM_PENDING``); a bash whose ``read`` gives up with
    status 130 makes ``confirm`` read again without printing.  Either
    way the question must appear exactly twice, the Ctrl-C must not be
    taken as an answer or EOF, and the "n" typed afterwards must be
    honoured.  The Ctrl-C is typed only after the question is on screen
    (bash blocked in ``read``) so it cannot land before the trap is
    installed.
    """
    sandbox = _darwin_sandbox(tmp_path)
    proc = subprocess.Popen(
        _pty_command(sandbox, ()),
        cwd=str(sandbox["script"].parent),
        env=sandbox["env"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert proc.stdin is not None
    try:
        head = _read_until(proc, BREW_QUESTION, deadline_s=60)
        time.sleep(0.5)
        proc.stdin.write(CTRL_C.encode())
        proc.stdin.flush()
        # One read: the trap's notice and the re-printed question arrive
        # together, and a second call would wait for output that has
        # already been consumed.
        after_interrupt = _read_until(proc, BREW_QUESTION, deadline_s=20)
        assert "Interrupt received but ignored" in after_interrupt, after_interrupt
        head += after_interrupt
        proc.stdin.write(b"n\n")
        proc.stdin.flush()
        # communicate() closes stdin itself; closing it here first makes
        # Python <= 3.12 raise "flush of closed file".
        rest, _ = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    out = head + rest.decode(errors="replace")
    assert out.count(BREW_QUESTION) == 2, out
    assert BREW_SKIPPED in out, out
    assert BREW_INSTALLING not in out, out
    assert NPM_MARKER in out, out
    # ``script -c`` runs install.sh under a ``sh -c`` wrapper that shares
    # the pty's foreground process group.  That untrapped, non-interactive
    # wrapper also receives the Ctrl-C and, in the way of non-interactive
    # shells, kills itself with SIGINT once its child exits, so ``script``
    # reports 130 whatever install.sh returned.  Prove install.sh stopped
    # at the stub's exit 7 under ``set -e`` from its output instead: the
    # step after the marker never ran.
    assert "Installing rsorcar" not in out, out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
