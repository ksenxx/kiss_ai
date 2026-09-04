# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` upgrade questions: interactive by default, never fatal.

Run from a terminal, ``install.sh`` asks ``[Y/n]`` before installing
Homebrew and before upgrading an outdated git, uv, Node.js or VS Code
("no" keeps the installed version and continues).  ``--non-interactive``
or ``KISS_NONINTERACTIVE=1`` answers every question with its default
(Yes) without touching the terminal, and so does the absence of a
terminal altogether (the kiss-web daemon, cron, the detached re-exec).

History the tests guard against
===============================
An earlier prompt implementation crashed under ``set -eo pipefail``:

* an unguarded ``read ... </dev/tty`` returned non-zero on EOF and killed
  the script silently, right at the question;
* the non-interactive path hard-exited on a failing ``brew``, so the
  Update button never completed;
* a missing version constant in ``DependencyInstaller.ts`` made the
  ``REQUIRED_*`` extraction pipelines fail under ``pipefail``.

The prompts were then removed, and are now back with guards.  The tests
run the real ``install.sh`` inside a hermetic sandbox (stub
``git``/``brew``/``sudo``/``node``/``npm``/``code``/... binaries, throwaway
``$HOME``) with an outdated git and assert the script gets *past* the
question all the way to the extension build step (a stub ``npm run
package`` prints a marker and stops the run deterministically).
Terminal runs use ``script(1)`` to attach a real controlling pty and
feed the answers through it.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

NPM_MARKER = "NPM-PACKAGE-MARKER"
NPM_EXIT = 7

OLD_GIT_VERSION = "2.30.0"
REQUIRED_GIT_VERSION = "2.49.0"
REQUIRED_UV_VERSION = "0.11.2"

GIT_QUESTION = f"Upgrade git to {REQUIRED_GIT_VERSION} or later? [Y/n]"
GIT_OUTDATED = f"git {OLD_GIT_VERSION} is older than the required version {REQUIRED_GIT_VERSION}."
GIT_UPGRADING = "Upgrading git..."
GIT_SKIPPED = "Skipping the git upgrade"

CTRL_C = "\x03"
CTRL_D = "\x04"


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    """Create an executable bash stub named *name* in *bin_dir*."""
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def make_sandbox(
    root: Path,
    with_dep_installer_ts: bool = True,
    log_perl: bool = False,
    darwin: bool = False,
    outdated_everything: bool = False,
) -> dict:
    """Build a hermetic install.sh sandbox under *root*.

    Returns a dict with the script path, the environment to run it with
    and *root*.  All external tools that install.sh probes are stubbed
    so no upgrade can touch the real system: ``brew`` and ``sudo`` log
    their arguments and fail, ``curl`` succeeds with empty output, and
    the stub ``npm run package`` stops the run with :data:`NPM_MARKER`.

    Args:
        root: Empty directory to build the sandbox in.
        with_dep_installer_ts: Write the ``DependencyInstaller.ts`` the
            script extracts ``REQUIRED_GIT_VERSION``/``REQUIRED_UV_VERSION``
            from; ``False`` reproduces the missing-constants regression.
        log_perl: Add a ``perl`` stub that records its invocation in
            ``root/perl.log`` and fails, so a test can tell whether the
            new-session re-exec block was even attempted.
        darwin: Add a ``uname`` stub reporting ``Darwin``/``x86_64`` and
            omit ``brew`` so the macOS Homebrew question is reached.
        outdated_everything: Report outdated uv, Node.js and VS Code as
            well as git so every upgrade question is asked.
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
    if with_dep_installer_ts:
        (src_dir / "src" / "DependencyInstaller.ts").write_text(
            f"const UV_VERSION = '{REQUIRED_UV_VERSION}';\n"
            f"const GIT_VERSION = '{REQUIRED_GIT_VERSION}';\n",
            encoding="utf-8",
        )
    (src_dir / "package.json").write_text(
        '{"engines": {"vscode": "^1.98.0"}}\n', encoding="utf-8",
    )
    scripts_dir = kiss_ai / "scripts"
    scripts_dir.mkdir()
    _write_stub(scripts_dir, "fetch-claude-skills.sh", "echo skills-ok\n")

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
    _write_stub(stub_bin, "apt-get", "exit 0\n")
    _write_stub(stub_bin, "curl", "exit 0\n")
    uv_version = "0.5.0" if outdated_everything else REQUIRED_UV_VERSION
    node_version = "v18.0.0" if outdated_everything else "v22.16.0"
    code_version = "1.90.0" if outdated_everything else "1.98.2"
    _write_stub(stub_bin, "uv", f'echo "uv {uv_version} (stub)"\n')
    _write_stub(stub_bin, "node", f'echo "{node_version}"\n')
    _write_stub(stub_bin, "npx", "exit 0\n")
    _write_stub(
        stub_bin,
        "npm",
        '[ "$1" = "ci" ] && exit 0\n'
        f'[ "$1" = "run" ] && {{ echo "{NPM_MARKER}"; exit {NPM_EXIT}; }}\n'
        "exit 0\n",
    )
    _write_stub(stub_bin, "code", f'printf "{code_version}\\nabcdef\\nstub\\n"\n')
    _write_stub(
        stub_bin,
        "xcode-select",
        f'[ "$1" = "-p" ] && {{ echo "{clt}"; exit 0; }}\nexit 1\n',
    )
    # ``upgrade_vscode`` on Darwin quits the running VS Code via osascript;
    # never let the sandbox reach the real one on a macOS host.
    _write_stub(stub_bin, "osascript", "exit 0\n")
    if log_perl:
        _write_stub(stub_bin, "perl", f'echo "perl $*" >> "{root}/perl.log"\nexit 1\n')
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
    there the marker plus the absence of the next build step is the proof.
    """
    assert NPM_MARKER in result.stdout, (
        f"install.sh did not reach the build step:\n{result.stdout}"
    )
    assert "Copying bundled KISS runtime" not in result.stdout, result.stdout
    if not (sys.platform == "darwin" and result.args[0] == "script"):
        assert result.returncode == NPM_EXIT, result.stdout


# ---------------------------------------------------------------------------
# Non-interactive paths
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_no_terminal_upgrades_without_asking(tmp_path: Path) -> None:
    """Daemon update path: no tty, old git, failing package manager.

    Without a terminal there is nobody to ask, so the run is
    non-interactive: git is upgraded unconditionally, the failing
    upgrade only warns, and the script continues to the build step.
    """
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=False)
    assert "Mode: non-interactive" in result.stdout, result.stdout
    assert GIT_OUTDATED in result.stdout, result.stdout
    assert GIT_UPGRADING in result.stdout, result.stdout
    assert "[Y/n]" not in result.stdout, result.stdout
    assert f"WARNING: git is still {OLD_GIT_VERSION}" in result.stdout, result.stdout
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
    "n" typed into the terminal must be ignored (git is upgraded anyway)
    and the new-session re-exec must still be attempted, which the
    ``perl`` stub records.
    """
    sandbox = make_sandbox(tmp_path, log_perl=True)
    result = run_install(sandbox, use_pty=True, args=args, input_text="n\n", extra_env=extra_env)
    assert "Mode: non-interactive" in result.stdout, result.stdout
    assert "[Y/n]" not in result.stdout, result.stdout
    assert GIT_UPGRADING in result.stdout, result.stdout
    assert (tmp_path / "perl.log").exists(), (
        "non-interactive runs must keep the setsid re-exec:\n" + result.stdout
    )
    _assert_reached_build(result)


@pytest.mark.slow
def test_missing_version_constants_do_not_crash(tmp_path: Path) -> None:
    """A missing DependencyInstaller.ts must not kill the script.

    Under ``pipefail`` the ``REQUIRED_*`` extraction pipelines exited
    non-zero when the constants file was absent, killing install.sh at
    the very top before any output.  Empty versions must simply skip
    the version checks.
    """
    sandbox = make_sandbox(tmp_path, with_dep_installer_ts=False)
    result = run_install(sandbox, use_pty=False)
    assert "Checking git" in result.stdout, result.stdout
    assert "older than the required version" not in result.stdout, result.stdout
    _assert_reached_build(result)


# ---------------------------------------------------------------------------
# Interactive paths (the default at a terminal)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_terminal_asks_and_no_keeps_installed_git(tmp_path: Path) -> None:
    """At a terminal the script asks; "n" skips the upgrade and continues.

    The answer is read from ``/dev/tty``, so the setsid re-exec must be
    skipped (otherwise there would be no controlling terminal); the
    ``perl`` stub proves the block was not even attempted.
    """
    sandbox = make_sandbox(tmp_path, log_perl=True)
    result = run_install(sandbox, use_pty=True, input_text="n\n")
    assert "Mode: interactive" in result.stdout, result.stdout
    assert GIT_OUTDATED in result.stdout, result.stdout
    assert GIT_QUESTION in result.stdout, result.stdout
    assert GIT_SKIPPED in result.stdout, result.stdout
    assert GIT_UPGRADING not in result.stdout, result.stdout
    assert not (tmp_path / "sudo.log").exists(), "declined upgrade must not run sudo"
    assert not (tmp_path / "brew.log").exists(), "declined upgrade must not run brew"
    assert not (tmp_path / "perl.log").exists(), (
        "interactive runs must skip the setsid re-exec:\n" + result.stdout
    )
    _assert_reached_build(result)
    # Question and echoed answer travel the same tee pipe, so the log has
    # the complete line.
    log = (tmp_path / "home" / ".kiss" / "install.log").read_text(encoding="utf-8")
    assert f"{GIT_QUESTION} n\n" in log, log
    assert GIT_SKIPPED in log, log


@pytest.mark.slow
@pytest.mark.parametrize("answer", ["\n", "y\n", "YES\n"], ids=["enter", "y", "YES"])
def test_terminal_yes_runs_the_upgrade(tmp_path: Path, answer: str) -> None:
    """Enter (the default), "y" and "yes" all run the upgrade.

    The stubbed package manager fails, so the run must warn and still
    reach the build step, exactly like the non-interactive path.
    """
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True, input_text=answer)
    assert f"{GIT_QUESTION} {answer.strip() or 'yes'}" in result.stdout, result.stdout
    assert GIT_UPGRADING in result.stdout, result.stdout
    assert GIT_SKIPPED not in result.stdout, result.stdout
    assert f"WARNING: git is still {OLD_GIT_VERSION}" in result.stdout, result.stdout
    _assert_reached_build(result)


@pytest.mark.slow
def test_terminal_invalid_answer_is_asked_again(tmp_path: Path) -> None:
    """Anything but y/yes/n/no/Enter re-asks the question."""
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True, input_text="maybe\nNO\n")
    assert "Please answer y or n." in result.stdout, result.stdout
    assert result.stdout.count(GIT_QUESTION) == 2, result.stdout
    assert GIT_SKIPPED in result.stdout, result.stdout
    _assert_reached_build(result)


@pytest.mark.slow
def test_terminal_eof_takes_the_default(tmp_path: Path) -> None:
    """Ctrl-D at the question is EOF on /dev/tty: default to yes, do not die.

    This is the exact ``read`` failure that once killed the script under
    ``set -e``; a ``/dev/tty`` that can no longer be opened fails the same
    ``read`` the same way.
    """
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True, input_text=CTRL_D)
    assert GIT_QUESTION in result.stdout, result.stdout
    assert "(no answer from the terminal; assuming yes)" in result.stdout, result.stdout
    assert GIT_UPGRADING in result.stdout, result.stdout
    _assert_reached_build(result)


@pytest.mark.slow
@pytest.mark.parametrize("answers", ["n\nn\nn\nn\n", "\n\n\n\n"], ids=["all-no", "all-yes"])
def test_terminal_asks_for_every_outdated_tool(tmp_path: Path, answers: str) -> None:
    """uv, Node.js and VS Code get their own questions, honoured independently."""
    sandbox = make_sandbox(tmp_path, outdated_everything=True)
    result = run_install(sandbox, use_pty=True, input_text=answers)
    out = result.stdout
    assert GIT_QUESTION in out, out
    assert f"Upgrade uv to {REQUIRED_UV_VERSION}? [Y/n]" in out, out
    assert "Upgrade Node.js to 22.16.0? [Y/n]" in out, out
    assert "Upgrade VS Code? [Y/n]" in out, out
    if answers.startswith("n"):
        assert GIT_SKIPPED in out, out
        assert "Skipping the uv upgrade; continuing with uv 0.5.0." in out, out
        assert "Skipping the Node.js upgrade" in out, out
        assert "Skipping the VS Code upgrade" in out, out
        assert "Upgrading" not in out, out
    else:
        assert GIT_UPGRADING in out, out
        assert f"Upgrading uv to {REQUIRED_UV_VERSION}..." in out, out
        assert "Upgrading Node.js to v22.16.0..." in out, out
        assert "Upgrading VS Code..." in out, out
        assert "WARNING: VS Code is still 1.90.0 (< 1.98.0)" in out, out
        assert "Skipping" not in out, out
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
    sandbox = make_sandbox(tmp_path)
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
        head = _read_until(proc, GIT_QUESTION, deadline_s=60)
        time.sleep(0.5)
        proc.stdin.write(CTRL_C.encode())
        proc.stdin.flush()
        # One read: the trap's notice and the re-printed question arrive
        # together, and a second call would wait for output that has
        # already been consumed.
        after_interrupt = _read_until(proc, GIT_QUESTION, deadline_s=20)
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
    assert out.count(GIT_QUESTION) == 2, out
    assert GIT_SKIPPED in out, out
    assert GIT_UPGRADING not in out, out
    assert NPM_MARKER in out, out
    # ``script -c`` runs install.sh under a ``sh -c`` wrapper that shares
    # the pty's foreground process group.  That untrapped, non-interactive
    # wrapper also receives the Ctrl-C and, in the way of non-interactive
    # shells, kills itself with SIGINT once its child exits, so ``script``
    # reports 130 whatever install.sh returned.  Prove install.sh stopped
    # at the stub's exit 7 under ``set -e`` from its output instead: the
    # step after the marker never ran.
    assert "Copying bundled KISS runtime" not in out, out


@pytest.mark.slow
@pytest.mark.parametrize("brew_answer", ["n", "y"])
def test_macos_asks_before_installing_homebrew(tmp_path: Path, brew_answer: str) -> None:
    """On macOS without Homebrew the install is a question too.

    A ``uname`` stub makes install.sh take its Darwin path on any host;
    the stubbed ``curl`` returns an empty installer so "y" runs a no-op
    Homebrew bootstrap that ends in the existing "did not complete"
    warning.  Either way the git question follows and "n" to it must
    still reach the build step.
    """
    if any(Path(p).exists() for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")):
        pytest.skip("a real Homebrew is installed; install.sh would find it before asking")
    sandbox = make_sandbox(tmp_path, darwin=True)
    result = run_install(sandbox, use_pty=True, input_text=f"{brew_answer}\nn\n")
    out = result.stdout
    assert "OS: Darwin" in out, out
    assert "Install Homebrew now? [Y/n]" in out, out
    if brew_answer == "n":
        assert "Skipping the Homebrew install" in out, out
        assert "Installing Homebrew..." not in out, out
    else:
        assert "Installing Homebrew..." in out, out
        assert "WARNING: Homebrew install did not complete" in out, out
    assert GIT_QUESTION in out, out
    assert GIT_SKIPPED in out, out
    _assert_reached_build(result)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
