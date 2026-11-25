# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` asks nothing: at a terminal or without one, on Linux or macOS.

The script has no question left.  It once asked ``[Y/n]`` before it
installed Homebrew on a Mac; it no longer installs or uses Homebrew.
Run from a terminal it only differs in its ``Mode:`` line and in
staying in the terminal's session (no setsid re-exec);
``--non-interactive`` or ``KISS_NONINTERACTIVE=1`` selects the other
mode at a terminal, and the absence of a terminal selects it too (the
kiss-web daemon, cron, the detached re-exec).

The script also has NO update/upgrade logic for third-party tools: an
installed git or uv is used as-is, whatever its version.  Every test
runs the script against stub binaries reporting very old versions and
asserts that no question is asked, nothing is upgraded or downloaded,
and the run still reaches the Python environment step.

The tests run the real ``install.sh`` inside a hermetic sandbox (stub
``git``/``brew``/``sudo``/``uv``/... binaries, a two-file ``src/kiss``,
throwaway ``$HOME``) up to the Python environment step (a stub
``uv sync`` prints a marker and fails, which stops the run
deterministically).  Terminal runs use ``script(1)`` to attach a real
controlling pty; what is typed into it must be ignored.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

pytestmark = posix_only("bash install.sh under a pty")

REPO = Path(__file__).resolve().parents[4]
INSTALL_SCRIPT = REPO / "install.sh"

UV_SYNC_MARKER = "UV-SYNC-MARKER"
UV_SYNC_FAILED = "ERROR: uv sync failed"

# Deliberately ancient stub versions: the script must accept them as-is.
OLD_GIT_VERSION = "2.30.0"
OLD_UV_VERSION = "0.5.0"


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    """Create an executable bash stub named *name* in *bin_dir*."""
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def make_sandbox(root: Path, log_perl: bool = False, darwin: bool = False) -> dict:
    """Build a hermetic install.sh sandbox under *root*.

    Returns a dict with the script path, the environment to run it with
    and *root*.  All external tools that install.sh probes are stubbed
    so no install can touch the real system: ``brew`` and ``sudo`` log
    their arguments and fail, ``curl`` succeeds with empty output, and
    the stub ``uv sync`` stops the run with :data:`UV_SYNC_MARKER`.
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
            omit ``brew``: a Mac without Homebrew, on any host.
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

    # The smallest checkout install.sh copies: the package and the
    # webapp frontend it serves.
    media = kiss_ai / "src" / "kiss" / "webapp" / "media"
    media.mkdir(parents=True)
    (kiss_ai / "src" / "kiss" / "__init__.py").write_text("", encoding="utf-8")
    (media / "chat.html").write_text("<html></html>\n", encoding="utf-8")
    (kiss_ai / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

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
    _write_stub(
        stub_bin,
        "uv",
        f'[ "$1" = "sync" ] && {{ echo "{UV_SYNC_MARKER}"; exit 7; }}\n'
        f'echo "uv {OLD_UV_VERSION} (stub)"\n',
    )
    # install.sh asks it for the developer directory on a Mac
    # (macos_developer_tools_missing); this one has the tools.
    _write_stub(
        stub_bin, "xcode-select", f'[ "$1" = "-p" ] && {{ echo "{clt}"; exit 0; }}\nexit 1\n'
    )
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
    typed into that pty (``script`` forwards its stdin).
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
    """The run must have reached the stub ``uv sync`` and stopped there.

    Under a pty the exit status is checked only where ``script(1)`` forwards
    it (``-e``, util-linux); BSD ``script`` on macOS always exits 0, so
    there the marker plus the absence of the next install step (the
    byte-compile step that follows the venv build) is the proof.
    """
    assert UV_SYNC_MARKER in result.stdout, (
        f"install.sh did not reach the Python environment step:\n{result.stdout}"
    )
    assert UV_SYNC_FAILED in result.stdout, result.stdout
    assert "[3/5]" not in result.stdout, result.stdout
    if not (sys.platform == "darwin" and result.args[0] == "script"):
        assert result.returncode == 1, result.stdout


def _assert_nothing_upgraded(result: subprocess.CompletedProcess, root: Path) -> None:
    """Old tools must be accepted as-is: no question, no upgrade attempt.

    Besides the console wording, every stub that a resurrected upgrade
    or Homebrew path would have to run (``sudo``, ``apt-get``, ``brew``,
    ``curl``) logs its invocation; those logs must not exist.
    """
    out = result.stdout
    assert "Upgrade" not in out, out
    assert "Upgrading" not in out, out
    assert "older than the required version" not in out, out
    assert f"Using uv at {root / 'stubbin' / 'uv'}" in out, out
    for log_name in ("sudo.log", "aptget.log", "brew.log", "curl.log"):
        assert not (root / log_name).exists(), (
            f"{log_name} written — a system-mutating command or a download ran:\n"
            + (root / log_name).read_text(encoding="utf-8")
        )


# ---------------------------------------------------------------------------
# Third-party tools are never upgraded
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_no_terminal_accepts_old_tools_without_upgrading(tmp_path: Path) -> None:
    """Daemon update path: no tty, ancient git/uv stubs.

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
    """Interactive runs ask nothing: there is no upgrade question.

    With the answer channel open (a real pty) and every stub tool
    ancient, the script must not ask a single ``[Y/n]`` question.  The
    setsid re-exec must be skipped in interactive mode, which the
    failing ``perl`` stub would record.
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
    """``--non-interactive`` / ``KISS_NONINTERACTIVE=1`` select the unattended mode on a pty.

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
# A Mac without Homebrew
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("use_pty", [True, False], ids=["terminal", "no-terminal"])
def test_macos_asks_nothing_and_fetches_nothing(tmp_path: Path, use_pty: bool) -> None:
    """On a Mac without Homebrew the install neither asks for it nor fetches it.

    A ``uname`` stub makes install.sh take its Darwin path on any host
    and no ``brew`` is on the PATH.  The run must go straight to the
    build step: no question, no word about Homebrew, no download and no
    ``sudo`` (their stubs would have logged the call).
    """
    sandbox = make_sandbox(tmp_path, darwin=True)
    result = run_install(sandbox, use_pty=use_pty, input_text="n\n" if use_pty else None)
    out = result.stdout
    assert "OS: Darwin" in out, out
    assert ("Mode: interactive" if use_pty else "Mode: non-interactive") in out, out
    assert "[Y/n]" not in out, out
    # The test's name (in every path printed) holds neither spelling.
    assert "homebrew" not in out.lower(), out
    _assert_nothing_upgraded(result, tmp_path)
    _assert_reached_build(result)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
