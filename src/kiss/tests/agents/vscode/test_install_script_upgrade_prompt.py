# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: ``install.sh`` must survive the tool-upgrade questions.

Background — the bug
====================
The Update button (VS Code extension and the remote webapp) runs
``install.sh``.  When the installed git was older than the repo's
required version the script asked "Upgrade git ...? [Y/n]" and crashed
under ``set -eo pipefail``:

* with a pty but no usable input (terminal closed / EOF), the unguarded
  ``read ... </dev/tty`` returned non-zero and ``set -e`` killed the
  script silently, right at the question;
* in the non-interactive default-Yes path, ``upgrade_git`` hard-exited
  with "Cannot upgrade git without Homebrew" (or died on a failing
  ``brew``), so the update never completed;
* independently, a missing/renamed version constant in
  ``DependencyInstaller.ts`` made the ``REQUIRED_*`` extraction
  pipelines fail under ``pipefail`` and killed the script before it
  printed anything.

These tests run the real ``install.sh`` inside a hermetic sandbox (stub
``git``/``brew``/``node``/``npm``/``code``/... binaries, throwaway
``$HOME``) and assert the script gets *past* the git-upgrade question
all the way to the extension build step (a stub ``npm run package``
prints a marker and stops the run deterministically).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

# Stub ``npm run package`` prints this and exits 7, deterministically
# ending the run *after* all the version checks under test.
NPM_MARKER = "NPM-PACKAGE-MARKER"
NPM_EXIT = 7

OLD_GIT_VERSION = "2.30.0"


def _write_stub(bin_dir: Path, name: str, body: str) -> None:
    """Create an executable bash stub named *name* in *bin_dir*."""
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def make_sandbox(root: Path, with_dep_installer_ts: bool = True) -> dict:
    """Build a hermetic install.sh sandbox under *root*.

    Returns a dict with the script path and the environment to run it
    with.  All external tools that install.sh probes are stubbed; the
    stub ``brew`` always fails so the git-upgrade attempt cannot touch
    the real system, and the stub ``npm run package`` stops the run
    with :data:`NPM_MARKER`.
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
            "const UV_VERSION = '0.11.2';\nconst GIT_VERSION = '2.49.0';\n",
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
    _write_stub(stub_bin, "brew", f'echo "brew $*" >> "{root}/brew.log"\nexit 1\n')
    _write_stub(stub_bin, "sudo", "exit 1\n")
    _write_stub(stub_bin, "curl", "exit 0\n")
    _write_stub(stub_bin, "uv", 'echo "uv 0.11.2 (stub)"\n')
    _write_stub(stub_bin, "node", 'echo "v22.16.0"\n')
    _write_stub(stub_bin, "npx", "exit 0\n")
    _write_stub(
        stub_bin,
        "npm",
        '[ "$1" = "ci" ] && exit 0\n'
        f'[ "$1" = "run" ] && {{ echo "{NPM_MARKER}"; exit {NPM_EXIT}; }}\n'
        "exit 0\n",
    )
    _write_stub(stub_bin, "code", 'printf "1.98.2\\nabcdef\\nstub\\n"\n')
    _write_stub(
        stub_bin,
        "xcode-select",
        f'[ "$1" = "-p" ] && {{ echo "{clt}"; exit 0; }}\nexit 1\n',
    )

    env = {
        "HOME": str(home),
        "PATH": f"{stub_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
    }
    return {"script": script, "env": env, "root": root}


def run_install(sandbox: dict, use_pty: bool) -> subprocess.CompletedProcess:
    """Run the sandboxed install.sh, optionally under a pty with EOF input.

    ``use_pty=False`` mirrors the webapp update button (detached, no
    controlling terminal — ``start_new_session`` guarantees that even
    when pytest itself runs from a terminal).  ``use_pty=True`` uses
    ``script(1)`` to attach a real controlling pty whose input hits EOF
    immediately, the exact condition that crashed the unguarded
    ``read``.
    """
    if use_pty:
        if sys.platform == "darwin":
            cmd = ["script", "-q", "/dev/null", "bash", str(sandbox["script"])]
        else:
            cmd = ["script", "-qec", f"bash '{sandbox['script']}'", "/dev/null"]
    else:
        cmd = ["bash", str(sandbox["script"])]
    return subprocess.run(
        cmd,
        cwd=str(sandbox["script"].parent),
        env=sandbox["env"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        timeout=120,
        text=True,
        errors="replace",
        check=False,
    )


def test_git_upgrade_question_without_tty_does_not_crash(tmp_path: Path) -> None:
    """Webapp update path: no tty, old git, failing brew — must proceed.

    Previously the default-Yes answer ran ``upgrade_git`` which aborted
    the whole update when Homebrew was missing from the daemon's PATH
    or failed; now it must warn and continue to the build step.
    """
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=False)
    assert "older than the required version" in result.stdout, result.stdout
    assert "defaulting to Yes" in result.stdout, result.stdout
    assert NPM_MARKER in result.stdout, (
        "install.sh died at the git-upgrade question instead of "
        f"continuing to the build step:\n{result.stdout}"
    )
    assert result.returncode == NPM_EXIT, result.stdout


def test_git_upgrade_question_with_eof_tty_does_not_crash(tmp_path: Path) -> None:
    """Terminal update path with EOF input: ``read`` fails — must proceed.

    /dev/tty opens fine under ``script(1)`` but the pty input is at EOF,
    so the unguarded ``read`` used to return non-zero and ``set -e``
    silently killed the script right at the question.
    """
    sandbox = make_sandbox(tmp_path)
    result = run_install(sandbox, use_pty=True)
    assert "older than the required version" in result.stdout, result.stdout
    assert NPM_MARKER in result.stdout, (
        "install.sh died at the git-upgrade question (EOF on /dev/tty) "
        f"instead of continuing to the build step:\n{result.stdout}"
    )


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
    assert NPM_MARKER in result.stdout, result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
