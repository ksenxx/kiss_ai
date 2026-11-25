# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The curl bootstrap on a Mac that has no git.

On a Mac with neither Xcode nor the Command Line Tools ``/usr/bin/git``
is a stub.  ``scripts/install.sh`` has to get KISS Sorcar onto such a
Mac without bringing up the installer of Apple's developer tools: no
``xcode-select --install`` and no Homebrew bootstrap, whose installer
installs them too.  It downloads the repository as a zip archive and
hands over to the ``install.sh`` in it.

The test runs the real ``scripts/install.sh`` with a home directory of
its own and, first on ``PATH``, the commands of such a Mac: a ``git``
that fails the way the stub does once ``DEVELOPER_DIR`` is set, an
``xcode-select`` that knows no developer directory, and no ``brew``.
``curl`` is a script that delivers an archive the test prepared, not
GitHub's (the tests use no network); it and ``xcode-select`` record how
they were called.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from kiss.core.macos_developer_tools import COMMAND_LINE_TOOLS_DIR
from kiss.tests.conftest import posix_only

BOOTSTRAP = Path(__file__).resolve().parents[4] / "scripts" / "install.sh"
ARCHIVE_URL = "https://github.com/ksenxx/kiss_ai/archive/refs/heads/main.zip"

pytestmark = [
    posix_only("runs the real bash bootstrap scripts/install.sh"),
    pytest.mark.skipif(shutil.which("unzip") is None, reason="the bootstrap unpacks with unzip"),
]


def write_command(bin_dir: Path, name: str, script: str) -> None:
    """Write the command *name* running *script* into *bin_dir*."""
    command = bin_dir / name
    command.write_text(f"#!/bin/bash\n{script}\n")
    command.chmod(0o755)


def test_mac_without_git_installs_from_the_zip_archive(tmp_path: Path) -> None:
    """No git: the archive is downloaded, and nothing that installs developer tools is run."""
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    home.mkdir()
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    archive = tmp_path / "prepared.zip"
    with zipfile.ZipFile(archive, "w") as prepared:
        prepared.writestr(
            "kiss_ai-main/install.sh",
            '#!/bin/bash\necho "DEVELOPER_DIR=${DEVELOPER_DIR:-}" > "$HOME/handed-over.txt"\n',
        )
    write_command(bin_dir, "uname", "echo Darwin")
    write_command(bin_dir, "git", 'echo "xcrun: error: missing DEVELOPER_DIR path" >&2\nexit 1')
    write_command(
        bin_dir, "xcode-select", f'echo "xcode-select $*" >> "{calls}"\n[ "$1" != -p ] || exit 2'
    )
    write_command(
        bin_dir, "curl", f'echo "curl $*" >> "{calls}"\n[ "$2" != -o ] || cp "{archive}" "$3"'
    )
    result = subprocess.run(
        ["bash", str(BOOTSTRAP)],
        cwd=tmp_path,
        env={"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Continuing without git: KISS Sorcar is downloaded as a zip archive instead." in output
    # The installer from the archive ran, with the developer-tool stubs silenced.
    handed_over = (home / "handed-over.txt").read_text()
    assert handed_over == f"DEVELOPER_DIR={COMMAND_LINE_TOOLS_DIR}\n"
    assert (home / ".kiss" / "kiss_ai" / "install.sh").is_file()
    assert not (home / ".kiss" / "main.zip").exists()
    # One question to xcode-select (is there a developer directory?) and
    # one download: no ``xcode-select --install``, no Homebrew installer.
    assert calls.read_text().splitlines() == [
        "xcode-select -p",
        f"curl -L -o main.zip {ARCHIVE_URL}",
    ]
