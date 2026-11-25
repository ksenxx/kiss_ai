# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the guard against macOS's "install the developer tools" dialog.

On a Mac with neither Xcode nor the Command Line Tools, ``/usr/bin/git``
and some seventy other commands are stubs that open that dialog when
they are run.  The guard is the environment variable ``DEVELOPER_DIR``,
exported

* by ``import kiss`` (``kiss.core.macos_developer_tools``), for the
  daemon, the CLI and every process they start, and
* by the ``kiss-macos-developer-tools`` block of ``install.sh`` and
  ``scripts/install.sh``, for their own git and uv calls.

Both ask ``xcode-select -p`` whether the tools are there.  The tests run
them with an ``xcode-select`` of their own first on ``PATH`` for the two
answers the real one gives only on a Mac without the tools (a failure,
a directory that does not exist), and with the real one for what it
does on every Mac: repeat a ``DEVELOPER_DIR`` it inherits.

Not run here: a real stub with the guard in place.  On a Mac that has
the tools there is no stub, and on one that has none a broken guard
would open the dialog on the screen of whoever runs the tests.  That
run was done by hand on a Mac without the tools (a virtual machine):
after ``import kiss``, ``git``, ``/usr/bin/python3`` and ``make`` exit 1
with "xcrun: error: missing DEVELOPER_DIR path" and no dialog opens.

Of ``developer_tools_missing``'s ``except (OSError, TimeoutExpired)``
clause the tests reach the ``OSError`` half (no ``xcode-select`` on
``PATH``); an ``xcode-select`` that hangs would hold a test for the ten
seconds of the timeout to run the same statement.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import kiss
from kiss.core.macos_developer_tools import COMMAND_LINE_TOOLS_DIR
from kiss.tests.conftest import posix_only

REPO = Path(__file__).resolve().parents[4]
INSTALLERS = (REPO / "install.sh", REPO / "scripts" / "install.sh")
ON_A_MAC = sys.platform == "darwin"
mac_only = pytest.mark.skipif(not ON_A_MAC, reason="the Python guard acts on macOS only")


def write_tool(directory: Path, name: str, script: str) -> str:
    """Write the command *name* running *script* into *directory*; return the directory."""
    directory.mkdir(exist_ok=True)
    tool = directory / name
    tool.write_text(f"#!/bin/bash\n{script}\n")
    tool.chmod(0o755)
    return str(directory)


def developer_dir_after_import(path: str, inherited: str | None = None) -> str:
    """Return what ``import kiss`` leaves in ``DEVELOPER_DIR`` when ``PATH`` is *path*."""
    env = {"PATH": path, "PYTHONPATH": str(Path(kiss.__file__).resolve().parents[1])}
    if inherited is not None:
        env["DEVELOPER_DIR"] = inherited
    result = subprocess.run(
        [sys.executable, "-c", "import os, kiss; print(os.environ.get('DEVELOPER_DIR', ''))"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# What ``xcode-select -p`` does, and the DEVELOPER_DIR the guard must leave.
ANSWERS = [
    pytest.param("exit 2", COMMAND_LINE_TOOLS_DIR, id="no developer directory selected"),
    pytest.param("echo /no/such/developer/dir", COMMAND_LINE_TOOLS_DIR, id="directory gone"),
    pytest.param("echo /usr", "", id="tools installed"),
]


@mac_only
@pytest.mark.parametrize(("answer", "exported"), ANSWERS)
def test_import_kiss_silences_the_dialog_when_the_tools_are_missing(
    tmp_path: Path, answer: str, exported: str
) -> None:
    """``import kiss`` exports the variable exactly when ``xcode-select`` finds no tools."""
    path = write_tool(tmp_path / "bin", "xcode-select", answer)
    assert developer_dir_after_import(path) == exported


@mac_only
def test_import_kiss_keeps_what_a_parent_already_decided(tmp_path: Path) -> None:
    """A process started by a silenced daemon keeps the variable without asking again."""
    asked = tmp_path / "asked"
    # ``: >`` is a shell builtin: ``PATH`` holds this one command only.
    path = write_tool(tmp_path / "bin", "xcode-select", f': > "{asked}"\nexit 2')
    assert developer_dir_after_import(path) == COMMAND_LINE_TOOLS_DIR
    assert asked.exists()  # the first process of the tree asks
    asked.unlink()
    assert developer_dir_after_import(path, COMMAND_LINE_TOOLS_DIR) == COMMAND_LINE_TOOLS_DIR
    assert not asked.exists()
    # The real xcode-select repeats the directory it inherits: one that
    # exists (an Xcode the user selected) is kept, one that does not is
    # replaced by the directory the Command Line Tools install into.
    assert developer_dir_after_import("/usr/bin", str(tmp_path)) == str(tmp_path)
    assert developer_dir_after_import("/usr/bin", str(tmp_path / "gone")) == COMMAND_LINE_TOOLS_DIR


@mac_only
def test_import_kiss_without_xcode_select_changes_nothing(tmp_path: Path) -> None:
    """A ``PATH`` without ``xcode-select`` is no reason to export anything."""
    (tmp_path / "bin").mkdir()
    assert developer_dir_after_import(str(tmp_path / "bin")) == ""


@posix_only("the stand-in for xcode-select is a shell script")
@pytest.mark.skipif(ON_A_MAC, reason="what the guard does off macOS")
def test_import_kiss_leaves_other_systems_alone(tmp_path: Path) -> None:
    """Off macOS nothing is exported, whatever an ``xcode-select`` on ``PATH`` says."""
    path = write_tool(tmp_path / "bin", "xcode-select", "exit 2")
    assert developer_dir_after_import(path) == ""


def developer_tools_block(installer: Path) -> str:
    """Return the ``kiss-macos-developer-tools`` block of *installer*."""
    match = re.search(
        r"^# BEGIN: kiss-macos-developer-tools.*?\n(.*?)^# END: kiss-macos-developer-tools$",
        installer.read_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"kiss-macos-developer-tools block not found in {installer}"
    return match.group(1)


def developer_dir_after_block(path: str) -> str:
    """Return ``DEVELOPER_DIR`` after the installers' block has run with ``PATH`` *path*."""
    script = f'{developer_tools_block(INSTALLERS[0])}\necho "${{DEVELOPER_DIR:-}}"\n'
    result = subprocess.run(
        ["/bin/bash", "-euo", "pipefail", "-c", script],
        env={"PATH": path},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@posix_only("the installers are bash scripts")
def test_both_installers_run_the_same_block() -> None:
    """``scripts/install.sh`` runs git before ``install.sh`` does: same guard, same text."""
    assert developer_tools_block(INSTALLERS[0]) == developer_tools_block(INSTALLERS[1])


@posix_only("the installers are bash scripts")
@pytest.mark.parametrize(("answer", "exported"), ANSWERS)
def test_installer_block_silences_the_dialog_on_a_mac(
    tmp_path: Path, answer: str, exported: str
) -> None:
    """On a Mac the block exports the variable exactly when the tools are missing."""
    write_tool(tmp_path / "bin", "uname", "echo Darwin")
    path = write_tool(tmp_path / "bin", "xcode-select", answer)
    assert developer_dir_after_block(path) == exported


@posix_only("the installers are bash scripts")
def test_installer_block_leaves_other_systems_alone(tmp_path: Path) -> None:
    """On Linux the block exports nothing, whatever an ``xcode-select`` on ``PATH`` says."""
    write_tool(tmp_path / "bin", "uname", "echo Linux")
    path = write_tool(tmp_path / "bin", "xcode-select", "exit 2")
    assert developer_dir_after_block(path) == ""
