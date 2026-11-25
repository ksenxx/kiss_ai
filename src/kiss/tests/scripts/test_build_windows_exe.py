# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the Windows setup program.

``scripts/build-windows-exe.sh`` is run for real against this checkout
(it builds on macOS and Linux too, with NSIS's ``makensis``) and the
setup program it writes is unpacked with 7-Zip, which reads what the
program would write into ``%USERPROFILE%\\.kiss\\kiss_pkg``: the payload
and the uninstaller.  The payload must be the one the Mac package
carries, as ``scripts/installer-payload.sh`` stages it.

The tests that build are skipped where ``makensis`` or 7-Zip (``7zz``
or ``7z``) is not installed.

Not run here: the setup program itself and its ``setup.ps1``.  They
need Windows, download Git, uv, Python and Chromium and replace the
user's kiss-web daemon.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest

from kiss import __version__
from kiss.tests.conftest import posix_only

pytestmark = posix_only("runs the bash build script and installer-payload.sh")

REPO = Path(__file__).resolve().parents[4]
BUILD = REPO / "scripts" / "build-windows-exe.sh"
SEVEN_ZIP = shutil.which("7zz") or shutil.which("7z")


class Setup(NamedTuple):
    """A built setup program: the file, what 7-Zip unpacked from it, the build's output."""

    file: Path
    unpacked: Path
    output: str


def build(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the build script with *args*."""
    bash = shutil.which("bash") or "bash"
    return subprocess.run(
        [bash, str(BUILD), *args], capture_output=True, text=True, timeout=300, env=env, check=False
    )


def files(root: Path) -> dict[str, int]:
    """The files under *root* by relative path, with their sizes; no ``$PLUGINSDIR`` of 7-Zip."""
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.relative_to(root).parts[0].startswith("$")
    }


@pytest.fixture(scope="module")
def setup_program(tmp_path_factory: pytest.TempPathFactory) -> Setup:
    """The setup program of this checkout, built and unpacked once per module."""
    if not (shutil.which("makensis") and SEVEN_ZIP):
        pytest.skip("building and unpacking the setup program needs makensis (NSIS) and 7-Zip")
    root = tmp_path_factory.mktemp("windows-exe")
    file = root / "out" / "setup.exe"
    result = build("-o", str(file))
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run(
        [SEVEN_ZIP, "x", "-y", f"-o{root / 'unpacked'}", str(file)], check=True, capture_output=True
    )
    return Setup(file, root / "unpacked", result.stdout)


def test_setup_program_holds_the_app_with_its_installer(setup_program: Setup) -> None:
    """It is a Windows program that unpacks what install.sh installs, install.sh and the commit."""
    assert setup_program.file.read_bytes()[:2] == b"MZ"
    assert f"KISS Sorcar {__version__}" in setup_program.output
    assert "It is not signed" in setup_program.output
    root = setup_program.unpacked
    for name in (
        "install.sh",
        "scripts/install.sh",
        "scripts/windows-exe/setup.ps1",
        "pyproject.toml",
        "uv.lock",
        "src/kiss/__init__.py",
    ):
        assert (root / name).is_file(), name
    for name in ("install.sh", "src/kiss/core/_version.py"):
        assert (root / name).read_bytes() == (REPO / name).read_bytes(), name
    assert (root / "uninstall.exe").read_bytes()[:2] == b"MZ"
    if (REPO / ".git").exists():
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
        assert (root / "src/kiss/core/_commit.txt").read_text().strip() == head != ""
    assert not (root / "src/kiss/tests").exists()
    assert sorted(p.name for p in (root / "src/kiss/webapp").iterdir()) == ["media"]
    assert sorted(p.name for p in root.iterdir() if not p.name.startswith("$")) == [
        "Dockerfile",
        "LICENSE",
        "README.md",
        "install.sh",
        "pyproject.toml",
        "rsorcar",
        "scripts",
        "sorcar-docker",
        "src",
        "uninstall.exe",
        "uv.lock",
    ]
    for excluded in ("installers", ".git", "node_modules", "__pycache__"):
        assert not list(root.rglob(excluded)), excluded


def test_payload_is_the_one_the_mac_package_carries(setup_program: Setup, tmp_path: Path) -> None:
    """The setup program unpacks the files ``stage_payload`` stages, and its uninstaller."""
    staged = tmp_path / "staged"
    script = (
        f"PROJECT_DIR={shlex.quote(str(REPO))}\n"
        'die() { echo "$*" >&2; exit 1; }\n'
        '. "$PROJECT_DIR/scripts/installer-payload.sh"\n'
        f"stage_payload {shlex.quote(str(staged))}\n"
    )
    subprocess.run(["bash", "-c", script], check=True, capture_output=True, timeout=120)
    unpacked = files(setup_program.unpacked)
    assert unpacked.pop("uninstall.exe") > 0
    assert unpacked == files(staged)
    assert not Path(f"{staged}.files").exists()


def test_help_prints_the_usage() -> None:
    """``-h`` prints the options and builds nothing."""
    result = build("-h")
    assert result.returncode == 0
    assert result.stdout.startswith("Build the Windows setup program of KISS Sorcar.")
    for option in ("scripts/build-windows-exe.sh [-o EXE]", "--output", "--help"):
        assert option in result.stdout
    assert result.stdout.rstrip().endswith("print this text")


@pytest.mark.parametrize(
    ("args", "message"),
    [(["--bogus"], "unknown argument: --bogus"), (["-o"], "-o needs a value")],
)
def test_bad_arguments_are_refused(args: list[str], message: str) -> None:
    """A wrong command line stops with its reason before anything is built."""
    result = build(*args)
    assert result.returncode == 1
    assert f"build-windows-exe.sh: {message}" in result.stderr
    assert "Built" not in result.stdout


def test_build_without_makensis_says_what_to_install(tmp_path: Path) -> None:
    """Without NSIS the build stops at once and names the package to install."""
    # A PATH of the two programs the script runs before it looks for makensis.
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("dirname", "sed"):
        (tools / tool).symlink_to(shutil.which(tool) or tool)
    output = tmp_path / "out" / "setup.exe"
    result = build("-o", str(output), env={"PATH": str(tools), "HOME": str(tmp_path)})
    assert result.returncode == 1
    assert "build-windows-exe.sh: makensis not found: install NSIS" in result.stderr
    assert "Built" not in result.stdout
    assert not output.parent.exists()
