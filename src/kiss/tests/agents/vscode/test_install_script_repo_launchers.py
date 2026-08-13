# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``install.sh`` must install ``rsorcar`` and ``sorcar-docker`` launchers
into ``$BIN_DIR`` the same way the ``sorcar`` CLI is installed by the VS Code
extension (``installCliScript`` in DependencyInstaller.ts): a thin wrapper
script that ``exec``s the real command.

The critical semantic: both repo-root scripts locate their own directory
(``dirname "$0"`` / ``BASH_SOURCE``) and treat it as the KISS Sorcar
checkout — ``rsorcar`` deploys the folder in which the script is actually
present, and ``sorcar-docker`` builds the Docker image from that folder.  A
symlink or a copy placed in ``~/.local/bin`` would make them resolve
``~/.local/bin`` instead of the checkout.  The launcher must therefore be a
wrapper that invokes the *real* script inside the checkout, so the script's
self-resolved directory stays the checkout.

The tests below extract the ``install_repo_script_launcher`` function
verbatim from ``install.sh`` and run it for real in a sandbox: they install
a launcher for a fake checkout script, execute the launcher from an
unrelated working directory, and assert the script resolves its own
directory to the checkout (never the bin dir) and receives its arguments
unchanged.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"

FAKE_SCRIPT = """\
#!/bin/bash
# Mimics ./rsorcar: resolves and reports the folder the script is actually in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "ARGS=$*"
"""


def extract_launcher_function() -> str:
    """Return the ``install_repo_script_launcher`` function body from install.sh."""
    text = INSTALL_SCRIPT.read_text()
    match = re.search(
        r"^install_repo_script_launcher\(\) \{\n.*?^\}$", text, re.MULTILINE | re.DOTALL
    )
    assert match, "install_repo_script_launcher() not found in install.sh"
    return match.group(0)


def run_launcher_install(tmp_path: Path, name: str) -> tuple[Path, Path, str]:
    """Run the extracted function in a sandbox; return (bin_dir, project_dir, output)."""
    project_dir = tmp_path / "checkout"
    bin_dir = tmp_path / "bin"
    project_dir.mkdir(exist_ok=True)
    bin_dir.mkdir(exist_ok=True)
    script = project_dir / name
    script.write_text(FAKE_SCRIPT)
    script.chmod(0o755)

    harness = (
        f'PROJECT_DIR="{project_dir}"\n'
        f'BIN_DIR="{bin_dir}"\n'
        f"{extract_launcher_function()}\n"
        f"install_repo_script_launcher {name}\n"
    )
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return bin_dir, project_dir, proc.stdout


@pytest.mark.parametrize("name", ["rsorcar", "sorcar-docker"])
def test_launcher_is_installed_executable_wrapper(tmp_path: Path, name: str) -> None:
    """The launcher is created in BIN_DIR, executable, and is not a symlink/copy."""
    bin_dir, project_dir, output = run_launcher_install(tmp_path, name)
    launcher = bin_dir / name
    assert launcher.is_file()
    assert not launcher.is_symlink()
    assert launcher.stat().st_mode & stat.S_IXUSR
    assert launcher.read_text() != FAKE_SCRIPT, "launcher must not be a copy"
    assert str(project_dir / name) in launcher.read_text()
    assert f"Installed {launcher}" in output


@pytest.mark.parametrize("name", ["rsorcar", "sorcar-docker"])
def test_launcher_resolves_checkout_not_bin_dir(tmp_path: Path, name: str) -> None:
    """Running the launcher from elsewhere, the real script must still resolve
    its own directory to the checkout (rsorcar deploys that folder)."""
    bin_dir, project_dir, _ = run_launcher_install(tmp_path, name)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc = subprocess.run(
        [str(bin_dir / name), "user@host", "--flag", "value with space"],
        capture_output=True,
        text=True,
        cwd=elsewhere,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"SCRIPT_DIR={project_dir}" in proc.stdout
    assert str(bin_dir) not in proc.stdout
    assert "ARGS=user@host --flag value with space" in proc.stdout


def test_missing_script_warns_and_does_not_fail(tmp_path: Path) -> None:
    """A checkout without the script warns and returns success (no launcher)."""
    project_dir = tmp_path / "checkout"
    bin_dir = tmp_path / "bin"
    project_dir.mkdir()
    bin_dir.mkdir()
    harness = (
        f'PROJECT_DIR="{project_dir}"\n'
        f'BIN_DIR="{bin_dir}"\n'
        f"{extract_launcher_function()}\n"
        f"install_repo_script_launcher rsorcar\n"
    )
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stdout
    assert not (bin_dir / "rsorcar").exists()


def test_install_sh_installs_both_launchers(tmp_path: Path) -> None:
    """install.sh's main body must install launchers for both repo scripts.

    Verified behaviorally: extract the invocation lines from install.sh and
    run them against the real repo-root scripts with a sandbox BIN_DIR, then
    check both launchers exist and point at this checkout's scripts.
    """
    text = INSTALL_SCRIPT.read_text()
    calls = re.findall(
        r"^\s*(install_repo_script_launcher \S+)$", text, re.MULTILINE
    )
    assert "install_repo_script_launcher rsorcar" in calls
    assert "install_repo_script_launcher sorcar-docker" in calls

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    harness = (
        f'PROJECT_DIR="{REPO}"\n'
        f'BIN_DIR="{bin_dir}"\n'
        f"{extract_launcher_function()}\n" + "\n".join(calls) + "\n"
    )
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("rsorcar", "sorcar-docker"):
        launcher = bin_dir / name
        assert launcher.is_file(), f"{name} launcher missing"
        assert str(REPO / name) in launcher.read_text()
        assert os.access(launcher, os.X_OK)
