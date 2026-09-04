# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``update_repo``'s early VSIX normalization runs the guard exactly once.

Audit 2026-09-03 (vscode-installer): the pre-stash normalization added to
``update_repo`` in ``install.sh`` by commit 1aad2045b ran
``guard_vsix_tracking`` twice —

    if ! guard_vsix_tracking "$PROJECT_DIR" >/dev/null 2>&1; then
        guard_vsix_tracking "$PROJECT_DIR" || exit 1
    fi

— a silenced probe followed by a loud re-run, repeating every git
operation of the guard on the failure path purely to shape output.  The
fix captures the single run's output and re-emits it only on failure,
which is behaviorally identical but does the work once.

The tests drive the functions extracted VERBATIM from install.sh (the
pattern of test_install_update_stash_vsix_lifecycle.py) against real git
repositories, with git itself wrapped by a PATH shim that logs every
invocation before delegating to the real binary — so the "runs once"
property is observed end to end, without touching the code under test.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
VSIX_REL = "src/kiss/agents/vscode/kiss-sorcar.vsix"


def extract_function(name: str) -> str:
    """Return the named shell function's source verbatim from install.sh."""
    text = INSTALL_SCRIPT.read_text()
    match = re.search(
        rf"^{name}\(\) \{{\n.*?^\}}$", text, re.MULTILINE | re.DOTALL
    )
    assert match, f"{name}() not found in install.sh"
    return match.group(0)


def make_git_shim(tmp_path: Path) -> tuple[Path, Path]:
    """Install a logging ``git`` wrapper; return (shim dir, log file)."""
    real_git = shutil.which("git")
    assert real_git, "git not on PATH"
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    log_file = tmp_path / "git-calls.log"
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/bash\n"
        f'printf \'%s\\n\' "$*" >> "{log_file}"\n'
        f'exec "{real_git}" "$@"\n'
    )
    shim.chmod(0o755)
    return shim_dir, log_file


def make_dev_repo_with_forced_vsix(tmp_path: Path) -> Path:
    """A development repo whose VSIX was accidentally ``git add -f``-ed."""
    dev = tmp_path / "dev"
    (dev / Path(VSIX_REL).parent).mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(dev)],
        check=True, capture_output=True,
    )
    git = ["git", "-C", str(dev)]
    subprocess.run([*git, "config", "user.email", "t@t"], check=True)
    subprocess.run([*git, "config", "user.name", "T"], check=True)
    (dev / ".gitignore").write_text("*.vsix\n")
    (dev / "README.md").write_text("dev repo\n")
    subprocess.run(
        [*git, "add", ".gitignore", "README.md"], check=True,
        capture_output=True,
    )
    subprocess.run(
        [*git, "commit", "-q", "-m", "init"], check=True, capture_output=True,
    )
    (dev / VSIX_REL).write_bytes(b"rebuilt")
    subprocess.run(
        [*git, "add", "-f", VSIX_REL], check=True, capture_output=True,
    )
    return dev


def run_update_repo(repo: Path, shim_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run install.sh's real update_repo against *repo* under the git shim."""
    script = "\n".join(
        [
            "set -eo pipefail",
            f'export PATH="{shim_dir}:$PATH"',
            f'PROJECT_DIR="{repo}"',
            "STASHED_CHANGES=0",
            extract_function("guard_vsix_tracking"),
            extract_function("restore_stashed_changes"),
            extract_function("update_repo"),
            "update_repo",
        ]
    )
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )


def test_dev_repo_failure_runs_guard_once_and_still_aborts(
    tmp_path: Path,
) -> None:
    """The force-added-VSIX hard error fires from a SINGLE guard run.

    The guard's dev-repo detection is the pair ``rev-parse --verify
    HEAD:<vsix>`` + ``ls-files --error-unmatch <vsix>``; the old
    probe-then-rerun shape executed both twice.  The error message and
    exit status must be unchanged.
    """
    shim_dir, log_file = make_git_shim(tmp_path)
    dev = make_dev_repo_with_forced_vsix(tmp_path)

    result = run_update_repo(dev, shim_dir)

    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert combined.count("must remain ignored") == 1, combined
    calls = log_file.read_text().splitlines()
    tracking_checks = [c for c in calls if "--error-unmatch" in c]
    assert len(tracking_checks) == 1, (
        "guard_vsix_tracking ran its tracked-VSIX check "
        f"{len(tracking_checks)} times for one update_repo call:\n"
        + "\n".join(calls)
    )
    head_probes = [c for c in calls if "rev-parse --verify" in c]
    assert len(head_probes) == 1, (
        "guard_vsix_tracking probed HEAD for the VSIX "
        f"{len(head_probes)} times for one update_repo call:\n"
        + "\n".join(calls)
    )


def test_healthy_dev_repo_normalization_stays_quiet_and_single(
    tmp_path: Path,
) -> None:
    """On the healthy no-op path the guard runs once and prints nothing.

    A clean development repo (untracked VSIX) is the common case for
    every Update-button run in a dev checkout; the normalization must
    neither spam output nor repeat work there.
    """
    shim_dir, log_file = make_git_shim(tmp_path)
    dev = make_dev_repo_with_forced_vsix(tmp_path)
    subprocess.run(
        ["git", "-C", str(dev), "rm", "-q", "--cached", VSIX_REL],
        check=True, capture_output=True,
    )

    result = run_update_repo(dev, shim_dir)

    # No origin: update_repo warns about the fetch and returns 0.
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Restored release-shipped" not in result.stdout + result.stderr
    calls = log_file.read_text().splitlines()
    tracking_checks = [c for c in calls if "--error-unmatch" in c]
    assert len(tracking_checks) == 1, "\n".join(calls)
