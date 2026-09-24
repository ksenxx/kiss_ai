# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``install.sh`` applies ``.brand/`` only for the build.

A white-label distribution puts its ``brand.json`` / ``brand.css`` / icon
files into the git-ignored ``.brand/`` directory of a KISS Sorcar
checkout.  ``install.sh`` snapshots the checkout's own copies, swaps the
overlay over ``src/kiss/agents/vscode/media/`` right before the extension
is packaged and copies the snapshot back afterwards (package.json keeps
the version copy-kiss.sh synced), so the checkout ends up exactly as a
stock build leaves it while the installed extension carries the brand.
Without ``.brand/`` nothing happens: the checked-in brand is KISS Sorcar.

The functions and the step [4/5] build block are extracted from
``install.sh`` and run under bash against a throw-away git repository
with a stub ``npm``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

_REPO = Path(__file__).resolve().parents[3]
_INSTALL_SH = (_REPO / "install.sh").read_text(encoding="utf-8")
MEDIA_REL = Path("src/kiss/agents/vscode/media")
MANIFEST_REL = Path("src/kiss/agents/vscode/package.json")
BRAND_FILES = ("brand.json", "brand.css", "kiss-icon.svg", "kiss-icon.png", "thumbnail.jpeg")
STOCK_MANIFEST = '{"displayName": "KISS Sorcar", "version": "0.0.0"}\n'

pytestmark = posix_only("runs the brand-overlay block of install.sh under bash")


def _function(name: str) -> str:
    """Return the ``name() { ... }`` definition from install.sh."""
    m = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", _INSTALL_SH, re.S | re.M)
    assert m is not None, f"install.sh no longer defines {name}()"
    return m.group(0)


def _overlay_functions() -> str:
    """The overlay variables and the two functions the build step uses."""
    variables = re.findall(
        r"^BRAND_(?:OVERLAY_FILES|MEDIA_REL|MANIFEST_REL|OVERLAY_BACKUP)=.*$", _INSTALL_SH, re.M
    )
    assert len(variables) == 4, variables
    return "\n".join(
        [*variables, _function("apply_brand_overlay"), _function("restore_brand_overlay")]
    )


def _build_block() -> str:
    """The step [4/5] lines from ``build_rc=0`` up to the ``cd "$PROJECT_DIR"`` after the build."""
    m = re.search(
        r'^\s*build_rc=0\n\s*apply_brand_overlay "\$PROJECT_DIR".*?^\s*fi\n'
        r'(?=\s*cd "\$PROJECT_DIR")',
        _INSTALL_SH,
        re.S | re.M,
    )
    assert m is not None, "install.sh step [4/5] no longer wraps the build in the brand overlay"
    assert "restore_brand_overlay" in m.group(0)
    return m.group(0)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _bash(
    script: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def _make_checkout(root: Path) -> Path:
    media = root / MEDIA_REL
    media.mkdir(parents=True)
    for name in BRAND_FILES:
        (media / name).write_bytes(b"stock " + name.encode())
    (root / MANIFEST_REL).write_text(STOCK_MANIFEST)
    return root


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A committed stock-shaped checkout: five media files and package.json."""
    root = _make_checkout(tmp_path / "kiss_ai")
    (root / ".gitignore").write_text("/.brand/\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "stock")
    return root


def _write_overlay(root: Path, names: tuple[str, ...] = BRAND_FILES) -> None:
    overlay = root / ".brand"
    overlay.mkdir()
    for name in names:
        (overlay / name).write_bytes(b"branded " + name.encode())


# What the build does to the checkout between apply and restore: copy-kiss.sh
# syncs the version and apply-brand.js rewrites the display strings.
FAKE_BUILD = (
    "cat {media}/brand.json > {out}\n"
    'printf \'%s\\n\' \'{{"displayName": "Seamless Loop", "version": "1.2.3"}}\' > {manifest}\n'
)


def _apply_build_restore(root: Path, build: str) -> subprocess.CompletedProcess[str]:
    script = (
        f"{_overlay_functions()}\n"
        f'apply_brand_overlay "{root}"\necho "backup-dir=${{BRAND_OVERLAY_BACKUP}}"\n'
        f'{build}\nrestore_brand_overlay "{root}"\n'
        'echo "backup=${BRAND_OVERLAY_BACKUP}"\n'
    )
    return _bash(script, root)


def test_apply_swaps_only_overlaid_files_and_restore_brings_back_local_copies(
    checkout: Path,
) -> None:
    """The snapshot, not git, is restored: a local edit to an overlaid file survives."""
    _write_overlay(checkout, ("brand.json", "kiss-icon.png"))
    media = checkout / MEDIA_REL
    (media / "brand.json").write_bytes(b"locally edited brand.json")
    (media / "brand.css").write_bytes(b"locally edited brand.css")
    seen = checkout.parent / "seen-brand.json"
    build = FAKE_BUILD.format(media=media, out=seen, manifest=checkout / MANIFEST_REL)
    proc = _apply_build_restore(checkout, build)
    assert proc.returncode == 0, proc.stderr
    assert "Applying brand overlay" in proc.stdout
    assert f"{MEDIA_REL}/brand.json" in proc.stdout and "kiss-icon.png" in proc.stdout
    assert "brand.css" not in proc.stdout
    assert "Restored the checkout's own brand files" in proc.stdout
    assert proc.stdout.rstrip().endswith("backup=")
    assert seen.read_bytes() == b"branded brand.json", "the build saw the overlay"
    assert (media / "brand.json").read_bytes() == b"locally edited brand.json"
    assert (media / "brand.css").read_bytes() == b"locally edited brand.css"
    assert (media / "kiss-icon.png").read_bytes() == b"stock kiss-icon.png"
    # package.json: pre-build content, but the version the build synced.
    assert json.loads((checkout / MANIFEST_REL).read_text()) == {
        "displayName": "KISS Sorcar",
        "version": "1.2.3",
    }
    assert sorted(line[3:] for line in _git(checkout, "status", "--porcelain").splitlines()) == [
        str(MEDIA_REL / "brand.css"),
        str(MEDIA_REL / "brand.json"),
        str(MANIFEST_REL),
    ]
    backup_dir = re.search(r"^backup-dir=(.+)$", proc.stdout, re.M)
    assert backup_dir is not None and "kiss-brand-backup." in backup_dir.group(1)
    assert not Path(backup_dir.group(1)).exists(), "the snapshot directory is removed"


def test_restore_leaves_a_clean_clone_when_the_version_did_not_change(checkout: Path) -> None:
    """The release clone case: same version before and after, so package.json is byte-identical."""
    _write_overlay(checkout)
    build = (
        f'printf \'%s\\n\' \'{{"displayName": "Seamless Loop", "version": "0.0.0"}}\''
        f" > {checkout / MANIFEST_REL}\n"
    )
    proc = _apply_build_restore(checkout, build)
    assert proc.returncode == 0, proc.stderr
    assert (checkout / MANIFEST_REL).read_text() == STOCK_MANIFEST
    assert _git(checkout, "status", "--porcelain") == ""
    assert (checkout / ".brand" / "brand.json").read_bytes() == b"branded brand.json"


def test_restore_falls_back_to_the_snapshot_when_the_manifest_is_unreadable(
    checkout: Path,
) -> None:
    """A build that leaves package.json unparsable gets the pre-build copy back."""
    _write_overlay(checkout, ("brand.json",))
    proc = _apply_build_restore(checkout, f"echo '<<<<<<< conflict' > {checkout / MANIFEST_REL}\n")
    assert proc.returncode == 0, proc.stderr
    assert (checkout / MANIFEST_REL).read_text() == STOCK_MANIFEST
    assert _git(checkout, "status", "--porcelain") == ""


def test_without_overlay_directory_nothing_happens(checkout: Path) -> None:
    """No .brand/: both functions are silent no-ops, even outside a git repo."""
    (checkout / MEDIA_REL / "brand.json").write_bytes(b"locally edited")
    plain = checkout.parent / "plain"
    plain.mkdir()
    script = (
        f"{_overlay_functions()}\n"
        f'apply_brand_overlay "{checkout}"; restore_brand_overlay "{checkout}"\n'
        f'apply_brand_overlay "{plain}"; restore_brand_overlay "{plain}"\n'
    )
    proc = _bash(script, checkout)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "" and proc.stderr == ""
    assert (checkout / MEDIA_REL / "brand.json").read_bytes() == b"locally edited"


def test_overlay_works_in_a_plain_directory_copy(tmp_path: Path) -> None:
    """No git needed: a zip-extracted kiss_ai (scripts/install.sh fallback) restores too."""
    root = _make_checkout(tmp_path / "copy")
    _write_overlay(root, ("thumbnail.jpeg",))
    proc = _apply_build_restore(root, "")
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""
    assert (root / MEDIA_REL / "thumbnail.jpeg").read_bytes() == b"stock thumbnail.jpeg"
    assert (root / MANIFEST_REL).read_text() == STOCK_MANIFEST


def _run_build_block(checkout: Path, npm_script: str) -> subprocess.CompletedProcess[str]:
    """Run install.sh's build block with a stub npm and a pass-through heartbeat."""
    bin_dir = checkout.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    npm = bin_dir / "npm"
    npm.write_text("#!/bin/bash\n" + npm_script)
    npm.chmod(0o755)
    script = (
        f"{_overlay_functions()}\n"
        'run_with_heartbeat() { shift; "$@"; }\n'
        f'PROJECT_DIR="{checkout}"\ncd "$PROJECT_DIR/src/kiss/agents/vscode"\n'
        f"{_build_block()}\necho BUILD-CONTINUES\n"
    )
    tmp_dir = checkout.parent / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(checkout.parent),
        "TMPDIR": str(tmp_dir),
    }
    return _bash(script, checkout, env)


def test_build_block_packages_the_branded_media_then_restores(checkout: Path) -> None:
    """The stub build sees the branded media; the checkout is clean afterwards."""
    _write_overlay(checkout)
    npm_script = (
        'if [ "$2" = package ]; then cp media/brand.json "$HOME/packaged-brand.json"; fi\n'
        'echo "npm $*"\n'
    )
    proc = _run_build_block(checkout, npm_script)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.index("Applying brand overlay") < proc.stdout.index("npm run copy-kiss")
    assert proc.stdout.index("npm run copy-kiss") < proc.stdout.index("npm run package")
    assert proc.stdout.index("npm run package") < proc.stdout.index("Restored the checkout's")
    assert proc.stdout.rstrip().endswith("BUILD-CONTINUES")
    assert (checkout.parent / "packaged-brand.json").read_bytes() == b"branded brand.json"
    assert _git(checkout, "status", "--porcelain") == ""


def test_build_block_restores_the_checkout_when_packaging_fails(checkout: Path) -> None:
    """A failing ``npm run package`` still restores the files and exits with its status."""
    _write_overlay(checkout)
    proc = _run_build_block(checkout, 'echo "npm $*"; [ "$2" != package ] || exit 7\n')
    assert proc.returncode == 7, proc.stdout
    assert "Restored the checkout's own brand files" in proc.stdout
    assert "ERROR: extension build failed (exit 7)" in proc.stdout
    assert "BUILD-CONTINUES" not in proc.stdout
    assert _git(checkout, "status", "--porcelain") == ""


def test_build_block_skips_the_build_and_restores_when_apply_fails(checkout: Path) -> None:
    """An overlay naming a media file the checkout lacks aborts before npm runs, restored."""
    _write_overlay(checkout, ("brand.json", "brand.css"))
    (checkout / MEDIA_REL / "brand.css").unlink()  # apply's snapshot copy of it fails
    proc = _run_build_block(checkout, 'echo "npm $*"\n')
    assert proc.returncode == 1, proc.stdout
    assert "npm run" not in proc.stdout
    assert "Restored the checkout's own brand files" in proc.stdout
    assert "ERROR: extension build failed (exit 1)" in proc.stdout
    assert (checkout / MEDIA_REL / "brand.json").read_bytes() == b"stock brand.json"
    assert _git(checkout, "status", "--porcelain").splitlines() == [f" D {MEDIA_REL}/brand.css"]


def test_build_block_survives_a_failed_manifest_snapshot(checkout: Path) -> None:
    """apply failing on its first copy (no package.json) still restores, cleans up, exits 1."""
    _write_overlay(checkout, ("brand.json",))
    (checkout / MANIFEST_REL).unlink()
    proc = _run_build_block(checkout, 'echo "npm $*"\n')
    assert proc.returncode == 1, proc.stdout
    assert "npm run" not in proc.stdout
    assert "ERROR: extension build failed (exit 1)" in proc.stdout
    assert (checkout / MEDIA_REL / "brand.json").read_bytes() == b"stock brand.json"
    assert _git(checkout, "status", "--porcelain").splitlines() == [f" D {MANIFEST_REL}"]
    assert not list((checkout.parent / "tmp").iterdir()), "the snapshot directory is removed"


def test_build_block_without_overlay_fails_fast_on_copy_kiss(checkout: Path) -> None:
    """Stock checkout: no overlay messages, and a copy-kiss failure stops before packaging."""
    proc = _run_build_block(checkout, 'echo "npm $*"; [ "$2" != copy-kiss ] || exit 3\n')
    assert proc.returncode == 3, proc.stdout
    assert "brand overlay" not in proc.stdout and "Restored" not in proc.stdout
    assert "npm run package" not in proc.stdout
    assert "ERROR: extension build failed (exit 3)" in proc.stdout


def test_dot_brand_is_git_ignored_in_this_repository() -> None:
    """``.brand/`` never ends up in a commit of the kiss repository."""
    proc = subprocess.run(
        ["git", "check-ignore", "-q", ".brand/brand.json"],
        cwd=_REPO,
        capture_output=True,
    )
    assert proc.returncode == 0, ".gitignore must list /.brand/"
