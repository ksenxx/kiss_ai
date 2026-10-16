# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: ``install.sh`` applies ``.brand/`` to the installed app.

A white-label build keeps its brand files in an untracked ``.brand/``
directory of the repository.  ``copy_brand_overlay`` copies the ones that
exist over the stock media of the app directory (``$APP_DIR``) after
``copy_app_files``; the repository's own ``src/kiss/webapp/media`` is
never touched, so a checkout stays stock no matter which brand was
installed from it.

Each test extracts the ``kiss-app-files`` block of ``install.sh`` verbatim
and runs ``copy_brand_overlay`` under a real bash against a temporary
``PROJECT_DIR`` and ``APP_DIR``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

pytestmark = posix_only("runs the kiss-app-files block of install.sh under bash")

_REPO = Path(__file__).resolve().parents[3]
_INSTALL = _REPO / "install.sh"
_MEDIA = "src/kiss/webapp/media"


def _app_files_block() -> str:
    """Return the ``kiss-app-files`` block of ``install.sh``."""
    text = _INSTALL.read_text(encoding="utf-8")
    m = re.search(r"^# BEGIN: kiss-app-files.*?^# END: kiss-app-files$", text, re.S | re.M)
    assert m is not None, "install.sh no longer has the kiss-app-files block"
    return m.group(0)


def _media_files(root: Path) -> dict[str, str]:
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((root / _MEDIA).iterdir())
        if p.is_file()
    }


class _Install:
    """A temp repository with stock media and an app directory installed from it."""

    def __init__(self, tmp_path: Path) -> None:
        self.project = tmp_path / "project"
        self.app = tmp_path / "home" / ".kiss" / "kiss"
        for root in (self.project, self.app):
            (root / _MEDIA).mkdir(parents=True)
            for name in ("brand.json", "brand.css", "kiss-icon.svg", "thumbnail.jpeg"):
                (root / _MEDIA / name).write_text(f"stock {name}\n", encoding="utf-8")
            (root / _MEDIA / "other.css").write_text("stock other\n", encoding="utf-8")
        self.stock = _media_files(self.project)

    def brand(self, **files: str) -> None:
        """Create ``.brand/`` in the project with the given files."""
        overlay = self.project / ".brand"
        overlay.mkdir(exist_ok=True)
        for name, content in files.items():
            (overlay / name).write_text(content, encoding="utf-8")

    def run(self, command: str = "copy_brand_overlay") -> subprocess.CompletedProcess[str]:
        script = (
            f"set -eo pipefail\nPROJECT_DIR={str(self.project)!r}\nAPP_DIR={str(self.app)!r}\n"
            f"{_app_files_block()}\n{command}\n"
        )
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_overlay_files_replace_the_app_media_and_are_listed(tmp_path: Path) -> None:
    inst = _Install(tmp_path)
    inst.brand(**{"brand.json": '{"name": "Loop"}\n', "kiss-icon.svg": "<svg/>\n"})
    res = inst.run()
    assert res.returncode == 0, res.stderr
    assert res.stdout == (
        f"   Applying brand overlay from {inst.project}/.brand...\n"
        f"      {_MEDIA}/brand.json\n      {_MEDIA}/kiss-icon.svg\n"
    )
    installed = _media_files(inst.app)
    assert installed["brand.json"] == '{"name": "Loop"}\n'
    assert installed["kiss-icon.svg"] == "<svg/>\n"
    # Files the overlay does not provide keep their stock content.
    assert installed["brand.css"] == "stock brand.css\n"
    assert installed["thumbnail.jpeg"] == "stock thumbnail.jpeg\n"
    assert installed["other.css"] == "stock other\n"


def test_overlay_never_touches_the_repository_media(tmp_path: Path) -> None:
    inst = _Install(tmp_path)
    inst.brand(**{"brand.json": "branded\n", "brand.css": "branded\n"})
    res = inst.run()
    assert res.returncode == 0, res.stderr
    assert _media_files(inst.project) == inst.stock


def test_only_known_brand_files_are_copied(tmp_path: Path) -> None:
    """A stray file in ``.brand/`` (notes, a backup) is not installed."""
    inst = _Install(tmp_path)
    inst.brand(**{"brand.css": "branded\n", "README.md": "notes\n", "other.css": "evil\n"})
    res = inst.run()
    assert res.returncode == 0, res.stderr
    installed = _media_files(inst.app)
    assert installed["brand.css"] == "branded\n"
    assert installed["other.css"] == "stock other\n"
    assert "README.md" not in installed
    assert res.stdout.count(_MEDIA) == 1


def test_without_a_brand_directory_nothing_happens(tmp_path: Path) -> None:
    inst = _Install(tmp_path)
    res = inst.run("copy_brand_overlay; echo rc=$?")
    assert res.stdout == "rc=0\n"
    assert _media_files(inst.app) == inst.stock


def test_an_empty_brand_directory_only_announces_itself(tmp_path: Path) -> None:
    inst = _Install(tmp_path)
    inst.brand()
    res = inst.run()
    assert res.returncode == 0, res.stderr
    assert res.stdout == f"   Applying brand overlay from {inst.project}/.brand...\n"
    assert _media_files(inst.app) == inst.stock


def test_a_failed_copy_fails_the_step(tmp_path: Path) -> None:
    """``copy_brand_overlay`` returns non-zero when a file cannot be copied
    (here: the app media directory is missing), so the install stops
    instead of shipping a half-branded app."""
    inst = _Install(tmp_path)
    inst.brand(**{"brand.json": "branded\n"})
    (inst.app / _MEDIA).rename(inst.app / "src" / "kiss" / "webapp" / "gone")
    res = inst.run("copy_brand_overlay || echo rc=$?")
    assert res.stdout.endswith("rc=1\n"), res.stdout
    assert not (inst.app / _MEDIA).exists()


@pytest.mark.skipif(not (_REPO / ".git").exists(), reason="not a git checkout")
def test_dot_brand_is_git_ignored_in_this_repository() -> None:
    """``.brand/`` must never be committed: it is a per-build secret."""
    res = subprocess.run(
        ["git", "-C", str(_REPO), "check-ignore", "-q", ".brand/brand.json"],
        capture_output=True,
    )
    assert res.returncode == 0, ".brand/ is not git-ignored"
