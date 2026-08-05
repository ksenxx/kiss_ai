# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: the installer must NOT copy ``INJECTIONS.md`` into ``~/.kiss/``.

Two install entry points must skip the copy:

* ``install.sh``       (bash bootstrap, runs before VS Code launches)
* ``DependencyInstaller.ts`` (VS Code extension finalization step)

These tests grep the source files for any path that would write
``$KISS_HOME/INJECTIONS.md`` or ``~/.kiss/INJECTIONS.md``.  The
bundled ``src/kiss/INJECTIONS.md`` is the runtime source of truth and
is read directly from the package; only ``MY_INJECTION.md`` (a purely
user-curated file) is written into ``~/.kiss/`` at runtime, lazily by
``ensure_user_asset_from_default``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def test_install_sh_does_not_copy_injections_md() -> None:
    """``install.sh`` must not ``cp ... INJECTIONS.md`` into ``~/.kiss/``."""
    install_sh = _REPO / "install.sh"
    assert install_sh.exists(), f"install.sh not found at {install_sh}"
    text = install_sh.read_text()
    forbidden_patterns = [
        r'cp\s+["\']?\$INJECTIONS_SRC["\']?\s+["\']?\$INJECTIONS_DST["\']?',
        r'cp\s+\S+\s+\S*\.kiss/INJECTIONS\.md',
        r'INJECTIONS_DST\s*=\s*["\']?\$KISS_HOME_DIR/INJECTIONS\.md',
    ]
    for pattern in forbidden_patterns:
        m = re.search(pattern, text)
        assert m is None, (
            f"install.sh still contains an INJECTIONS.md copy "
            f"matching {pattern!r}: {m.group(0) if m else ''!r}"
        )


def test_dependency_installer_does_not_copy_injections_md() -> None:
    """``DependencyInstaller.ts`` must not include INJECTIONS.md in install assets."""
    di = (
        _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
        / "DependencyInstaller.ts"
    )
    assert di.exists(), f"DependencyInstaller.ts not found at {di}"
    text = di.read_text()
    if "installMarkdownAssets" in text:
        for m in re.finditer(r"copyFileSync\(([^)]*)\)", text):
            args = m.group(1)
            assert "INJECTIONS.md" not in args, (
                f"DependencyInstaller.ts still calls "
                f"copyFileSync with INJECTIONS.md: {m.group(0)!r}"
            )
    assert (
        re.search(
            r"path\.join\(\s*kissHomeDir\s*,\s*['\"]INJECTIONS\.md['\"]\s*\)",
            text,
        )
        is None
    ), "DependencyInstaller.ts still writes to kissHomeDir/INJECTIONS.md"


def test_install_sh_explains_no_injections_copy() -> None:
    """A comment in install.sh explains why INJECTIONS.md is no longer copied.

    A future maintainer must not "fix" the missing copy by re-adding it.
    The narrative comment is the canonical rationale.
    """
    install_sh = _REPO / "install.sh"
    text = install_sh.read_text()
    assert "INJECTIONS.md" in text, (
        "install.sh should retain a comment explaining that "
        "INJECTIONS.md is intentionally NOT copied (bundled file is "
        "read directly at runtime)."
    )


def test_dependency_installer_explains_no_injections_copy() -> None:
    """A comment in DependencyInstaller.ts explains why no copy happens."""
    di = (
        _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
        / "DependencyInstaller.ts"
    )
    text = di.read_text()
    assert "INJECTIONS.md" in text, (
        "DependencyInstaller.ts should retain a comment explaining "
        "that INJECTIONS.md is intentionally NOT copied."
    )
