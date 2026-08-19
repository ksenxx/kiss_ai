# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: ``DependencyInstaller.ts`` must NOT copy ``INJECTIONS.md``.

Split out of ``kiss.tests.test_install_no_injections_copy``: these two
tests depend only on the VS Code extension source
``src/kiss/agents/vscode/src/DependencyInstaller.ts``, so they belong
here; the ``install.sh`` (repo-root bootstrap) tests remain in the
root module.

The bundled ``src/kiss/INJECTIONS.md`` is the runtime source of truth
and is read directly from the package; only ``MY_INJECTION.md`` (a
purely user-curated file) is written into ``~/.kiss/`` at runtime,
lazily by ``ensure_user_asset_from_default``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]


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
