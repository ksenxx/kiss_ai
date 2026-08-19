# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: ``DependencyInstaller.ts`` must NOT copy ``MODEL_INFO.json``.

Split out of ``kiss.tests.test_install_no_model_info_copy``: these two
tests depend only on the VS Code extension source
``src/kiss/agents/vscode/src/DependencyInstaller.ts``, so they belong
here; the ``install.sh`` (repo-root bootstrap) tests remain in the
root module.

The bundled ``src/kiss/core/models/MODEL_INFO.json`` is the runtime
source of truth for pricing/context tables and is read directly from
the installed package by ``kiss.core.models.model_info``.  Only
``MY_MODELS.json`` (a purely user-curated overrides/extensions file)
is lazily written into ``~/.kiss/`` at runtime, by
``ensure_user_asset_from_default``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]


def test_dependency_installer_does_not_copy_model_info_json() -> None:
    """``DependencyInstaller.ts`` must not actively copy MODEL_INFO.json."""
    di = (
        _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
        / "DependencyInstaller.ts"
    )
    assert di.exists(), f"DependencyInstaller.ts not found at {di}"
    text = di.read_text()
    for m in re.finditer(r"copyFileSync\(([^)]*)\)", text):
        args = m.group(1)
        assert "MODEL_INFO.json" not in args, (
            f"DependencyInstaller.ts still calls copyFileSync with "
            f"MODEL_INFO.json: {m.group(0)!r}"
        )
    assert (
        re.search(
            r"path\.join\(\s*LOG_DIR\s*,\s*['\"]MODEL_INFO\.json['\"]\s*\)",
            text,
        )
        is None
    ), "DependencyInstaller.ts still writes to LOG_DIR/MODEL_INFO.json"


def test_dependency_installer_explains_no_model_info_copy() -> None:
    """DependencyInstaller.ts keeps a comment explaining why no copy happens."""
    di = (
        _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
        / "DependencyInstaller.ts"
    )
    text = di.read_text()
    assert "MODEL_INFO.json" in text, (
        "DependencyInstaller.ts should retain a comment explaining "
        "that MODEL_INFO.json is intentionally NOT copied."
    )
