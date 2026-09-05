# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: ``DependencyInstaller.ts`` MUST copy ``MODEL_INFO.json``.

The VS Code extension's finalization step refreshes
``kissHomeDir()/MODEL_INFO.json`` from the bundled
``kiss_project/src/kiss/core/models/MODEL_INFO.json`` on every
install/update: an *installed* KISS Sorcar reads its model catalog from
that copy at runtime
(``kiss.core.models.model_info._select_catalog_path``), and the settings
panel's "Update Models" button refreshes the copy in place via
``kiss.scripts.update_models --model-info``.

The copy logic is TypeScript running inside the extension host, so it is
exercised for real by executing the extracted copy statements under
Node.js against temp directories; the ``install.sh`` twin lives in
``kiss.tests.test_install_model_info_copy``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]

_DI = (
    _REPO / "src" / "kiss" / "agents" / "vscode" / "src"
    / "DependencyInstaller.ts"
)


def _copy_snippet() -> str:
    """Return the MODEL_INFO.json copy try-block from DependencyInstaller.ts.

    Located by the ``modelInfoSrc`` declaration; the surrounding
    ``try { ... } catch`` is included so the real error handling runs too.
    """
    text = _DI.read_text()
    m = re.search(
        r"try \{\s*\n\s*const modelInfoSrc[\s\S]*?\n  \}\n", text
    )
    assert m is not None, (
        "DependencyInstaller.ts no longer contains the modelInfoSrc "
        "copy block that refreshes kissHomeDir()/MODEL_INFO.json"
    )
    return m.group(0)


def _run_snippet(tmp_path: Path, kiss_project: Path, kiss_home: Path) -> str:
    """Execute the extracted copy block under Node with real fs/path."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - CI always has node
        pytest.skip("node not installed")
    harness = f"""
const fs = require('fs');
const path = require('path');
const logs = [];
const log = (m) => logs.push(String(m));
const kissProjectPath = {json.dumps(str(kiss_project))};
const kissHomeDir = () => {json.dumps(str(kiss_home))};
{_copy_snippet()}
console.log(logs.join('\\n'));
"""
    script = tmp_path / "copy_model_info.js"
    script.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_dependency_installer_copies_model_info_json(tmp_path: Path) -> None:
    """The copy block populates ``kissHomeDir()/MODEL_INFO.json``."""
    kiss_project = tmp_path / "kiss_project"
    src = kiss_project / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    src.parent.mkdir(parents=True)
    payload = {"vendor/model": {"context_length": 1}}
    src.write_text(json.dumps(payload), encoding="utf-8")
    kiss_home = tmp_path / "kiss-home"

    out = _run_snippet(tmp_path, kiss_project, kiss_home)

    dst = kiss_home / "MODEL_INFO.json"
    assert dst.exists(), out
    assert json.loads(dst.read_text(encoding="utf-8")) == payload
    assert "Copied MODEL_INFO.json" in out


def test_dependency_installer_copy_overwrites_a_stale_user_copy(
    tmp_path: Path,
) -> None:
    """A pre-existing (stale) user copy is refreshed, not preserved."""
    kiss_project = tmp_path / "kiss_project"
    src = kiss_project / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    src.parent.mkdir(parents=True)
    src.write_text('{"fresh/model": {}}', encoding="utf-8")
    kiss_home = tmp_path / "kiss-home"
    kiss_home.mkdir()
    dst = kiss_home / "MODEL_INFO.json"
    dst.write_text('{"stale/model": {}}', encoding="utf-8")

    _run_snippet(tmp_path, kiss_project, kiss_home)

    assert dst.read_text(encoding="utf-8") == '{"fresh/model": {}}'


def test_dependency_installer_copy_failure_is_logged_not_thrown(
    tmp_path: Path,
) -> None:
    """A missing bundled catalog logs a failure instead of raising."""
    kiss_project = tmp_path / "kiss_project"
    kiss_project.mkdir()
    kiss_home = tmp_path / "kiss-home"

    out = _run_snippet(tmp_path, kiss_project, kiss_home)

    assert "Failed to copy MODEL_INFO.json" in out
    assert not (kiss_home / "MODEL_INFO.json").exists()
