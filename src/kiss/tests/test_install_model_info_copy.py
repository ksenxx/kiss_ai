# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end test: ``install.sh`` MUST copy ``MODEL_INFO.json`` into ``$KISS_HOME``.

Every install/update refreshes ``$KISS_HOME/MODEL_INFO.json`` (default
``~/.kiss/MODEL_INFO.json``) from the bundled
``src/kiss/core/models/MODEL_INFO.json``.  An *installed* KISS Sorcar
reads its model catalog from that copy at runtime
(``kiss.core.models.model_info._select_catalog_path``), and the settings
panel's "Update Models" button refreshes the copy in place via
``kiss.scripts.update_models --model-info``.  Development checkouts
(project roots with a ``.git`` marker) keep reading the bundled file, so
the copy never shadows a checkout's own source of truth.

The copy is exercised for real: the relevant ``install.sh`` block is run
under bash against a temp ``PROJECT_DIR`` / ``KISS_HOME``.  The
``DependencyInstaller.ts`` (VS Code extension finalization step) twin
lives in ``kiss.tests.agents.vscode.test_install_model_info_copy``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _install_sh_text() -> str:
    install_sh = _REPO / "install.sh"
    assert install_sh.exists(), f"install.sh not found at {install_sh}"
    return install_sh.read_text()


def _model_info_copy_block(text: str) -> str:
    """Return the MODEL_INFO copy block of ``install.sh``.

    The block starts at the ``MODEL_INFO_SRC=`` assignment and ends just
    before the following blank-line-separated section.
    """
    m = re.search(r"^\s*MODEL_INFO_SRC=.*?(?=\n\s*\n)", text, re.S | re.M)
    assert m is not None, (
        "install.sh no longer contains the MODEL_INFO_SRC=... copy block "
        "that refreshes $KISS_HOME/MODEL_INFO.json on every install"
    )
    return m.group(0)


def test_install_sh_copies_model_info_json_for_real(tmp_path: Path) -> None:
    """Running the copy block populates ``$KISS_HOME/MODEL_INFO.json``."""
    block = _model_info_copy_block(_install_sh_text())
    project_dir = tmp_path / "project"
    src = project_dir / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    src.parent.mkdir(parents=True)
    payload = {"vendor/model": {"context_length": 1}}
    src.write_text(json.dumps(payload), encoding="utf-8")
    kiss_home = tmp_path / "kiss-home"

    script = f'PROJECT_DIR="{project_dir}"\nKISS_HOME="{kiss_home}"\n{block}\n'
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    dst = kiss_home / "MODEL_INFO.json"
    assert dst.exists(), proc.stdout
    assert json.loads(dst.read_text(encoding="utf-8")) == payload
    assert str(dst) in proc.stdout


def test_install_sh_copy_block_overwrites_a_stale_user_copy(
    tmp_path: Path,
) -> None:
    """A pre-existing (stale) user copy is refreshed, not preserved."""
    block = _model_info_copy_block(_install_sh_text())
    project_dir = tmp_path / "project"
    src = project_dir / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    src.parent.mkdir(parents=True)
    src.write_text('{"fresh/model": {}}', encoding="utf-8")
    kiss_home = tmp_path / "kiss-home"
    kiss_home.mkdir()
    dst = kiss_home / "MODEL_INFO.json"
    dst.write_text('{"stale/model": {}}', encoding="utf-8")

    script = f'PROJECT_DIR="{project_dir}"\nKISS_HOME="{kiss_home}"\n{block}\n'
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert dst.read_text(encoding="utf-8") == '{"fresh/model": {}}'


def test_install_sh_copy_block_survives_a_missing_source(
    tmp_path: Path,
) -> None:
    """A missing bundled catalog warns instead of failing the install."""
    block = _model_info_copy_block(_install_sh_text())
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    kiss_home = tmp_path / "kiss-home"

    script = f'PROJECT_DIR="{project_dir}"\nKISS_HOME="{kiss_home}"\n{block}\n'
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stdout
    assert not (kiss_home / "MODEL_INFO.json").exists()


def test_install_sh_copy_defaults_kiss_home_to_dot_kiss(
    tmp_path: Path,
) -> None:
    """With ``KISS_HOME`` unset, the copy lands in ``$HOME/.kiss``."""
    block = _model_info_copy_block(_install_sh_text())
    project_dir = tmp_path / "project"
    src = project_dir / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    src.parent.mkdir(parents=True)
    src.write_text("{}", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()

    script = f'PROJECT_DIR="{project_dir}"\nunset KISS_HOME\n{block}\n'
    proc = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        capture_output=True,
        text=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert (home / ".kiss" / "MODEL_INFO.json").exists()
