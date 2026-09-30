# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The VSIX never packages ``*.kiss-rescued-*`` collision siblings.

The worktree teardown rescue (``GitWorktreeOps.rescue_ignored_files`` in
git_worktree.py) copies git-ignored task output back into the main
checkout; when the main tree already holds a different file of the same
name, the worktree copy lands beside it as ``<stem>.kiss-rescued-<ns><ext>``
(older runs: ``<name>.kiss-rescued-<ns>``).  A rebuilt ``kiss-sorcar.vsix``
collides on every install, so ``kiss-sorcar.vsix.kiss-rescued-<ns>``
files accumulated next to the extension manifest and — since that shape
no longer matches ``*.vsix`` — vsce packaged them into the VSIX.

This test runs the real ``vsce ls`` against a throwaway extension folder
that uses the shipped ``.vscodeignore`` and contains siblings of every
shape the rescue produces, and asserts none of them is listed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kiss.tests.conftest import posix_only

pytestmark = posix_only("vsce ls is exercised through the node_modules shim")

VSCODE_DIR = Path(__file__).resolve().parents[5] / "src" / "kiss" / "agents" / "vscode"
VSCE = VSCODE_DIR / "node_modules" / ".bin" / "vsce"

RESCUED_SIBLINGS = (
    "kiss-sorcar.vsix.kiss-rescued-1788470990563603656",
    "kiss-sorcar.kiss-rescued-1788481044058298118.vsix",
    "media/icon.kiss-rescued-1788481044058298119.png",
    "kiss_project/.env.kiss-rescued-1788481044058298120",
)


def _make_extension(tmp_path: Path) -> Path:
    """Create a minimal packable extension that uses the shipped .vscodeignore."""
    ext = tmp_path / "ext"
    (ext / "out").mkdir(parents=True)
    (ext / "media").mkdir()
    (ext / "kiss_project").mkdir()
    shutil.copy(VSCODE_DIR / ".vscodeignore", ext / ".vscodeignore")
    (ext / "package.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "publisher": "demo",
                "version": "0.0.1",
                "engines": {"vscode": "^1.90.0"},
                "main": "./out/extension.js",
                "activationEvents": ["onStartupFinished"],
            }
        )
        + "\n"
    )
    (ext / "LICENSE").write_text("MIT\n")
    (ext / "out" / "extension.js").write_text("exports.activate = () => {};\n")
    (ext / "media" / "icon.png").write_bytes(b"png")
    for rel in RESCUED_SIBLINGS:
        (ext / rel).write_bytes(b"stale rescue copy")
    return ext


def test_vsce_ls_omits_rescued_siblings(tmp_path: Path) -> None:
    """`vsce ls` lists the runtime files but no `*.kiss-rescued-*` sibling."""
    if not VSCE.exists():
        pytest.skip("vsce is not installed (run npm ci in src/kiss/agents/vscode)")
    ext = _make_extension(tmp_path)
    result = subprocess.run([str(VSCE), "ls"], cwd=ext, capture_output=True, text=True, check=True)
    listed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert "out/extension.js" in listed, result.stdout
    assert "media/icon.png" in listed, result.stdout
    leaked = [line for line in listed if ".kiss-rescued-" in line]
    assert leaked == [], f"rescue siblings would be packaged: {leaked}"
