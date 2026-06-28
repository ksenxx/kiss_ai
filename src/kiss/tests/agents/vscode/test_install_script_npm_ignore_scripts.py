# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: the Update button's ``install.sh`` must run ``npm ci`` with
``--ignore-scripts`` so the extension build cannot hang on dependency
install scripts.

Background — the bug
====================
Clicking the settings Update button runs ``~/kiss_ai/install.sh`` in an
integrated terminal.  Step ``[5/6] Building VS Code extension...`` ran a
plain ``npm ci``, which executes every dependency lifecycle (install /
postinstall) script.  The extension's lockfile contains exactly two packages
with install scripts:

* ``keytar`` — an *optional*, lazily-imported dependency of ``@vscode/vsce``
  used only for publish credential storage, never by ``vsce package``.  Its
  install script is ``prebuild-install || node-gyp rebuild``:
  ``prebuild-install`` downloads a prebuilt binary from the **archived**
  atom/node-keytar GitHub releases and the fallback compiles natively with
  node-gyp.  Either path can block forever (no timeout, no output) on
  network or toolchain trouble, freezing the update at::

      >>> [5/6] Building VS Code extension...
      npm warn deprecated prebuild-install@7.1.3: No longer maintained. ...
      ^C

* ``@vscode/vsce-sign`` — postinstall used only for VSIX *signing*, which
  ``vsce package`` never does.

Neither script is needed to compile and package the VSIX, so ``install.sh``
(and the release scripts, which build the same extension) now pass
``--ignore-scripts --no-audit --no-fund`` to ``npm ci``.

The behavioural test below executes ``npm ci`` with the exact flags parsed
out of ``install.sh`` against a throwaway project whose dependency carries a
marker-writing install script, proving lifecycle scripts cannot run (and
therefore cannot hang) during the update build.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
INSTALL_SCRIPT = REPO / "install.sh"
RELEASE_SCRIPTS = [
    REPO / "scripts" / "release.sh",
    REPO / "scripts" / "release_exp.sh",
]


def _npm_ci_line(script: Path) -> str:
    """Return the (single) ``npm ci`` invocation line of *script*."""
    lines = [
        line.strip()
        for line in script.read_text(encoding="utf-8").splitlines()
        if re.match(r"^\s*npm ci\b", line)
    ]
    assert len(lines) == 1, f"expected exactly one 'npm ci' line in {script}"
    return lines[0]


def test_install_sh_npm_ci_ignores_lifecycle_scripts() -> None:
    """``install.sh`` must build the extension with ``npm ci --ignore-scripts``.

    A plain ``npm ci`` runs keytar's ``prebuild-install || node-gyp rebuild``
    install script, which can hang the Update button forever at step [5/6].
    """
    line = _npm_ci_line(INSTALL_SCRIPT)
    assert "--ignore-scripts" in line, (
        "install.sh must pass --ignore-scripts to npm ci; without it the "
        "keytar/prebuild-install lifecycle script can hang the update at "
        "'[5/6] Building VS Code extension...'"
    )
    src = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert src.index(line) < src.index("\n    npm run compile"), (
        "npm ci --ignore-scripts must run before compiling the extension"
    )
    assert src.index("\n    npm run compile") < src.index("\n    npm run copy-kiss"), (
        "install.sh must compile before copying the bundled runtime"
    )
    assert src.index("\n    npm run copy-kiss") < src.index("\n    npm run package"), (
        "install.sh must copy the bundled runtime before packaging the VSIX"
    )


def test_release_scripts_npm_ci_ignore_lifecycle_scripts() -> None:
    """The release scripts build the same extension and need the same guard."""
    for script in RELEASE_SCRIPTS:
        line = _npm_ci_line(script)
        assert "--ignore-scripts" in line, (
            f"{script.name} must pass --ignore-scripts to npm ci "
            "(parity with install.sh)"
        )
        src = script.read_text(encoding="utf-8")
        assert src.index(line) < src.index("\n    npm run compile")
        assert src.index("\n    npm run compile") < src.index("\n    npm run copy-kiss")
        assert src.index("\n    npm run copy-kiss") < src.index("\n    npm run package")


def _write_dep_tarball(dest: Path) -> None:
    """Create ``evil-dep-1.0.0.tgz`` in *dest* with a marker install script.

    The dependency's install script writes ``$MARKER`` when executed — the
    stand-in for keytar's hang-capable ``prebuild-install`` script.
    """
    manifest = {
        "name": "evil-dep",
        "version": "1.0.0",
        "scripts": {
            "install": (
                "node -e \"require('fs')"
                ".writeFileSync(process.env.MARKER, 'ran')\""
            ),
        },
    }
    data = json.dumps(manifest).encode()
    with tarfile.open(dest / "evil-dep-1.0.0.tgz", "w:gz") as tar:
        info = tarfile.TarInfo("package/package.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def _run_npm_ci(flags: list[str], tmp_path: Path, name: str) -> Path:
    """Run ``npm ci`` with *flags* in a fresh project; return the marker path.

    The project depends on the local ``evil-dep`` tarball whose install
    script writes the returned marker file when lifecycle scripts execute.
    Everything resolves offline (``file:`` dependency only).
    """
    proj = tmp_path / name
    proj.mkdir()
    _write_dep_tarball(proj)
    (proj / "package.json").write_text(
        json.dumps(
            {
                "name": "update-build-fixture",
                "version": "1.0.0",
                "dependencies": {"evil-dep": "file:./evil-dep-1.0.0.tgz"},
            }
        ),
        encoding="utf-8",
    )
    marker = proj / "lifecycle-script-ran"
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)}
    npm = shutil.which("npm")
    assert npm is not None
    env["PATH"] = f"{Path(npm).parent}:{env['PATH']}"
    # Build the lockfile without installing anything (runs no scripts).
    subprocess.run(
        [npm, "install", "--package-lock-only", "--no-audit", "--no-fund"],
        cwd=proj,
        env={**env, "MARKER": str(marker)},
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [npm, "ci", *flags],
        cwd=proj,
        env={**env, "MARKER": str(marker)},
        check=True,
        capture_output=True,
        timeout=120,
    )
    assert (proj / "node_modules" / "evil-dep" / "package.json").is_file(), (
        "npm ci must still install the dependency tree"
    )
    return marker


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm not installed")
def test_update_build_npm_ci_flags_block_install_scripts(
    tmp_path: Path,
) -> None:
    """``npm ci`` with install.sh's exact flags must not run install scripts.

    First reproduces the original bug — a plain ``npm ci`` *does* execute the
    dependency's install script (the keytar hang vector) — then proves the
    flags parsed from ``install.sh`` prevent it.
    """
    # Reproduce the bug: without install.sh's flags the lifecycle script runs.
    marker = _run_npm_ci([], tmp_path, "buggy")
    assert marker.is_file(), (
        "fixture self-check: plain npm ci should run the install script"
    )

    # The fix: install.sh's flags must prevent the script from running.
    flags = _npm_ci_line(INSTALL_SCRIPT).split()[2:]
    marker = _run_npm_ci(flags, tmp_path, "fixed")
    assert not marker.exists(), (
        "npm ci with install.sh's flags executed a dependency install "
        "script — the Update button can hang again on keytar's "
        "prebuild-install/node-gyp"
    )
