# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression: reinstalling the extension while VS Code is open must not
reload the window into a half-installed extension — that is what makes the
chat webview come up blank.

Background — the bug
====================
When ``install.sh`` runs while VS Code is open, step [6/6] reinstalls the
extension with ``code --install-extension --force``.  For a *same-version*
reinstall VS Code first **deletes** the extension directory and then
re-extracts it, so ``out/extension.js`` is transiently missing (and, briefly,
present-but-partially-written) for a noticeable window.

The extension watches ``out/extension.js`` with a 2 s ``fs.watchFile`` poll to
auto-reload after a reinstall.  The original watcher fired
``workbench.action.reloadWindow`` the instant it saw *any* stat change — so a
poll landing inside the delete window reloaded the window against a
half-installed extension (its ``media`` chat resources gone) and before the
kiss-web daemon had been restarted.  The chat view rendered blank and the
extension appeared not to update.

The fix
=======
``src/reloadGuard.js`` exposes :func:`isReloadReady`, and ``extension.ts``
now defers the reload until the reinstall has fully settled — ``extension.js``
present, non-empty and size-stable across consecutive polls, and the kiss-web
daemon socket back — instead of reloading on the first transient stat change.

These tests genuinely execute:

* the real ``code`` CLI, to prove the same-version reinstall really does
  transiently remove ``out/extension.js`` (the precondition for the bug), and
* the real ``src/reloadGuard.js`` module via ``node``, to prove the guard
  refuses to reload during every transient state and only allows it once the
  reinstall has settled and the daemon socket is back.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
VSCODE_DIR = REPO / "src" / "kiss" / "agents" / "vscode"
RELOAD_GUARD_JS = VSCODE_DIR / "src" / "reloadGuard.js"

_EXT_ID = "ksentest.kissreloadtest"
_EXT_VERSION = "9.9.9"


def _find_code_cli() -> str | None:
    """Return an absolute path to the VS Code CLI, or ``None`` if absent."""
    candidates = [
        shutil.which("code"),
        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
        "/usr/local/bin/code",
        "/usr/bin/code",
        "/snap/bin/code",
    ]
    for c in candidates:
        if c and Path(c).exists() and os.access(c, os.X_OK):
            return c
    return None


def _build_vsix(dest: Path, marker: str) -> None:
    """Write a minimal installable VSIX whose ``out/extension.js`` is *marker*."""
    manifest = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<PackageManifest Version="2.0.0" '
        'xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">\n'
        "  <Metadata>\n"
        f'    <Identity Language="en-US" Id="kissreloadtest" '
        f'Version="{_EXT_VERSION}" Publisher="ksentest" />\n'
        "    <DisplayName>kissreloadtest</DisplayName>\n"
        "    <Description>test</Description>\n"
        "  </Metadata>\n"
        "  <Installation>\n"
        '    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>\n'
        "  </Installation>\n"
        "  <Assets>\n"
        '    <Asset Type="Microsoft.VisualStudio.Code.Manifest" '
        'Path="extension/package.json" Addressable="true" />\n'
        "  </Assets>\n"
        "</PackageManifest>\n"
    )
    package_json = json.dumps(
        {
            "name": "kissreloadtest",
            "publisher": "ksentest",
            "version": _EXT_VERSION,
            "engines": {"vscode": "^1.50.0"},
            "main": "./out/extension.js",
            "contributes": {},
        }
    )
    content_types = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="json" ContentType="application/json"/>'
        '<Default Extension="js" ContentType="application/javascript"/>'
        '<Default Extension="vsixmanifest" ContentType="text/xml"/>'
        '<Default Extension="xml" ContentType="text/xml"/></Types>\n'
    )
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("extension.vsixmanifest", manifest)
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("extension/package.json", package_json)
        z.writestr("extension/out/extension.js", marker)


class TestReinstallTransientWindow(unittest.TestCase):
    """The real ``code`` CLI must be shown to delete ``extension.js`` mid-reinstall."""

    def setUp(self) -> None:
        code = _find_code_cli()
        if not code:
            self.skipTest("VS Code CLI not available")
        self.code: str = code
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-reload-"))
        self.ext_dir = self.tmp / "extensions"
        self.ext_dir.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, vsix: Path) -> None:
        subprocess.run(
            [
                self.code,
                "--extensions-dir",
                str(self.ext_dir),
                "--install-extension",
                str(vsix),
                "--force",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )

    def test_same_version_reinstall_transiently_removes_extension_js(self) -> None:
        """A same-version ``--install-extension --force`` deletes the extension
        directory before re-extracting, leaving ``out/extension.js`` transiently
        missing.  A naive watcher polling during that window would reload the
        window against a half-installed extension — the blank-webview bug.
        """
        v1 = self.tmp / "v1.vsix"
        v2 = self.tmp / "v2.vsix"
        _build_vsix(v1, "MARKER_V1")
        _build_vsix(v2, "MARKER_V2")

        self._install(v1)
        ext_js = self.ext_dir / f"{_EXT_ID}-{_EXT_VERSION}" / "out" / "extension.js"
        self.assertTrue(ext_js.exists(), "first install must create extension.js")

        observed_missing = threading.Event()
        stop = threading.Event()

        def monitor() -> None:
            while not stop.is_set():
                if not ext_js.exists():
                    observed_missing.set()
                    return

        t = threading.Thread(target=monitor)
        t.start()
        try:
            self._install(v2)
        finally:
            stop.set()
            t.join(timeout=5)

        self.assertTrue(
            observed_missing.is_set(),
            "same-version reinstall should transiently remove out/extension.js; "
            "if it never goes missing the bug's precondition has changed.",
        )
        # And the reinstall really did update the file once it settled.
        self.assertEqual(ext_js.read_text(), "MARKER_V2")


class TestReloadGuardBehavior(unittest.TestCase):
    """Exercise the real ``reloadGuard.js`` via node across reinstall states."""

    def setUp(self) -> None:
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        self.node: str = node
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-guard-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ready(self, ext_js: Path, sock: Path, prev_size: int) -> dict:
        """Call ``isReloadReady`` in the real guard module and return its result."""
        script = (
            f"const g = require({json.dumps(str(RELOAD_GUARD_JS))});"
            f"const r = g.isReloadReady("
            f"{json.dumps(str(ext_js))}, {json.dumps(str(sock))}, {prev_size});"
            "process.stdout.write(JSON.stringify(r));"
        )
        out = subprocess.run(
            [self.node, "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        data = json.loads(out)
        assert isinstance(data, dict)
        return data

    def test_guard_blocks_reload_until_reinstall_settles(self) -> None:
        """Walk the exact filesystem states a reinstall passes through and assert
        the guard only permits a reload once everything has settled.
        """
        ext_js = self.tmp / "out" / "extension.js"
        ext_js.parent.mkdir(parents=True)
        sock = self.tmp / "sorcar.sock"

        # 1. Mid-delete: extension.js missing → never reload.
        r = self._ready(ext_js, sock, -1)
        self.assertFalse(r["ready"])
        self.assertEqual(r["size"], -1)

        # 2. Mid-write: file present but empty → never reload.
        ext_js.write_text("")
        r = self._ready(ext_js, sock, 0)
        self.assertFalse(r["ready"])
        self.assertEqual(r["size"], 0)

        # 3. Present but still growing (size changed since last poll) → wait.
        ext_js.write_text("partial-bundle")
        size = len("partial-bundle")
        r = self._ready(ext_js, sock, -1)  # prev != current size
        self.assertFalse(r["ready"])
        self.assertEqual(r["size"], size)

        # 4. Size stable but daemon socket not back yet → still wait, so the
        #    reloaded webview would have something to connect to.
        r = self._ready(ext_js, sock, size)
        self.assertFalse(r["ready"])

        # 5. Settled: stable non-empty file AND daemon socket present → reload.
        sock.write_text("")  # stand-in for the daemon's UDS socket
        r = self._ready(ext_js, sock, size)
        self.assertTrue(r["ready"])
        self.assertEqual(r["size"], size)


if __name__ == "__main__":
    unittest.main()
